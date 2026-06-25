import pandas as pd
import numpy as np
import lightgbm as lgb
import pulp
import joblib
import os

class HybridPricingOptimizer:
    def __init__(self):
        # T-Learner Models
        self.model_0 = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42
        )
        self.model_1 = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42
        )
        
        self.features_m0 = ['product_name', 'mrp', 'stock_qty', 'days_to_expiry', 'is_weekend', 'units_sold_7d_avg']
        # Model 1 also needs to know the size of the treatment (discount_pct)
        self.features_m1 = self.features_m0 + ['discount_pct']
        self.categorical_features = ['product_name']

    def load_and_preprocess(self, filepath):
        df = pd.read_csv(filepath)
        # LightGBM handles categoricals natively, but they must be explicitly typed as 'category'
        df['product_name'] = df['product_name'].astype('category')
        return df

    def train_causal_t_learner(self, df):
        print("Training LightGBM T-Learner...")
        
        # 1. Split Data into Control and Treatment
        df_control = df[df['discount_pct'] == 0].copy()
        df_treatment = df[df['discount_pct'] > 0].copy()
        
        # 2. Train Model 0 (Control)
        X_control = df_control[self.features_m0]
        y_control = df_control['units_sold_today']
        self.model_0.fit(
            X_control, y_control,
            categorical_feature=self.categorical_features
        )
        print("    Model 0 (Control - Base Demand) Trained")
        
        # 3. Train Model 1 (Treatment)
        X_treatment = df_treatment[self.features_m1]
        y_treatment = df_treatment['units_sold_today']
        self.model_1.fit(
            X_treatment, y_treatment,
            categorical_feature=self.categorical_features
        )
        print("    Model 1 (Treatment - Promoted Demand) Trained\n")

    def save_pipeline(self, folder_path="models"):
        """Saves the trained T-Learner models to disk."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        print("Saving trained LightGBM models to disk...")
        joblib.dump(self.model_0, f"{folder_path}/model_0_control.joblib")
        joblib.dump(self.model_1, f"{folder_path}/model_1_treatment.joblib")
        print(f"Models saved successfully in '{folder_path}/' directory!\n")

    def load_pipeline(self, folder_path="models"):
        """Loads previously trained models from disk."""
        print("Loading pre-trained LightGBM models...")
        try:
            self.model_0 = joblib.load(f"{folder_path}/model_0_control.joblib")
            self.model_1 = joblib.load(f"{folder_path}/model_1_treatment.joblib")
            print("Models loaded and ready for prediction!\n")
        except FileNotFoundError:
            print(f"Error: Model files not found in '{folder_path}/'. You need to train them first.\n")
            raise

    def predict_causal_demand(self, product, mrp, stock, expiry, is_weekend, avg_7d, discount):
        """Calculates expected demand using the T-Learner logic."""
        base_df = pd.DataFrame({
            'product_name': [product], 'mrp': [mrp], 'stock_qty': [stock],
            'days_to_expiry': [expiry], 'is_weekend': [is_weekend], 'units_sold_7d_avg': [avg_7d]
        })
        base_df['product_name'] = base_df['product_name'].astype('category')

        # Baseline Demand (mu_0)
        base_demand = self.model_0.predict(base_df)[0]
        
        if discount == 0:
            return max(0, base_demand), 0.0
            
        # Treatment Demand (mu_1)
        treat_df = base_df.copy()
        treat_df['discount_pct'] = discount
        treatment_demand = self.model_1.predict(treat_df)[0]
        
        # Causal Uplift (tau = mu_1 - mu_0)
        uplift = treatment_demand - base_demand
        
        return max(0, treatment_demand), max(0, uplift)

    def optimize_store_portfolio(self, inventory_df, constraints):
        """
        Uses PuLP to find the globally optimal discount strategy across the whole store,
        respecting business constraints.
        """
        print("Running PuLP Operations Research Solver...")
        allowed_discounts = [0, 5, 10, 20, 30, 40]
        
        # 1. Pre-compute the ML predictions for all combinations
        scenarios = []
        for _, row in inventory_df.iterrows():
            item_id = row['item_id'] # Unique identifier for the batch
            for d in allowed_discounts:
                # Get ML predictions
                total_demand, uplift = self.predict_causal_demand(
                    row['product_name'], row['mrp'], row['stock_qty'], 
                    row['days_to_expiry'], row['is_weekend'], row['units_sold_7d_avg'], d
                )
                
                # Calculate Business Metrics
                sales = min(total_demand, row['stock_qty']) # Can't sell more than stock
                waste = row['stock_qty'] - sales
                selling_price = row['mrp'] * (1 - (d / 100))
                revenue = sales * selling_price
                cost = row['stock_qty'] * row['cost']
                profit = revenue - cost
                
                scenarios.append({
                    'item_id': item_id, 'product_name': row['product_name'], 'discount': d,
                    'profit': profit, 'waste': waste, 'sales': sales
                })
                
        scenario_df = pd.DataFrame(scenarios)

        # 2. Initialize the Optimization Problem
        prob = pulp.LpProblem("Maximize_Retail_Profit", pulp.LpMaximize)
        
        # Create binary decision variables: e.g., x_('batch1', 10) = 1 means batch 1 gets 10% discount
        decision_vars = pulp.LpVariable.dicts(
            "Select",
            ((row.item_id, row.discount) for _, row in scenario_df.iterrows()),
            cat='Binary'
        )
        
        # 3. Define the Objective Function (Maximize Total Profit)
        prob += pulp.lpSum(
            decision_vars[row.item_id, row.discount] * row.profit 
            for _, row in scenario_df.iterrows()
        ), "Total_Profit"
        
        # 4. Define Constraints
        
        # Constraint A: Every item batch must have exactly ONE discount level chosen
        for item_id in inventory_df['item_id']:
            prob += pulp.lpSum(decision_vars[item_id, d] for d in allowed_discounts) == 1
            
        # Constraint B: Global maximum waste allowed across all items being optimized
        prob += pulp.lpSum(
            decision_vars[row.item_id, row.discount] * row.waste 
            for _, row in scenario_df.iterrows()
        ) <= constraints['max_total_waste'], "Max_Waste_Allowed"
        
        # Constraint C: Custom Item-specific rules (e.g., Milk max discount is 10%)
        for _, row in scenario_df.iterrows():
            if "Milk" in row['product_name'] and row['discount'] > constraints['milk_max_discount']:
                 # Force the decision variable to 0 (disallow this combination)
                 prob += decision_vars[row.item_id, row.discount] == 0

        # 5. Solve the system
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # 6. Extract Results
        results = []
        total_opt_profit = 0
        total_opt_waste = 0
        
        for _, row in scenario_df.iterrows():
            if pulp.value(decision_vars[row.item_id, row.discount]) == 1:
                results.append({
                    'Product': row['product_name'],
                    'Discount (%)': row['discount'],
                    'Est. Sales': round(row['sales'], 1),
                    'Est. Waste': round(row['waste'], 1),
                    'Projected Profit (₹)': round(row['profit'], 2)
                })
                total_opt_profit += row['profit']
                total_opt_waste += row['waste']
                
        results_df = pd.DataFrame(results)
        return results_df, total_opt_profit, total_opt_waste


if __name__ == "__main__":
    optimizer = HybridPricingOptimizer()
    
    # 1. SMART LOADING OR TRAINING
    # Check if models already exist in the "models" folder
    if os.path.exists("models/model_0_control.joblib") and os.path.exists("models/model_1_treatment.joblib"):
        optimizer.load_pipeline()
    else:
        print("First run detected. Initiating training sequence...\n")
        try:
            df = optimizer.load_and_preprocess('data/freshmark_sales.csv')
            optimizer.train_causal_t_learner(df)
            optimizer.save_pipeline()
        except FileNotFoundError:
            print("Error: 'data/freshmark_sales.csv' not found. Please run your data generator script first.")
            exit()
    
    # 2. End-of-Day Store Optimization Scenario
    # Let's say it's evening, and the shopkeeper has these 3 batches of items expiring tomorrow
    current_inventory = pd.DataFrame([
        {'item_id': 'b1', 'product_name': 'Milk 1L (Toned)', 'mrp': 66, 'cost': 58, 'stock_qty': 15, 'days_to_expiry': 1, 'is_weekend': 0, 'units_sold_7d_avg': 30},
        {'item_id': 'b2', 'product_name': 'Coriander (Bunch)', 'mrp': 15, 'cost': 8, 'stock_qty': 20, 'days_to_expiry': 1, 'is_weekend': 0, 'units_sold_7d_avg': 22},
        {'item_id': 'b3', 'product_name': 'White Bread',       'mrp': 45, 'cost': 32, 'stock_qty': 10, 'days_to_expiry': 1, 'is_weekend': 0, 'units_sold_7d_avg': 14}
    ])
    
    # Define Business Constraints
    constraints = {
        'max_total_waste': 5,        # Shopkeeper refuses to throw away more than 5 items total across these batches
        'milk_max_discount': 10      # Milk margins are too thin to discount more than 10%
    }
    
    # Run Optimization
    optimal_plan, total_profit, total_waste = optimizer.optimize_store_portfolio(current_inventory, constraints)
    
    print("="*60)
    print("OPTIMAL PRICING PORTFOLIO GENERATED")
    print("="*60)
    print(optimal_plan.to_string(index=False))
    print("-" * 60)
    print(f"Total Projected Profit: ₹{total_profit:.2f}")
    print(f"Total Projected Waste:  {total_waste:.1f} units")
    print("="*60)