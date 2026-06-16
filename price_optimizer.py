"""
step2_optimize_pricing_rf.py
----------------------------
Two-stage Random Forest optimizer adapted for the time-series
FreshMark synthetic dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class PeakShelfOptimizer:
    def __init__(self):
        self.baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.uplift_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        # Core features matching the new generate_data.py schema
        self.base_features = [
            'product_encoded', 'mrp', 'stock_qty', 
            'days_to_expiry', 'is_weekend', 'units_sold_7d_avg'
        ]
        
    def load_and_preprocess(self, filepath):
        df = pd.read_csv(filepath)
        
        self.label_encoders['product_name'] = LabelEncoder()
        df['product_encoded'] = self.label_encoders['product_name'].fit_transform(df['product_name'])
        
        return df

    def train_models(self, df):
        print("Training models... This may take a moment.")
        
        # ---------------------------------------------------------
        # STAGE 1: Train Baseline Model (Discount == 0)
        # ---------------------------------------------------------
        baseline_data = df[df['discount_pct'] == 0].copy()
        
        X_base = baseline_data[self.base_features]
        y_base = baseline_data['units_sold_today']
        
        self.baseline_model.fit(X_base, y_base)
        print("Stage 1: Baseline Demand Model Trained")
        
        # ---------------------------------------------------------
        # STAGE 2: Train Uplift Model (Discount > 0)
        # ---------------------------------------------------------
        discount_data = df[df['discount_pct'] > 0].copy()
        X_uplift_base = discount_data[self.base_features]
        
        # Predict baseline for the discounted rows
        discount_data['estimated_baseline'] = self.baseline_model.predict(X_uplift_base)
        
        # Uplift = Actual - Expected Baseline (clipped at 0 to avoid negative noise)
        discount_data['uplift'] = (discount_data['units_sold_today'] - discount_data['estimated_baseline']).clip(lower=0)
        
        # Add discount to features for uplift prediction
        uplift_features_cols = self.base_features + ['discount_pct']
        X_uplift = discount_data[uplift_features_cols]
        y_uplift = discount_data['uplift']
        
        self.uplift_model.fit(X_uplift, y_uplift)
        print("Stage 2: Promotional Uplift Model Trained")

    def find_optimal_discount(self, product_name, cost, mrp, days_to_expiry, stock, is_weekend, avg_7d_sales):
        """
        Evaluates discrete discount steps to maximize profit.
        """
        prod_encoded = self.label_encoders['product_name'].transform([product_name])[0]
        
        test_discounts = [0, 5, 10, 20, 30, 40]
        
        best_profit = -np.inf
        best_discount = 0
        expected_sales_at_best = 0
        
        # 1. Predict baseline demand
        base_df = pd.DataFrame({
            'product_encoded': [prod_encoded],
            'mrp': [mrp],
            'stock_qty': [stock],
            'days_to_expiry': [days_to_expiry],
            'is_weekend': [is_weekend],
            'units_sold_7d_avg': [avg_7d_sales]
        })
        pred_baseline = self.baseline_model.predict(base_df)[0]
        
        print(f"\n--- Optimizing markdown for {stock} units of {product_name} ({days_to_expiry} days to expiry) ---")
        
        for discount in test_discounts:
            # 2. Predict Uplift
            if discount == 0:
                pred_uplift = 0
            else:
                uplift_df = base_df.copy()
                uplift_df['discount_pct'] = discount
                pred_uplift = self.uplift_model.predict(uplift_df)[0]
                
            # Cap total sales at available stock
            total_pred_sales = min(pred_baseline + pred_uplift, stock)
            
            # 3. Calculate Financials
            selling_price = mrp * (1 - (discount / 100))
            total_revenue = total_pred_sales * selling_price
            
            # Assume unsold units expire tomorrow and become total loss
            total_cost = stock * cost 
            projected_profit = total_revenue - total_cost
            
            print(f"Discount {discount:2d}% | Selling Price: ₹{selling_price:5.2f} | Est. Sales: {total_pred_sales:4.1f}/{stock} | Profit: ₹{projected_profit:6.2f}")
            
            if projected_profit > best_profit:
                best_profit = projected_profit
                best_discount = discount
                expected_sales_at_best = total_pred_sales
                
        return best_discount, best_profit, expected_sales_at_best

if __name__ == "__main__":
    optimizer = PeakShelfOptimizer()
    df = optimizer.load_and_preprocess('sales.csv')
    optimizer.train_models(df)
    
    # Scenario: 20 packets of Milk left, 1 day to expiry. Cost 58, MRP 66.
    optimal_disc, max_profit, expected_sales = optimizer.find_optimal_discount(
        product_name='Milk 1L', 
        cost=58.0, 
        mrp=66.0, 
        days_to_expiry=1, 
        stock=20,
        is_weekend=0,          # E.g., it's a Monday
        avg_7d_sales=18.5      # Typical sales velocity
    )
    
    print("\n" + "="*50)
    print(f"RECOMMENDATION: Apply a {optimal_disc}% discount.")
    print(f"Expected to sell {expected_sales:.1f} units, yielding a profit of ₹{max_profit:.2f}.")
    print("="*50)