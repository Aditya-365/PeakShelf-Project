"""
hybrid_model.py  (v2 – FreshMark)
-----------------------------------
T-Learner causal model + PuLP portfolio optimizer.

Compatible with freshmark_sales.csv v2 column names:
  - days_remaining   (was: days_to_expiry)
  - cost_price       (was: cost)
  - category         (new)
  - shelf_life_days  (new — used in urgency-aware constraint logic)
  - expiry_date      (new datetime column)

T-Learner design:
  model_0  trained on undiscounted rows  → predicts baseline demand
  model_1  trained on discounted rows    → predicts demand under a given discount
  uplift   = model_1(x, discount) − model_0(x)

PuLP optimizer:
  Decides ONE discount level per inventory batch to maximise total store profit,
  subject to:
    A) Exactly one discount chosen per batch (one-hot)
    B) Global waste cap across all batches
    C) Category-aware max-discount caps (e.g. Dairy ≤ 10 %)
    D) Urgency floor — high-urgency items are forced to discount ≥ a minimum
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pulp
import joblib


class HybridPricingOptimizer:

    def __init__(self):
        # T-Learner: two separate LightGBM regressors
        self.model_0 = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42
        )
        self.model_1 = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42
        )

        # Features for Model 0 (control — no discount signal)
        # days_remaining replaces the old days_to_expiry
        self.features_m0 = [
            'product_name', 'category', 'mrp', 'stock_qty',
            'days_remaining', 'shelf_life_days', 'is_weekend', 'units_sold_7d_avg'
        ]
        # Model 1 also sees the discount size (the "treatment" variable)
        self.features_m1 = self.features_m0 + ['discount_pct']

        self.categorical_features = ['product_name', 'category']

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """
        Loads the v2 CSV and prepares it for training.
        LightGBM handles categoricals natively when typed as 'category'.
        """
        df = pd.read_csv(filepath, parse_dates=["date", "expiry_date"])
        df['product_name'] = df['product_name'].astype('category')
        df['category']     = df['category'].astype('category')
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_causal_t_learner(self, df: pd.DataFrame):
        """
        Trains two separate models:
          Model 0 on control rows (discount_pct == 0) → baseline demand
          Model 1 on treatment rows (discount_pct > 0) → promoted demand
        """
        print("Training LightGBM T-Learner...")

        df_control   = df[df['discount_pct'] == 0].copy()
        df_treatment = df[df['discount_pct'] > 0].copy()

        print(f"    Control rows  : {len(df_control):,}")
        print(f"    Treatment rows: {len(df_treatment):,}")

        # Model 0 — baseline demand (no discount features)
        self.model_0.fit(
            df_control[self.features_m0],
            df_control['units_sold_today'],
            categorical_feature=self.categorical_features
        )
        print("    Model 0 (Control – Base Demand) trained")

        # Model 1 — demand under discount
        self.model_1.fit(
            df_treatment[self.features_m1],
            df_treatment['units_sold_today'],
            categorical_feature=self.categorical_features
        )
        print("    Model 1 (Treatment – Promoted Demand) trained\n")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pipeline(self, folder_path: str = "models"):
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.model_0, f"{folder_path}/model_0_control.joblib")
        joblib.dump(self.model_1, f"{folder_path}/model_1_treatment.joblib")
        print(f"Models saved to '{folder_path}/'\n")

    def load_pipeline(self, folder_path: str = "models"):
        try:
            self.model_0 = joblib.load(f"{folder_path}/model_0_control.joblib")
            self.model_1 = joblib.load(f"{folder_path}/model_1_treatment.joblib")
            print("Pre-trained models loaded and ready.\n")
        except FileNotFoundError:
            print(f"Error: model files not found in '{folder_path}/'. Train first.\n")
            raise

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_causal_demand(
        self,
        product: str,
        category: str,
        mrp: float,
        stock: int,
        days_remaining: int,
        shelf_life_days: int,
        is_weekend: int,
        avg_7d: float,
        discount: float,
    ) -> tuple[float, float]:
        """
        Returns (expected_units_sold, causal_uplift).

        Uses the T-Learner logic:
          mu_0 = model_0.predict(x)          — baseline demand
          mu_1 = model_1.predict(x, discount) — demand under discount
          tau  = mu_1 - mu_0                 — causal uplift

        If discount == 0, uplift is 0 by definition.
        """
        base_row = pd.DataFrame([{
            'product_name':   product,
            'category':       category,
            'mrp':            mrp,
            'stock_qty':      stock,
            'days_remaining': days_remaining,
            'shelf_life_days': shelf_life_days,
            'is_weekend':     is_weekend,
            'units_sold_7d_avg': avg_7d,
        }])
        base_row['product_name'] = base_row['product_name'].astype('category')
        base_row['category']     = base_row['category'].astype('category')

        base_demand = float(self.model_0.predict(base_row)[0])

        if discount == 0:
            return max(0.0, base_demand), 0.0

        treat_row = base_row.copy()
        treat_row['discount_pct'] = discount
        treatment_demand = float(self.model_1.predict(treat_row)[0])

        uplift = treatment_demand - base_demand
        return max(0.0, treatment_demand), max(0.0, uplift)

    # ------------------------------------------------------------------
    # Portfolio optimisation
    # ------------------------------------------------------------------
    def optimize_store_portfolio(
        self,
        inventory_df: pd.DataFrame,
        constraints: dict,
    ) -> tuple[pd.DataFrame, float, float]:
        """
        Given today's at-risk inventory, finds the globally optimal discount
        assignment across all batches using Integer Linear Programming (PuLP/CBC).

        inventory_df must have these columns:
            item_id, product_name, category, mrp, cost_price, stock_qty,
            days_remaining, shelf_life_days, is_weekend, units_sold_7d_avg

        constraints dict keys:
            max_total_waste        : int   — maximum units wasted across all batches
            category_max_discount  : dict  — e.g. {'Dairy': 10, 'Bakery': 30}
            urgency_min_discount   : float — urgency threshold above which a minimum
                                            discount is forced (default: apply 0 if absent)

        Returns (results_df, total_projected_profit, total_projected_waste)
        """
        print("Running PuLP portfolio optimiser...")
        allowed_discounts = [0, 5, 10, 20, 30, 40, 50, 60]

        # ── Step 1: pre-compute ML predictions for every (batch × discount) combo ──
        scenarios = []
        for _, row in inventory_df.iterrows():
            for d in allowed_discounts:
                total_demand, uplift = self.predict_causal_demand(
                    product        = row['product_name'],
                    category       = row['category'],
                    mrp            = row['mrp'],
                    stock          = row['stock_qty'],
                    days_remaining = row['days_remaining'],
                    shelf_life_days= row['shelf_life_days'],
                    is_weekend     = row['is_weekend'],
                    avg_7d         = row['units_sold_7d_avg'],
                    discount       = d,
                )

                sales         = min(total_demand, row['stock_qty'])
                waste         = row['stock_qty'] - sales
                selling_price = row['mrp'] * (1 - d / 100)

                # For vegetables, blend morning (MRP) and evening (discounted) price
                # using the same 60/40 split as the data generator
                if row['category'] == 'Vegetables':
                    blended_price = 0.60 * row['mrp'] + 0.40 * selling_price
                    revenue = sales * blended_price
                else:
                    revenue = sales * selling_price

                cost   = sales * row['cost_price']   # cost on units sold only
                profit = revenue - cost

                # Urgency for this batch (used in constraint D)
                urgency = 1.0 - (row['days_remaining'] / row['shelf_life_days'])

                scenarios.append({
                    'item_id':      row['item_id'],
                    'product_name': row['product_name'],
                    'category':     row['category'],
                    'discount':     d,
                    'profit':       profit,
                    'waste':        waste,
                    'sales':        sales,
                    'urgency':      urgency,
                })

        scenario_df = pd.DataFrame(scenarios)

        # ── Step 2: set up ILP ──────────────────────────────────────────────────
        prob = pulp.LpProblem("FreshMark_Profit_Maximisation", pulp.LpMaximize)

        # Binary decision variable: Select[(item_id, discount)] = 1 if chosen
        x = pulp.LpVariable.dicts(
            "Select",
            ((r.item_id, r.discount) for _, r in scenario_df.iterrows()),
            cat='Binary'
        )

        # ── Objective: maximise total profit ───────────────────────────────────
        prob += pulp.lpSum(
            x[r.item_id, r.discount] * r.profit
            for _, r in scenario_df.iterrows()
        ), "Total_Profit"

        # ── Constraint A: exactly one discount chosen per batch ────────────────
        for item_id in inventory_df['item_id']:
            prob += (
                pulp.lpSum(x[item_id, d] for d in allowed_discounts) == 1,
                f"OneDiscount_{item_id}"
            )

        # ── Constraint B: global waste cap ─────────────────────────────────────
        prob += (
            pulp.lpSum(
                x[r.item_id, r.discount] * r.waste
                for _, r in scenario_df.iterrows()
            ) <= constraints['max_total_waste'],
            "GlobalWasteCap"
        )

        # ── Constraint C: category-level max-discount caps ─────────────────────
        # e.g. {'Dairy': 10, 'Bakery': 30} means Dairy items can't exceed 10%
        cat_caps = constraints.get('category_max_discount', {})
        for _, r in scenario_df.iterrows():
            cap = cat_caps.get(r['category'])
            if cap is not None and r['discount'] > cap:
                prob += (x[r.item_id, r.discount] == 0,
                         f"CatCap_{r.item_id}_{r.discount}")

        # ── Constraint D: urgency floor — force a minimum discount ─────────────
        # If urgency >= urgency_threshold, the item MUST receive at least
        # min_urgency_discount percent off (avoids leaving high-risk stock
        # at full price by accident).
        urgency_threshold    = constraints.get('urgency_threshold', 0.85)
        min_urgency_discount = constraints.get('min_urgency_discount', 20)

        for item_id in inventory_df['item_id']:
            item_rows = scenario_df[scenario_df['item_id'] == item_id]
            if item_rows['urgency'].iloc[0] >= urgency_threshold:
                # Force all discount options BELOW the minimum to 0
                for d in [d for d in allowed_discounts if d < min_urgency_discount]:
                    prob += (x[item_id, d] == 0,
                             f"UrgencyFloor_{item_id}_{d}")

        # ── Step 3: solve ───────────────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]
        if status != 'Optimal':
            print(f"  Warning: solver returned status '{status}'")

        # ── Step 4: extract results ─────────────────────────────────────────────
        results          = []
        total_profit_out = 0.0
        total_waste_out  = 0.0

        for _, r in scenario_df.iterrows():
            if pulp.value(x[r.item_id, r.discount]) == 1:
                urgency_pct = round(r['urgency'] * 100, 1)
                results.append({
                    'Product':              r['product_name'],
                    'Category':             r['category'],
                    'Urgency (%)':          urgency_pct,
                    'Discount (%)':         r['discount'],
                    'Est. Sales (units)':   round(r['sales'], 1),
                    'Est. Waste (units)':   round(r['waste'], 1),
                    'Projected Profit': round(r['profit'], 2),
                })
                total_profit_out += r['profit']
                total_waste_out  += r['waste']

        results_df = pd.DataFrame(results)
        return results_df, total_profit_out, total_waste_out


# =============================================================================
# Main — train (or load) and run an end-of-day optimisation example
# =============================================================================
if __name__ == "__main__":
    optimizer = HybridPricingOptimizer()

    # ── Smart load-or-train ───────────────────────────────────────────────────
    m0_path = "models/model_0_control.joblib"
    m1_path = "models/model_1_treatment.joblib"

    if os.path.exists(m0_path) and os.path.exists(m1_path):
        optimizer.load_pipeline()
    else:
        print("First run — training models...\n")
        try:
            df = optimizer.load_and_preprocess("data/freshmark_sales.csv")
            optimizer.train_causal_t_learner(df)
            optimizer.save_pipeline()
        except FileNotFoundError:
            print("Error: 'data/freshmark_sales.csv' not found. Run generate_data.py first.")
            raise SystemExit(1)

    # ── End-of-day scenario ───────────────────────────────────────────────────
    # Three batches that will expire tomorrow.
    # Note: cost_price (not 'cost') matches the v2 CSV column name.
    current_inventory = pd.DataFrame([
        {
            'item_id': 'b1', 'product_name': 'Milk 1L (Toned)', 'category': 'Dairy',
            'mrp': 66, 'cost_price': 58, 'stock_qty': 15,
            'days_remaining': 1, 'shelf_life_days': 2,
            'is_weekend': 0, 'units_sold_7d_avg': 18.0,
        },
        {
            'item_id': 'b2', 'product_name': 'Coriander (Bunch)', 'category': 'Vegetables',
            'mrp': 15, 'cost_price': 8, 'stock_qty': 20,
            'days_remaining': 1, 'shelf_life_days': 2,
            'is_weekend': 0, 'units_sold_7d_avg': 11.0,
        },
        {
            'item_id': 'b3', 'product_name': 'White Bread', 'category': 'Bakery',
            'mrp': 45, 'cost_price': 32, 'stock_qty': 10,
            'days_remaining': 1, 'shelf_life_days': 4,
            'is_weekend': 0, 'units_sold_7d_avg': 12.0,
        },
        {
            'item_id': 'b4', 'product_name': 'Basmati Rice 1kg', 'category': 'Grocery',
            'mrp': 95, 'cost_price': 72, 'stock_qty': 30,
            'days_remaining': 5, 'shelf_life_days': 365,   # urgency ≈ 0.99 → big discount
            'is_weekend': 0, 'units_sold_7d_avg': 22.0,
        },
    ])

    # ── Business constraints ──────────────────────────────────────────────────
    constraints = {
        'max_total_waste':      5,        # throw away no more than 5 units total
        'category_max_discount': {
            'Dairy': 10,                  # thin margins on milk/paneer
        },
        'urgency_threshold':    0.85,     # urgency above this triggers a minimum discount
        'min_urgency_discount': 20,       # that minimum is 20 %
    }

    # ── Run optimiser ─────────────────────────────────────────────────────────
    plan, total_profit, total_waste = optimizer.optimize_store_portfolio(
        current_inventory, constraints
    )

    print("=" * 65)
    print("  OPTIMAL PRICING PORTFOLIO")
    print("=" * 65)
    print(plan.to_string(index=False))
    print("-" * 65)
    print(f"  Total Projected Profit : {total_profit:.2f}")
    print(f"  Total Projected Waste  : {total_waste:.1f} units")
    print("=" * 65)