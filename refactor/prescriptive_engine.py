"""
prescriptive_engine.py — PeakShelf
-----------------------------------
PriceOptimizer: an Integer Linear Program (PuLP/CBC) that decides ONE
discount level per inventory batch to maximise total store profit.

Constraints currently enforced:
  A) Exactly one discount chosen per batch (one-hot)
  B) Global waste cap across all batches
  D) Urgency floor — high-urgency items are forced to discount >= a minimum

(Constraint C, a category-level max-discount cap, was removed as part of
the Loss Minimization change — see config.DEFAULT_CONSTRAINTS.)

Objective: true loss minimization, not gross-margin-on-sold. cost_price is
charged against full stock_qty (sold + wasted), so waste is penalized at
its full cost rather than free — see the `_build_scenarios` note below.

This module owns everything pricing/business-logic: turning ML demand
forecasts into a profit scenario table, wiring up the ILP, and extracting
the winning plan. It depends on a DemandPredictor for forecasts but knows
nothing about how those forecasts are produced (LightGBM, T-Learner, etc.)
— that separation is what lets the ML model evolve independently.
"""

import pandas as pd
import pulp

import config


class PriceOptimizer:

    def __init__(self, predictor):
        """
        predictor: a DemandPredictor instance (already trained or loaded)
        used to forecast demand for every (batch x discount) scenario.
        """
        self.predictor = predictor
        self.discount_tiers = config.DISCOUNT_TIERS

    # ------------------------------------------------------------------
    # Step 1: turn raw inventory into a (batch x discount) scenario table
    # ------------------------------------------------------------------
    def _build_scenarios(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """
        inventory_df must have these columns:
            item_id, product_name, category, mrp, cost_price, stock_qty,
            days_remaining, shelf_life_days, is_weekend, units_sold_7d_avg
        """
        scenarios = []
        for _, row in inventory_df.iterrows():
            for d in self.discount_tiers:
                total_demand, _uplift = self.predictor.predict(
                    product=row['product_name'],
                    category=row['category'],
                    mrp=row['mrp'],
                    stock=row['stock_qty'],
                    days_remaining=row['days_remaining'],
                    shelf_life_days=row['shelf_life_days'],
                    is_weekend=row['is_weekend'],
                    avg_7d=row['units_sold_7d_avg'],
                    discount=d,
                )

                sales = min(total_demand, row['stock_qty'])
                waste = row['stock_qty'] - sales
                selling_price = row['mrp'] * (1 - d / 100)

                # Vegetables blend a morning (MRP) and evening (discounted)
                # price using the same split as the data generator.
                if row['category'] == 'Vegetables':
                    blended_price = (
                        config.VEGETABLE_MORNING_WEIGHT * row['mrp']
                        + config.VEGETABLE_EVENING_WEIGHT * selling_price
                    )
                    revenue = sales * blended_price
                else:
                    revenue = sales * selling_price

                # LOSS MINIMIZATION FIX (was: cost charged on sold units
                # only, i.e. `revenue - sales * cost_price`). That silently
                # optimizes gross margin on units sold, not total P&L,
                # because it gives waste a $0 penalty even though the
                # inventory was already paid for. Charging cost_price
                # against the *full* stock_qty (sold + wasted) makes the
                # objective true loss minimization: it now correctly
                # rewards clearing high-volume/low-margin stock instead of
                # protecting margin at the cost of higher waste.
                cost_of_sold = sales * row['cost_price']
                cost_of_waste = waste * row['cost_price']
                profit = revenue - cost_of_sold - cost_of_waste

                urgency = 1.0 - (row['days_remaining'] / row['shelf_life_days'])

                scenarios.append({
                    'item_id': row['item_id'],
                    'product_name': row['product_name'],
                    'category': row['category'],
                    'discount': d,
                    'profit': profit,
                    'waste': waste,
                    'sales': sales,
                    'urgency': urgency,
                })

        return pd.DataFrame(scenarios)

    # ------------------------------------------------------------------
    # Step 2-4: build the ILP, solve it, extract the winning plan
    # ------------------------------------------------------------------
    def optimize(
        self,
        inventory_df: pd.DataFrame,
        constraints: dict,
    ) -> tuple[pd.DataFrame, float, float]:
        """
        constraints dict keys:
            max_total_waste        : int   — maximum units wasted across all batches
            urgency_threshold      : float — urgency above which a minimum discount is forced
            min_urgency_discount   : float — that minimum discount, in percent

        Returns (results_df, total_projected_profit, total_projected_waste)
        """
        print("Running PuLP portfolio optimiser...")
        scenario_df = self._build_scenarios(inventory_df)

        prob = pulp.LpProblem("PeakShelf_Profit_Maximisation", pulp.LpMaximize)

        # Binary decision variable: Select[(item_id, discount)] = 1 if chosen
        x = pulp.LpVariable.dicts(
            "Select",
            ((r.item_id, r.discount) for _, r in scenario_df.iterrows()),
            cat='Binary',
        )

        # ── Objective: maximise total profit ────────────────────────────
        prob += pulp.lpSum(
            x[r.item_id, r.discount] * r.profit
            for _, r in scenario_df.iterrows()
        ), "Total_Profit"

        # ── Constraint A: exactly one discount chosen per batch ─────────
        for item_id in inventory_df['item_id']:
            prob += (
                pulp.lpSum(x[item_id, d] for d in self.discount_tiers) == 1,
                f"OneDiscount_{item_id}"
            )

        # ── Constraint B: global waste cap ───────────────────────────────
        prob += (
            pulp.lpSum(
                x[r.item_id, r.discount] * r.waste
                for _, r in scenario_df.iterrows()
            ) <= constraints['max_total_waste'],
            "GlobalWasteCap"
        )

        # Constraint C (category-level max-discount cap) removed — see
        # config.DEFAULT_CONSTRAINTS for rationale (Loss Minimization,
        # Item 1). The waste-penalized objective now owns that tradeoff.

        # ── Constraint D: urgency floor ───────────────────────────────────
        urgency_threshold = constraints.get('urgency_threshold', 0.85)
        min_urgency_discount = constraints.get('min_urgency_discount', 20)

        for item_id in inventory_df['item_id']:
            item_rows = scenario_df[scenario_df['item_id'] == item_id]
            if item_rows['urgency'].iloc[0] >= urgency_threshold:
                for d in [d for d in self.discount_tiers if d < min_urgency_discount]:
                    prob += (x[item_id, d] == 0,
                             f"UrgencyFloor_{item_id}_{d}")

        # ── Solve ─────────────────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]
        if status != 'Optimal':
            print(f"  Warning: solver returned status '{status}'")

        # ── Extract results ──────────────────────────────────────────────
        results = []
        total_profit_out = 0.0
        total_waste_out = 0.0

        for _, r in scenario_df.iterrows():
            if pulp.value(x[r.item_id, r.discount]) == 1:
                urgency_pct = round(r['urgency'] * 100, 1)
                results.append({
                    'Product': r['product_name'],
                    'Category': r['category'],
                    'Urgency (%)': urgency_pct,
                    'Discount (%)': r['discount'],
                    'Est. Sales (units)': round(r['sales'], 1),
                    'Est. Waste (units)': round(r['waste'], 1),
                    'Projected Profit': round(r['profit'], 2),
                })
                total_profit_out += r['profit']
                total_waste_out += r['waste']

        results_df = pd.DataFrame(results)
        return results_df, total_profit_out, total_waste_out
