"""
main.py — PeakShelf Terminal UI
-----------------------------------
The interactive controller. Loads the predictive and prescriptive engines, 
then runs a continuous terminal loop allowing the shopkeeper to query 
individual batches of inventory for optimal salvage pricing.
"""

import os
from datetime import datetime
import pandas as pd

import config
from predictive_engine import DemandPredictor
from prescriptive_engine import PriceOptimizer

# ---------------------------------------------------------------------------
# 1. Mock Store Database (Fills the gap between UI and ML features)
# ---------------------------------------------------------------------------
STORE_DB = {
    'White Bread': {'mrp': 45.0, 'avg_7d': 15.0},
    'Milk 1L (Toned)': {'mrp': 66.0, 'avg_7d': 20.0},
    'Coriander (Bunch)': {'mrp': 15.0, 'avg_7d': 11.0},
    'Basmati Rice 1kg': {'mrp': 95.0, 'avg_7d': 25.0},
}


def load_or_train_predictor() -> DemandPredictor:
    predictor = DemandPredictor()
    if predictor.is_trained_on_disk():
        predictor.load()
    else:
        print("First run — training models...\n")
        try:
            df = predictor.load_and_preprocess()
            predictor.train(df)
            predictor.save()
        except FileNotFoundError:
            print(f"Error: '{config.DATA_PATH}' not found. Run data_generator.py first.")
            raise SystemExit(1)
    return predictor


def run_interactive_cli(predictor: DemandPredictor, optimizer: PriceOptimizer):
    """The main interactive loop for the terminal."""
    
    while True:
        print("\n" + "=" * 50)
        print("  PEAKSHELF - SALVAGE PRICING TERMINAL")
        print("=" * 50)
        
        now = datetime.now()
        print(f"[System Time]: {now.strftime('%a, %b %d, %Y - %H:%M')}\n")

        # --- 1. Collect User Input ---
        print("Available test products: White Bread, Milk 1L (Toned), Coriander (Bunch)")
        product_name = input("> Enter Product Name (or 'q' to quit): ").strip()
        if product_name.lower() in ['q', 'quit', 'exit']:
            break
            
        if product_name not in STORE_DB:
            print(f"[Error] '{product_name}' not found in store database. Try again.")
            continue

        try:
            expiry_str = input("> Enter Expiry (YYYY-MM-DD HH:MM) : ").strip()
            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M")
            cost_price = float(input("> Enter Cost Price (₹) : ").strip())
            stock_qty = int(input("> Enter Current Stock : ").strip())
        except ValueError:
            print("[Error] Invalid input format. Please check your dates and numbers.")
            continue

        # --- 2. Auto-fill the missing ML features ---
        category = config.PRODUCT_CATEGORY_MAP.get(product_name, "Grocery")
        shelf_life = config.CATEGORY_SHELF_LIFE_DAYS.get(category, 7)
        mrp = STORE_DB[product_name]['mrp']
        avg_7d = STORE_DB[product_name]['avg_7d']
        
        # Calculate fractional days remaining
        time_diff = expiry_dt - now
        days_remaining = round(time_diff.total_seconds() / 86400.0, 3)
        
        is_weekend = 1 if now.weekday() >= 5 else 0

        # Build the single-item inventory DataFrame
        inventory_df = pd.DataFrame([{
            'item_id': 'batch_01', 
            'product_name': product_name, 
            'category': category,
            'mrp': mrp, 
            'cost_price': cost_price, 
            'stock_qty': stock_qty,
            'days_remaining': days_remaining, 
            'shelf_life_days': shelf_life,
            'is_weekend': is_weekend, 
            'units_sold_7d_avg': avg_7d,
        }])

        print("\n--- CALCULATING OPTIMAL SALVAGE STRATEGY ---")
        if days_remaining < 0:
            print(f"[WARNING] Product is already expired by {abs(days_remaining):.1f} days!")
        
        # --- 3. Evaluate Options & Optimize ---
        # We manually call _build_scenarios to show the user the ML predictions
        scenarios = optimizer._build_scenarios(inventory_df)
        
        print("\nProjected Demand by Discount Tier:")
        for _, row in scenarios.iterrows():
            d = row['discount']
            price = mrp * (1 - d/100)
            sales = int(row['sales'])
            profit = row['profit']
            
            flag = "<- Below cost" if price < cost_price else ""
            print(f" {d:2.0f}% (₹{price:5.2f}) : {sales:2d} units sold | Projected P&L: ₹{profit:6.2f} {flag}")

        # Run the actual PuLP optimizer to pick the winner
        plan, total_profit, total_waste = optimizer.optimize(inventory_df, config.DEFAULT_CONSTRAINTS)
        
        if not plan.empty:
            winner = plan.iloc[0]
            print("\n" + "=" * 50)
            print(f"RECOMMENDATION: Apply {winner['Discount (%)']:.0f}% Discount")
            print(f"Est. Sales: {winner['Est. Sales (units)']} units | Total Waste: {winner['Est. Waste (units)']} units")
            print(f"Net Financial Impact: ₹{total_profit:.2f}")
            print("=" * 50)
        else:
            print("Optimizer failed to find a valid pricing plan.")
            
        input("\nPress Enter to query another product...")


def main():
    # Setup engines
    predictor = load_or_train_predictor()
    optimizer = PriceOptimizer(predictor)

    # Launch CLI
    try:
        run_interactive_cli(predictor, optimizer)
    except KeyboardInterrupt:
        print("\nExiting PeakShelf Terminal. Goodbye!")


if __name__ == "__main__":
    main()