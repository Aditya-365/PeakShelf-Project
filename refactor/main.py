"""
main.py — PeakShelf controller
-----------------------------------
The lightweight glue file. Loads (or trains) the predictive engine, hands
it to the prescriptive engine, runs an end-of-day optimisation over
today's at-risk inventory, and prints the recommended plan.

Once Item 2 (Interactive Terminal GUI) is built, the hardcoded
`get_demo_inventory()` below will be replaced by a `while True:` input
loop that builds the same inventory_df shape from shopkeeper input.
"""

import pandas as pd

import config
from predictive_engine import DemandPredictor
from prescriptive_engine import PriceOptimizer


def get_demo_inventory() -> pd.DataFrame:
    """
    Placeholder inventory: batches expiring soon.
    cost_price (not 'cost') matches the training CSV column name.
    """
    return pd.DataFrame([
        {
            'item_id': 'b1', 'product_name': 'Milk 1L (Toned)', 'category': 'Dairy',
            'mrp': 66, 'cost_price': 58, 'stock_qty': 15,
            'days_remaining': 1.0, 'shelf_life_days': 2,
            'is_weekend': 0, 'units_sold_7d_avg': 18.0,
        },
        {
            'item_id': 'b2', 'product_name': 'Coriander (Bunch)', 'category': 'Vegetables',
            'mrp': 15, 'cost_price': 8, 'stock_qty': 20,
            'days_remaining': 0.15, 'shelf_life_days': 2,
            'is_weekend': 0, 'units_sold_7d_avg': 11.0,
        },
        {
            'item_id': 'b3', 'product_name': 'White Bread', 'category': 'Bakery',
            'mrp': 45, 'cost_price': 32, 'stock_qty': 10,
            'days_remaining': 0.5, 'shelf_life_days': 4,
            'is_weekend': 0, 'units_sold_7d_avg': 12.0,
        },
        {
            'item_id': 'b4', 'product_name': 'Basmati Rice 1kg', 'category': 'Grocery',
            'mrp': 95, 'cost_price': 72, 'stock_qty': 30,
            'days_remaining': 5.0, 'shelf_life_days': 365,   # urgency ~= 0.99 -> big discount
            'is_weekend': 0, 'units_sold_7d_avg': 22.0,
        },
    ])


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
            print(f"Error: '{config.DATA_PATH}' not found. Run generate_data.py first.")
            raise SystemExit(1)

    return predictor


def print_plan(plan: pd.DataFrame, total_profit: float, total_waste: float) -> None:
    print("=" * 65)
    print("  OPTIMAL PRICING PORTFOLIO")
    print("=" * 65)
    print(plan.to_string(index=False))
    print("-" * 65)
    print(f"  Total Projected Profit : {total_profit:.2f}")
    print(f"  Total Projected Waste  : {total_waste:.1f} units")
    print("=" * 65)


def main():
    predictor = load_or_train_predictor()
    optimizer = PriceOptimizer(predictor)

    inventory = get_demo_inventory()
    plan, total_profit, total_waste = optimizer.optimize(inventory, config.DEFAULT_CONSTRAINTS)

    print_plan(plan, total_profit, total_waste)


if __name__ == "__main__":
    main()
