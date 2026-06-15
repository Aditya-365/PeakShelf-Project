"""
generate_data.py
-----------------
Creates a synthetic dataset for FreshMark — a kirana shop selling
perishable goods at MRP/Cost-based discounts near expiry.
 
Each row = one product, in one store, on one day.
 
The synthetic discount -> demand "uplift" relationship is loosely
inspired by patterns seen in the FreshRetailNet-50K dataset
(discounts of ~10-30% produce a real but diminishing sales bump),
scaled down to a single small shop.
 
Output: data/freshmark_sales.csv
"""
 
import numpy as np
import pandas as pd
from datetime import date, timedelta
 
RNG = np.random.default_rng(42)
 
# ---------------------------------------------------------------------------
# 1. Define products typical of an Indian kirana shop
#    Each has: MRP, cost price, shelf life (days), and average daily demand
#    at full price (no discount).
# ---------------------------------------------------------------------------
PRODUCTS = [
    # name,        mrp,  cost, shelf_life_days, base_demand_units/day
    ("Tomatoes",    40,   28,   4,   12),
    ("Bananas",     60,   42,   5,   10),
    ("Milk 1L",     66,   58,   2,   20),
    ("Bread",       45,   32,   3,    8),
    ("Paneer 200g", 90,   65,   4,    6),
    ("Spinach Bunch",30,   18,   2,    9),
    ("White Bread",         45,   32,   3,   15),
    ("Brown Bread",         55,   40,   4,    8),
    ("Ladi Pav (6 pcs)",    20,   14,   3,   20),
    ("Burger Buns (4 pcs)", 30,   22,   4,    6),
    ("Onions 1kg",          35,   24,  15,   30),
    ("Potatoes 1kg",        30,   20,  20,   35),
    ("Eggs (1 Dozen)",      85,   68,  14,   20),
]
 
N_DAYS = 180
START_DATE = date(2025, 1, 1)
 
# Allowed discount steps a kirana shopkeeper realistically offers
DISCOUNT_STEPS = [0, 5, 10, 20, 30, 40]
 
# ---------------------------------------------------------------------------
# 2. Discount -> demand uplift curve
#    Diminishing returns: big jump from 10% -> 20%, flattening after that.
#    Returns a multiplier applied to baseline demand.
# ---------------------------------------------------------------------------
def uplift_multiplier(discount_pct: float) -> float:
    # Saturating curve: uplift = max_uplift * (1 - exp(-k * discount))
    max_uplift = 1.4      # at very high discount, demand can be up to +140%
    k = 0.045
    return 1 + max_uplift * (1 - np.exp(-k * discount_pct))
 
 
# ---------------------------------------------------------------------------
# 3. Day-of-week demand factor (weekends busier for a neighbourhood shop)
# ---------------------------------------------------------------------------
DOW_FACTOR = {
    0: 0.95,  # Monday
    1: 0.95,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.00,  # Thursday
    4: 1.05,  # Friday
    5: 1.20,  # Saturday
    6: 1.15,  # Sunday
}
 
 
# ---------------------------------------------------------------------------
# 4. Simulation loop
# ---------------------------------------------------------------------------
rows = []
 
for name, mrp, cost, shelf_life, base_demand in PRODUCTS:
    stock_qty = 0
    days_to_expiry = 0
 
    for day_idx in range(N_DAYS):
        the_date = START_DATE + timedelta(days=day_idx)
        dow = the_date.weekday()
 
        # Restock when the previous batch has expired / run out
        if days_to_expiry <= 0 or stock_qty <= 0:
            # Order roughly enough stock to last the shelf life, with noise
            stock_qty = int(round(base_demand * shelf_life * RNG.uniform(0.9, 1.3)))
            days_to_expiry = shelf_life
 
        # ---- Discount policy for generating varied training data ----
        # Shopkeeper applies a markdown only in the last 2 days before expiry.
        # To get a good spread of (discount -> demand) observations for
        # modelling, the discount level in that window is randomised across
        # the allowed steps (in real life this would be a fixed policy;
        # here we vary it on purpose so the dataset covers all discount
        # levels for curve-fitting).
        if days_to_expiry <= 2:
            discount_pct = int(RNG.choice(DISCOUNT_STEPS, p=[0.10, 0.15, 0.20, 0.30, 0.15, 0.10]))
        else:
            discount_pct = 0
 
        current_price = round(mrp * (1 - discount_pct / 100), 2)
 
        # ---- Demand simulation ----
        expected_demand = base_demand * DOW_FACTOR[dow] * uplift_multiplier(discount_pct)
        # Poisson noise around expected demand
        demand = RNG.poisson(lam=max(expected_demand, 0.1))
 
        units_sold = int(min(demand, stock_qty))
        waste_if_unsold = stock_qty - units_sold if days_to_expiry == 1 else 0
 
        rows.append({
            "date": the_date.isoformat(),
            "product_name": name,
            "mrp": mrp,
            "cost_price": cost,
            "current_price": current_price,
            "discount_pct": discount_pct,
            "stock_qty": stock_qty,
            "days_to_expiry": days_to_expiry,
            "units_sold_today": units_sold,
            "is_weekend": int(dow >= 5),
        })
 
        # Update stock & countdown for next day
        stock_qty -= units_sold
        days_to_expiry -= 1
 
df = pd.DataFrame(rows)
 
# ---------------------------------------------------------------------------
# 5. Derived columns (computed, not hand-entered — per reviewer feedback)
# ---------------------------------------------------------------------------
df["margin_pct"] = round((df["mrp"] - df["cost_price"]) / df["mrp"] * 100, 1)
df["profit_per_unit"] = round(df["current_price"] - df["cost_price"], 2)
df["revenue_today"] = round(df["current_price"] * df["units_sold_today"], 2)
df["profit_today"] = round(df["profit_per_unit"] * df["units_sold_today"], 2)
 
# 7-day rolling average of units sold, per product (simple baseline demand signal)
df = df.sort_values(["product_name", "date"]).reset_index(drop=True)
df["units_sold_7d_avg"] = (
    df.groupby("product_name")["units_sold_today"]
      .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
      .round(2)
)
df["units_sold_7d_avg"] = df["units_sold_7d_avg"].fillna(df["units_sold_today"])
 
# Loss if the remaining stock had expired unsold (the "do nothing" cost)
df["waste_if_unsold"] = (df["stock_qty"] - df["units_sold_today"]).clip(lower=0)
df.loc[df["days_to_expiry"] != 1, "waste_if_unsold"] = 0
df["loss_if_expired"] = (df["waste_if_unsold"] * df["cost_price"]).round(2)
 
# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out_path = "data/freshmark_sales.csv"
df.to_csv(out_path, index=False)
 
print(f"Generated {len(df)} rows across {df['product_name'].nunique()} products")
print(f"Saved to {out_path}")
print()
print(df.head(10).to_string(index=False))
print()
print("Discount level distribution (days where a discount was applied):")
print(df.loc[df['discount_pct'] > 0, 'discount_pct'].value_counts().sort_index())