"""
generate_data.py  (v3 - PeakShelf Salvage & Fractional Days)
------------------------------------
Creates a synthetic sales dataset for a kirana shop selling perishable goods.

Key updates for Loss Minimization:
  - days_remaining : Now a float (e.g., 3.84) representing fractional days, 
                     simulating batches arriving at different hours.
  - Discount logic : Urgency bands expanded to include 50%, 60%, and 75% 
                     salvage tiers for items within hours of expiry.
  - Margin Caps    : Removed hard caps (e.g., Dairy max 20%) to allow 
                     the solver to learn deep salvage pricing for all categories.
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Product catalogue
#    Tuple layout: (name, category, mrp, cost_price, shelf_life_days, base_demand/day)
# ---------------------------------------------------------------------------
PRODUCTS = [
    # name                   category      mrp  cost  shelf  demand/day
    ("Tomatoes",          "Vegetables",    40,   28,    4,    12),
    ("Spinach Bunch",     "Vegetables",    30,   18,    2,     9),
    ("Coriander (Bunch)", "Vegetables",    15,    8,    2,    11),

    ("Milk 1L (Toned)",   "Dairy",         66,   58,    2,    20),
    ("Paneer 200g",       "Dairy",         90,   65,    4,     6),

    ("White Bread",       "Bakery",        45,   32,    4,    15),
    ("Brown Bread",       "Bakery",        55,   40,    4,     8),
    ("Ladi Pav (6 pcs)",  "Bakery",        20,   14,    3,    20),
    ("Burger Buns (4 pcs)","Bakery",       30,   22,    4,     6),

    ("Onions 1kg",        "Grocery",       35,   24,   15,    30),
    ("Potatoes 1kg",      "Grocery",       30,   20,   20,    35),
    ("Basmati Rice 1kg",  "Grocery",       95,   72,  365,    25),  

    ("Eggs (1 Dozen)",    "Eggs",          85,   68,   14,    20),
    ("Bananas",           "Fruits",        60,   42,    5,    10),
]

N_DAYS     = 180
START_DATE = date(2025, 1, 1)

# Day-of-week demand multiplier (weekends busier for a neighbourhood kirana)
DOW_FACTOR = {0: 0.95, 1: 0.95, 2: 1.00, 3: 1.00,
              4: 1.05, 5: 1.20, 6: 1.15}


# ---------------------------------------------------------------------------
# 2. Demand uplift curve
# ---------------------------------------------------------------------------
def uplift_multiplier(discount_pct: float) -> float:
    max_uplift = 1.4   # ceiling: +140 % at extreme discount
    k = 0.045
    return 1.0 + max_uplift * (1.0 - np.exp(-k * discount_pct))


# ---------------------------------------------------------------------------
# 3. Urgency-relative discount picker (UPDATED FOR SALVAGE TIERS)
# ---------------------------------------------------------------------------
def pick_discount(days_remaining: float, shelf_life: int, category: str) -> int:
    # Urgency scales continuously from 0.0 to 1.0
    urgency = 1.0 - (days_remaining / shelf_life)

    if urgency < 0.25:
        band = [0]
    elif urgency < 0.50:
        band = [0, 0, 5]
    elif urgency < 0.70:
        band = [5, 10]
    elif urgency < 0.85:
        band = [10, 20]
    elif urgency < 0.95:
        band = [20, 30, 40]
    elif urgency < 0.98: # Extreme urgency (e.g., last ~12 hours)
        band = [40, 50, 60]
    else:                # Salvage mode (e.g., last ~3-4 hours)
        band = [60, 75]

    return int(RNG.choice(band))


# ---------------------------------------------------------------------------
# 4. Simulation loop (UPDATED FOR FRACTIONAL DAYS)
# ---------------------------------------------------------------------------
rows = []

for name, category, mrp, cost, shelf_life, base_demand in PRODUCTS:
    stock_qty      = 0
    days_remaining = 0.0
    batch_arrival  = START_DATE

    for day_idx in range(N_DAYS):
        the_date = START_DATE + timedelta(days=day_idx)
        dow      = the_date.weekday()

        # Restock when exhausted or expired (days_remaining <= 0)
        if days_remaining <= 0 or stock_qty <= 0:
            stock_qty = int(round(base_demand * shelf_life * RNG.uniform(0.9, 1.3)))
            batch_arrival = the_date
            
            # THE OFFSET TRICK: Simulate arrival at a random hour of the day
            # so days_remaining becomes a continuous float (e.g., 3.84 days)
            batch_offset = RNG.uniform(0.0, 1.0) 
            days_remaining = float(shelf_life) - batch_offset

        # Expiry date for this batch
        expiry_date = batch_arrival + timedelta(days=shelf_life)

        # Discount based on the new float urgency
        discount_pct = pick_discount(days_remaining, shelf_life, category)
        discounted_price = round(mrp * (1 - discount_pct / 100), 2)

        # Vegetable-specific: two price tiers per day
        if category == "Vegetables":
            morning_price = float(mrp)           
            evening_price = float(discounted_price)  
        else:
            morning_price = np.nan
            evening_price = np.nan

        # Demand (Poisson noise around expected demand)
        expected_demand = base_demand * DOW_FACTOR[dow] * uplift_multiplier(discount_pct)
        units_sold      = int(min(RNG.poisson(lam=max(expected_demand, 0.1)), stock_qty))

        # Revenue calculation
        if category == "Vegetables":
            morning_units = int(round(units_sold * 0.60))
            evening_units = units_sold - morning_units
            revenue_today = round(morning_units * morning_price +
                                  evening_units * evening_price, 2)
        else:
            revenue_today = round(units_sold * discounted_price, 2)

        cost_today         = round(units_sold * cost, 2)
        gross_profit_today = round(revenue_today - cost_today, 2)

        rows.append({
            "date":               the_date.isoformat(),
            "product_name":       name,
            "category":           category,
            "mrp":                mrp,
            "cost_price":         cost,
            "shelf_life_days":    shelf_life,
            "expiry_date":        expiry_date.isoformat(),
            "days_remaining":     round(days_remaining, 3), # Saved as float
            "discount_pct":       discount_pct,
            "discounted_price":   discounted_price,
            "morning_price":      morning_price,    
            "evening_price":      evening_price,    
            "stock_qty":          stock_qty,
            "units_sold_today":   units_sold,
            "is_weekend":         int(dow >= 5),
            "margin_pct":         round((mrp - cost) / mrp * 100, 1),
            "revenue_today":      revenue_today,
            "cost_today":         cost_today,
            "gross_profit_today": gross_profit_today,
        })

        stock_qty      -= units_sold
        days_remaining -= 1.0  # Subtract exactly 24 hours (1.0 days)

# ---------------------------------------------------------------------------
# 5. Build DataFrame
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)

# Convert date columns to proper datetime
df["date"]        = pd.to_datetime(df["date"])
df["expiry_date"] = pd.to_datetime(df["expiry_date"])

# 7-day rolling average demand (feature for ML model — computed per product)
df = df.sort_values(["product_name", "date"]).reset_index(drop=True)
df["units_sold_7d_avg"] = (
    df.groupby("product_name")["units_sold_today"]
      .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
      .round(2)
)
df["units_sold_7d_avg"] = df["units_sold_7d_avg"].fillna(df["units_sold_today"])

# Final sort: date-first, then category, then product name
df = df.sort_values(["date", "category", "product_name"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out_dir  = "data"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/peakshelf_sales_float.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df):,} rows  |  {df['product_name'].nunique()} products  |  {df['date'].nunique()} days")
print(f"Saved → {out_path}")
print()
print("Sample — checking for fractional days and deep discounts:")
sample_df = df[(df["days_remaining"] < 0.5) & (df["discount_pct"] >= 50)]
if not sample_df.empty:
    print(sample_df[["date", "product_name", "days_remaining", "discount_pct"]].head().to_string(index=False))
else:
    print("No extreme urgencies in the first few rows, but logic is active.")