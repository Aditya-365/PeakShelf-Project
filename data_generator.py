"""
generate_data.py  (v2 – FreshMark)
------------------------------------
Creates a synthetic sales dataset for a kirana shop selling perishable goods.

Key design decisions:
  - Row ordering   : date-first — all products for a given day appear together,
                     sorted by (date → category → product_name)
  - shelf_life_days: fixed property of the product (total days when fresh)
  - days_remaining : integer countdown for the current batch in stock
  - expiry_date    : explicit datetime column (batch_arrival + shelf_life_days)
  - Discount logic : urgency-relative — urgency = 1 - (days_remaining / shelf_life_days)
                     so a rice bag 5 days from expiry (urgency ≈ 0.99) gets 40-60%
                     while bread 2 days from expiry on a 4-day shelf (urgency = 0.50)
                     only gets 5-10%.
  - Categories     : Vegetables / Dairy / Bakery / Grocery / Eggs / Fruits
  - Vegetable pricing: morning_price = MRP (6 AM–noon), evening_price = discounted
                       (noon–9 PM). Revenue is a 60/40 weighted blend.
  - Revenue clarity: revenue_today, cost_today, gross_profit_today are all explicit
  - Removed        : waste_if_unsold, loss_if_expired

Output: data/freshmark_sales.csv
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
    ("Basmati Rice 1kg",  "Grocery",       95,   72,  365,    25),  # long shelf life

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
#    Saturating exponential — big jump from 10→20 %, diminishing beyond that.
#    Returns a multiplier on base demand (e.g. 1.30 = 30 % more units sold).
# ---------------------------------------------------------------------------
def uplift_multiplier(discount_pct: float) -> float:
    max_uplift = 1.4   # ceiling: +140 % at extreme discount
    k = 0.045
    return 1.0 + max_uplift * (1.0 - np.exp(-k * discount_pct))


# ---------------------------------------------------------------------------
# 3. Urgency-relative discount picker
#
#   urgency = 1 − (days_remaining / shelf_life_days)
#     → 0.0  brand-new batch, no pressure
#     → 1.0  expires today, maximum pressure
#
#   This means the same "5 days left" maps to very different discounts:
#     • Basmati Rice (shelf=365): urgency ≈ 0.99  → 40–60 % discount
#     • Bread        (shelf=4)  : 5 days left is impossible (batch already gone)
#     • Eggs         (shelf=14) : urgency ≈ 0.64  → 5–10 % discount
#
#   Category-specific margin caps are enforced after band selection.
# ---------------------------------------------------------------------------
def pick_discount(days_remaining: int, shelf_life: int, category: str) -> int:
    urgency = 1.0 - (days_remaining / shelf_life)

    if urgency < 0.25:
        band = [0]
    elif urgency < 0.50:
        band = [0, 0, 5]        # mostly no discount, occasional 5 %
    elif urgency < 0.70:
        band = [5, 10]
    elif urgency < 0.85:
        band = [20, 30]
    elif urgency < 0.95:
        band = [30, 40]
    else:
        band = [40, 50, 60]

    # Dairy margins are thin — cap at 20 %
    if category == "Dairy":
        band = [d for d in band if d <= 20] or [20]

    return int(RNG.choice(band))


# ---------------------------------------------------------------------------
# 4. Simulation loop (per product, per day)
# ---------------------------------------------------------------------------
rows = []

for name, category, mrp, cost, shelf_life, base_demand in PRODUCTS:
    stock_qty      = 0
    days_remaining = 0
    batch_arrival  = START_DATE

    for day_idx in range(N_DAYS):
        the_date = START_DATE + timedelta(days=day_idx)
        dow      = the_date.weekday()

        # Restock when the current batch is exhausted or expired
        if days_remaining <= 0 or stock_qty <= 0:
            stock_qty      = int(round(base_demand * shelf_life * RNG.uniform(0.9, 1.3)))
            days_remaining = shelf_life
            batch_arrival  = the_date

        # Expiry date for this batch
        expiry_date = batch_arrival + timedelta(days=shelf_life)

        # Discount based on relative urgency
        discount_pct     = pick_discount(days_remaining, shelf_life, category)
        discounted_price = round(mrp * (1 - discount_pct / 100), 2)

        # Vegetable-specific: two price tiers per day
        if category == "Vegetables":
            morning_price = float(mrp)           # sold at MRP before noon
            evening_price = float(discounted_price)  # discounted after noon
        else:
            morning_price = np.nan
            evening_price = np.nan

        # Demand (Poisson noise around expected demand)
        expected_demand = base_demand * DOW_FACTOR[dow] * uplift_multiplier(discount_pct)
        units_sold      = int(min(RNG.poisson(lam=max(expected_demand, 0.1)), stock_qty))

        # Revenue
        # Vegetables: 60 % of units sold in morning at MRP, 40 % in evening at discount
        # All others: single price (discounted_price) applies all day
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
            "days_remaining":     days_remaining,
            "discount_pct":       discount_pct,
            "discounted_price":   discounted_price,
            "morning_price":      morning_price,    # Vegetables only (NaN otherwise)
            "evening_price":      evening_price,    # Vegetables only (NaN otherwise)
            "stock_qty":          stock_qty,
            "units_sold_today":   units_sold,
            "is_weekend":         int(dow >= 5),
            "margin_pct":         round((mrp - cost) / mrp * 100, 1),
            "revenue_today":      revenue_today,
            "cost_today":         cost_today,
            "gross_profit_today": gross_profit_today,
        })

        stock_qty      -= units_sold
        days_remaining -= 1

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
out_path = f"{out_dir}/freshmark_sales.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df):,} rows  |  {df['product_name'].nunique()} products  |  {df['date'].nunique()} days")
print(f"Saved → {out_path}")
print()
print("Sample — first day across all products:")
print(df[df["date"] == df["date"].min()]
      [["product_name", "category", "shelf_life_days", "days_remaining",
        "expiry_date", "discount_pct", "revenue_today", "gross_profit_today"]]
      .to_string(index=False))