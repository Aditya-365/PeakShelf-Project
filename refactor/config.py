"""
config.py — PeakShelf
-----------------------------------
Single source of truth for static constants shared across the predictive
engine, the prescriptive engine, and the (future) CLI. Nothing in here
should depend on any other project module — this file sits at the bottom
of the import graph.
"""

# ---------------------------------------------------------------------------
# Discount tiers
# ---------------------------------------------------------------------------
# The full action space the ILP solver is allowed to choose from, and the
# set of "treatment" levels the T-Learner's Model 1 is trained across.
#
# CAVEAT (Loss Minimization, Item 1): tiers 50/60/75 were added for the
# salvage-pricing action space, but as of this training CSV there are ZERO
# treatment rows at those discounts (max observed is 40%, and only 6 rows
# at that). Model 1 will extrapolate flatly past 40% until the data is
# regenerated with real treatment rows at 50/60/75 — don't trust those
# tiers' predicted demand until then.
DISCOUNT_TIERS = [0, 5, 10, 20, 30, 40, 50, 60, 75]

# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------
# Default total shelf life (in days) per category — used by the CLI to
# auto-fill shelf_life_days when a shopkeeper only types a product name.
CATEGORY_SHELF_LIFE_DAYS = {
    'Dairy': 2,
    'Vegetables': 2,
    'Bakery': 4,
    'Grocery': 365,
}

# Product name -> category lookup, used by the CLI (Item 2) so a shopkeeper
# typing "Milk" doesn't need to know it belongs to "Dairy".
PRODUCT_CATEGORY_MAP = {
    'Milk 1L (Toned)': 'Dairy',
    'Coriander (Bunch)': 'Vegetables',
    'White Bread': 'Bakery',
    'Basmati Rice 1kg': 'Grocery',
}

# Vegetables are sold at a blended price across the day: morning (full MRP)
# and evening (discounted). Weights must sum to 1.0.
VEGETABLE_MORNING_WEIGHT = 0.60
VEGETABLE_EVENING_WEIGHT = 0.40

# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_0_PATH = f"{MODEL_DIR}/model_0_control.joblib"
MODEL_1_PATH = f"{MODEL_DIR}/model_1_treatment.joblib"

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
DATA_PATH = "data/peakshelf_sales_float.csv"

# ---------------------------------------------------------------------------
# LightGBM hyperparameters (shared by both T-Learner arms)
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# Model 0 (control) never sees the discount — it predicts baseline demand.
FEATURES_M0 = [
    'product_name', 'category', 'mrp', 'stock_qty',
    'days_remaining', 'shelf_life_days', 'is_weekend', 'units_sold_7d_avg',
]
# Model 1 (treatment) additionally sees discount_pct.
FEATURES_M1 = FEATURES_M0 + ['discount_pct']

CATEGORICAL_FEATURES = ['product_name', 'category']

# ---------------------------------------------------------------------------
# Default business constraints for the ILP optimizer
# ---------------------------------------------------------------------------
DEFAULT_CONSTRAINTS = {
    'max_total_waste': 5,               # units wasted, across all batches
    # 'category_max_discount' removed (Loss Minimization, Item 1): a hard
    # cap on discount depth actively fights salvage pricing — it can force
    # the solver into wasting a unit outright rather than selling it at a
    # steep loss, which is worse. The waste-penalized objective in
    # prescriptive_engine now bears the full cost of that tradeoff, so a
    # separate cap is no longer needed to protect margin.
    'urgency_threshold': 0.85,          # urgency above this forces a floor
    'min_urgency_discount': 20,         # ...of at least this discount %
}
