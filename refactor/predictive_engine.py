"""
predictive_engine.py — PeakShelf
-----------------------------------
DemandPredictor: a LightGBM-based T-Learner for causal, discount-aware
demand forecasting.

T-Learner design:
  model_0  trained on undiscounted rows  -> predicts baseline demand
  model_1  trained on discounted rows    -> predicts demand under a given discount
  uplift   = model_1(x, discount) - model_0(x)

This module owns everything ML: loading/preprocessing training data,
fitting both arms of the T-Learner, persisting/restoring models, and
turning a single (batch, discount) pair into a demand forecast. It knows
nothing about pricing, profit, or the optimizer — that lives in
prescriptive_engine.py.
"""

import os

import joblib
import lightgbm as lgb
import pandas as pd

import config


class DemandPredictor:

    def __init__(self):
        # Two separate LightGBM regressors — one per T-Learner arm.
        self.model_0 = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        self.model_1 = lgb.LGBMRegressor(**config.LGBM_PARAMS)

        self.features_m0 = config.FEATURES_M0
        self.features_m1 = config.FEATURES_M1
        self.categorical_features = config.CATEGORICAL_FEATURES

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_and_preprocess(self, filepath: str = config.DATA_PATH) -> pd.DataFrame:
        """
        Loads the training CSV and prepares it for LightGBM, which handles
        categoricals natively when typed as 'category'.
        """
        df = pd.read_csv(filepath, parse_dates=["date", "expiry_date"])
        df['product_name'] = df['product_name'].astype('category')
        df['category'] = df['category'].astype('category')
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> None:
        """
        Trains both T-Learner arms:
          Model 0 on control rows (discount_pct == 0)  -> baseline demand
          Model 1 on treatment rows (discount_pct > 0) -> promoted demand
        """
        print("Training LightGBM T-Learner...")

        df_control = df[df['discount_pct'] == 0].copy()
        df_treatment = df[df['discount_pct'] > 0].copy()

        print(f"    Control rows  : {len(df_control):,}")
        print(f"    Treatment rows: {len(df_treatment):,}")

        self.model_0.fit(
            df_control[self.features_m0],
            df_control['units_sold_today'],
            categorical_feature=self.categorical_features,
        )
        print("    Model 0 (Control - Base Demand) trained")

        self.model_1.fit(
            df_treatment[self.features_m1],
            df_treatment['units_sold_today'],
            categorical_feature=self.categorical_features,
        )
        print("    Model 1 (Treatment - Promoted Demand) trained\n")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, folder_path: str = config.MODEL_DIR) -> None:
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.model_0, f"{folder_path}/model_0_control.joblib")
        joblib.dump(self.model_1, f"{folder_path}/model_1_treatment.joblib")
        print(f"Models saved to '{folder_path}/'\n")

    def load(self, folder_path: str = config.MODEL_DIR) -> None:
        try:
            self.model_0 = joblib.load(f"{folder_path}/model_0_control.joblib")
            self.model_1 = joblib.load(f"{folder_path}/model_1_treatment.joblib")
            print("Pre-trained models loaded and ready.\n")
        except FileNotFoundError:
            print(f"Error: model files not found in '{folder_path}/'. Train first.\n")
            raise

    def is_trained_on_disk(self) -> bool:
        """True if both model artifacts already exist at the configured paths."""
        return os.path.exists(config.MODEL_0_PATH) and os.path.exists(config.MODEL_1_PATH)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        product: str,
        category: str,
        mrp: float,
        stock: int,
        days_remaining: float,
        shelf_life_days: int,
        is_weekend: int,
        avg_7d: float,
        discount: float,
    ) -> tuple[float, float]:
        """
        Returns (expected_units_sold, causal_uplift).

        Uses the T-Learner logic:
          mu_0 = model_0.predict(x)           -> baseline demand
          mu_1 = model_1.predict(x, discount)  -> demand under discount
          tau  = mu_1 - mu_0                   -> causal uplift

        If discount == 0, uplift is 0 by definition.
        """
        base_row = pd.DataFrame([{
            'product_name': product,
            'category': category,
            'mrp': mrp,
            'stock_qty': stock,
            'days_remaining': days_remaining,
            'shelf_life_days': shelf_life_days,
            'is_weekend': is_weekend,
            'units_sold_7d_avg': avg_7d,
        }])
        base_row['product_name'] = base_row['product_name'].astype('category')
        base_row['category'] = base_row['category'].astype('category')

        base_demand = float(self.model_0.predict(base_row)[0])

        if discount == 0:
            return max(0.0, base_demand), 0.0

        treat_row = base_row.copy()
        treat_row['discount_pct'] = discount
        treatment_demand = float(self.model_1.predict(treat_row)[0])

        uplift = treatment_demand - base_demand
        return max(0.0, treatment_demand), max(0.0, uplift)
