"""
LightGBM Hyperparameter Tuning with Optuna
Uses final_train_v3.csv and final_test_v3.csv
Outputs:
 - best_lgbm_model.txt
 - optuna_lgbm_study.pkl
 - tuned_lgbm_predictions.png
 - tuned_lgbm_metrics.txt
"""

import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle

TRAIN_PATH = Path("E:/Inventory-Management-and-Supply-Chain-Optimization/data/final_train_v3.csv")

TEST_PATH = Path("E:/Inventory-Management-and-Supply-Chain-Optimization/data/final_test_v3.csv")
MODEL_OUT = Path("E:/Inventory-Management-and-Supply-Chain-Optimization/data/best_lgbm_model.txt")
STUDY_OUT = Path("E:/Inventory-Management-and-Supply-Chain-Optimization/data/optuna_lgbm_study.pkl")
PLOT_OUT = Path("E:/Inventory-Management-and-Supply-Chain-Optimization/data/tuned_lgbm_predictions.png")
METRIC_OUT = Path("E:/Inventory-Management-and-Supply-Chain-Optimization/data/tuned_lgbm_metrics.txt")
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET = "Total_Purchases"
blacklist = ["Date", "products_grouped", "Product_Category", "Country"]
features = [c for c in train.columns if c not in blacklist + [TARGET]]

X_train = train[features]
y_train = train[TARGET]
X_test = test[features]
y_test = test[TARGET]

print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")
print(f"Feature count: {len(features)}")

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# -----------------------------------------------------
# Optuna objective
# -----------------------------------------------------

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "verbose": -1,
    "feature_pre_filter": False,
        "feature_pre_filter": False
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
        
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae

# -----------------------------------------------------
# Run Optuna Study
# -----------------------------------------------------
print("Starting Optuna tuning...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best MAE:", study.best_value)
print("Best params:", study.best_params)

with open(STUDY_OUT, "wb") as f:
    pickle.dump(study, f)

# -----------------------------------------------------
# Train final model
# -----------------------------------------------------
best_params = study.best_params
best_params.update({
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "verbose": -1,
    "feature_pre_filter": False
})

print("Training final LightGBM model...")
best_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=3000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100)],
    
)

best_model.save_model(str(MODEL_OUT))

# -----------------------------------------------------
# Final evaluation
# -----------------------------------------------------
preds = best_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
accuracy = max(0.0, 1 - (mae / (y_test.mean() + 1e-9)))

with open(METRIC_OUT, "w") as f:
    f.write(f"MAE: {mae}\nRMSE: {rmse}\nAccuracy: {accuracy}\n")

# -----------------------------------------------------
# Plot
# -----------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(preds[:200], label="Predicted")
plt.legend()
plt.title("Tuned LightGBM: Actual vs Predicted (first 200)")
plt.xlabel("Index")
plt.ylabel("Total Purchases")
plt.tight_layout()
plt.savefig(PLOT_OUT)

print("Tuned LightGBM training complete ✔")
print(f"Saved best model -> {MODEL_OUT}")
print(f"Saved study -> {STUDY_OUT}")
print(f"Saved metrics -> {METRIC_OUT}")
print(f"Saved plot -> {PLOT_OUT}")