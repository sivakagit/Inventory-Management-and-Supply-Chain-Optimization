# === Power BI Files Generator: Product-level, Daily-level, Inventory-level ===
# Run this in the same folder where test_with_predictions.csv (and optional cleaned_retail_data.csv) live.

import pandas as pd
import numpy as np
from scipy.stats import norm
import os

# -----------------------------
# 0) Configuration / parameters
# -----------------------------
TEST_PRED_FILE = "E:/Inventory-Management-and-Supply-Chain-Optimization/data/test_with_predictions.csv"   # REQUIRED input
CLEANED_FILE = "E:/Inventory-Management-and-Supply-Chain-Optimization/data/cleaned_retail_data.csv"       # OPTIONAL (for unit_cost, product metadata)
OUTPUT_PRODUCT = "E:/Inventory-Management-and-Supply-Chain-Optimization/data/powerbi_product_forecast_dataset.csv"
OUTPUT_DAILY = "E:/Inventory-Management-and-Supply-Chain-Optimization/data/powerbi_daily_forecast_dataset.csv"
OUTPUT_INVENTORY = "E:/Inventory-Management-and-Supply-Chain-Optimization/data/powerbi_inventory_dataset.csv"
# inventory settings (tweak for business assumptions)
HORIZON_DAYS = 60            # use last N days of forecast to estimate avg demand
LEAD_TIME_DAYS = 7           # supplier lead time in days (default)
SERVICE_LEVEL = 0.98         # desired service level (z-score used)
ORDERING_COST = 80.0         # cost per order (S) used in EOQ formula
HOLDING_COST_RATE = 0.25     # annual holding rate (fraction of unit cost)
DEFAULT_UNIT_COST = 2.0      # used if no unit_cost available

# -----------------------------
# 1) Load test_with_predictions
# -----------------------------
if not os.path.exists(TEST_PRED_FILE):
    raise FileNotFoundError(f"Required file not found: {TEST_PRED_FILE}")

test = pd.read_csv(TEST_PRED_FILE)
# normalize column names if needed
test.columns = [c.strip() for c in test.columns]

# Ensure the Date column is datetime
if 'Date' not in test.columns:
    raise KeyError("Input file must contain a 'Date' column.")
test['Date'] = pd.to_datetime(test['Date'])

# Standardize predicted column name
if 'preds' in test.columns:
    test['Predicted'] = test['preds']
elif 'Predicted' in test.columns:
    pass
else:
    raise KeyError("Input file must contain predicted column named 'preds' or 'Predicted'.")

# Ensure Actual column exists; try Total_Purchases or Actual
if 'Total_Purchases' in test.columns:
    test['Actual'] = test['Total_Purchases']
elif 'Actual' in test.columns:
    pass
else:
    raise KeyError("Input file must contain actual demand column named 'Total_Purchases' or 'Actual'.")

# Fill missing useful columns if absent
for col in ['product_id','Country','State','Product_Category','Product_Brand','Product_Type','products']:
    if col not in test.columns:
        test[col] = np.nan

# -----------------------------
# 2) Create product-level Power BI dataset
# -----------------------------
prod = test.copy()

# Row-level (per product-date) metrics
prod['AbsError'] = (prod['Actual'] - prod['Predicted']).abs()
# APE: avoid division by 0
prod['APE'] = (prod['AbsError'] / prod['Actual'].replace(0, np.nan)) * 100
prod['APE'] = prod['APE'].fillna(0)

# Time features helpful in Power BI
prod['Year'] = prod['Date'].dt.year
prod['Month'] = prod['Date'].dt.month
prod['MonthName'] = prod['Date'].dt.month_name()
prod['Quarter'] = prod['Date'].dt.quarter
# ISO week number (pandas >= 1.1)
prod['Week'] = prod['Date'].dt.isocalendar().week.astype(int)
prod['DayOfWeek'] = prod['Date'].dt.day_name()

# Select and order columns for Power BI product dataset
product_columns = [
    'Date', 'Year', 'Month', 'MonthName', 'Quarter', 'Week', 'DayOfWeek',
    'product_id', 'products', 'Product_Category', 'Product_Brand', 'Product_Type',
    'Country', 'State',
    'Actual', 'Predicted', 'AbsError', 'APE'
]
# if some columns missing, keep available ones
product_columns = [c for c in product_columns if c in prod.columns]
product_level = prod[product_columns].copy()

# Save product-level dataset
product_level.to_csv(OUTPUT_PRODUCT, index=False)
print(f"Saved product-level Power BI file: {OUTPUT_PRODUCT} (rows: {len(product_level)})")

# -----------------------------
# 3) Create daily-level Power BI dataset
# -----------------------------
daily = product_level.groupby('Date')[['Actual','Predicted']].sum().reset_index()

# Daily error metrics
daily['AbsError'] = (daily['Actual'] - daily['Predicted']).abs()
daily['MAPE'] = (daily['AbsError'] / daily['Actual'].replace(0, np.nan)) * 100
daily['MAPE'] = daily['MAPE'].fillna(0)

# Add time columns
daily['Year'] = daily['Date'].dt.year
daily['Month'] = daily['Date'].dt.month
daily['MonthName'] = daily['Date'].dt.month_name()
daily['Quarter'] = daily['Date'].dt.quarter
daily['Week'] = daily['Date'].dt.isocalendar().week.astype(int)
daily['DayOfWeek'] = daily['Date'].dt.day_name()

# Save daily dataset
daily.to_csv(OUTPUT_DAILY, index=False)
print(f"Saved daily-level Power BI file: {OUTPUT_DAILY} (rows: {len(daily)})")

# -----------------------------
# 4) Build inventory dataset (EOQ, ROP, Safety Stock)
# -----------------------------
# Create forecast_df for inventory calc: product_id, Date, forecast
forecast_df = product_level.rename(columns={'Predicted':'forecast'})[['product_id','Date','forecast']].copy()
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

# allow optional unit_cost merge if cleaned file present
unit_costs = None
if os.path.exists(CLEANED_FILE):
    try:
        cleaned = pd.read_csv(CLEANED_FILE)
        cleaned_cols = [c.strip() for c in cleaned.columns]
        # If cleaned has unit_cost and product identifier, try to merge
        # We attempt to find a unit cost column name commonly named 'unit_cost' or 'Unit_Cost'
        uc_candidates = [c for c in cleaned_cols if 'unit' in c.lower() and 'cost' in c.lower()]
        if uc_candidates:
            # get first candidate
            uc_col = uc_candidates[0]
            # ensure product identifier present in cleaned file (products or product_id)
            if 'product_id' in cleaned_cols:
                unit_costs = cleaned[['product_id', uc_col]].drop_duplicates().rename(columns={uc_col:'unit_cost'})
            elif 'products' in cleaned_cols:
                unit_costs = cleaned[['products', uc_col]].drop_duplicates().rename(columns={uc_col:'unit_cost'})
                # later we'll try merge on 'products' if test contains it
    except Exception as e:
        print("Warning: failed to load cleaned_retail_data for unit cost merge:", e)

# inventory metric computation
def compute_inventory_metrics(forecast_df, horizon_days=HORIZON_DAYS,
                              lead_time_days=LEAD_TIME_DAYS,
                              service_level=SERVICE_LEVEL,
                              ordering_cost=ORDERING_COST,
                              holding_cost_rate=HOLDING_COST_RATE,
                              unit_cost_map=None,
                              default_unit_cost=DEFAULT_UNIT_COST):
    df = forecast_df.copy()
    last_date = df['Date'].max()
    if horizon_days is not None:
        df = df[df['Date'] >= (last_date - pd.Timedelta(days=horizon_days - 1))]
    # daily aggregated forecast per product
    daily = df.groupby(['product_id','Date'])['forecast'].sum().reset_index()
    summary = daily.groupby('product_id')['forecast'].agg(['mean','std','sum']).reset_index()
    summary = summary.rename(columns={'mean':'avg_daily','std':'sigma_daily','sum':'sum_horizon'})
    summary['sigma_daily'] = summary['sigma_daily'].fillna(0.0)
    # z-score
    z = norm.ppf(service_level)
    summary['lead_time'] = lead_time_days
    summary['service_level'] = service_level
    summary['z_value'] = z
    # safety stock
    summary['safety_stock'] = (summary['sigma_daily'] * np.sqrt(lead_time_days) * z).round(3)
    # reorder point
    summary['ROP'] = (summary['avg_daily'] * lead_time_days + summary['safety_stock']).round(3)
    # annual demand
    summary['annual_demand'] = (summary['avg_daily'] * 365).round(1)

    # unit cost logic: merge unit_cost_map if provided
    if unit_cost_map is not None:
        # unit_cost_map should be DataFrame with columns ['product_id','unit_cost'] or ['products','unit_cost']
        if 'product_id' in unit_cost_map.columns:
            u = unit_cost_map[['product_id','unit_cost']].drop_duplicates()
            summary = summary.merge(u, on='product_id', how='left')
        elif 'products' in unit_cost_map.columns:
            # attempt to merge by product name if available in forecast_df
            prod_names = product_level[['product_id','products']].drop_duplicates()
            u = unit_cost_map[['products','unit_cost']].drop_duplicates()
            prod_unit = prod_names.merge(u, on='products', how='left')
            summary = summary.merge(prod_unit[['product_id','unit_cost']], on='product_id', how='left')
        else:
            summary['unit_cost'] = np.nan
        summary['unit_cost'] = summary['unit_cost'].fillna(default_unit_cost)
    else:
        summary['unit_cost'] = default_unit_cost

    # holding cost per unit per year
    summary['holding_cost_per_unit'] = (holding_cost_rate * summary['unit_cost']).round(4)

    # EOQ
    # guard H small -> add tiny epsilon
    eps = 1e-9
    summary['EOQ'] = np.sqrt((2.0 * summary['annual_demand'] * ordering_cost) / (summary['holding_cost_per_unit'] + eps))
    summary['EOQ'] = summary['EOQ'].replace([np.inf, -np.inf], np.nan).fillna(0)
    summary['EOQ'] = summary['EOQ'].round(0).astype(int)

    # recommended order quantity
    summary['recommended_order_qty'] = summary['EOQ'].apply(lambda x: max(1, int(x)))

    # reorder columns and tidy
    summary = summary.reset_index(drop=True)
    # round/format
    for c in ['avg_daily','sigma_daily','safety_stock','ROP','annual_demand']:
        if c in summary.columns:
            summary[c] = summary[c].round(3)
    return summary

# attempt to use unit_costs if found earlier
inventory_df = compute_inventory_metrics(forecast_df,
                                         horizon_days=HORIZON_DAYS,
                                         lead_time_days=LEAD_TIME_DAYS,
                                         service_level=SERVICE_LEVEL,
                                         ordering_cost=ORDERING_COST,
                                         holding_cost_rate=HOLDING_COST_RATE,
                                         unit_cost_map=unit_costs,
                                         default_unit_cost=DEFAULT_UNIT_COST)

# Save inventory dataset
inventory_df.to_csv(OUTPUT_INVENTORY, index=False)
print(f"Saved inventory Power BI file: {OUTPUT_INVENTORY} (rows: {len(inventory_df)})")

# -----------------------------
# 5) Summary of outputs
# -----------------------------
print("\n=== Outputs created ===")
print(f"1) {OUTPUT_PRODUCT}  - product-level forecast rows: {len(product_level)}")
print(f"2) {OUTPUT_DAILY}    - daily aggregated rows: {len(daily)}")
print(f"3) {OUTPUT_INVENTORY}- inventory metrics rows: {len(inventory_df)}")
