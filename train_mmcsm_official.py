import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

# --- 0. Setup Logging and Environment ---
if not os.path.exists('models'):
    os.makedirs('models')

log_filename = "mmcsm_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

logger.info("="*50)
logger.info("STARTING mmCSM-PPI OFFICIAL TRAINING PIPELINE")
logger.info("="*50)

# --- 1. Data Preparation & Symmetry Augmentation ---
logger.info("🚀 [1/6] Loading and Cleaning Data...")
try:
    df = pd.read_csv('data/processed/mmcsm_features.csv')
except FileNotFoundError:
    logger.error("Data file 'data/processed/mmcsm_features.csv' not found.")
    exit()

# --- DATA SANITIZER ---
# 1. Replace any 'inf' (from log calculations) with NaN
# 2. Drop any row that contains a NaN in the target OR features
initial_count = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
clean_count = len(df)

if clean_count < initial_count:
    logger.warning(f"⚠️ Dropped {initial_count - clean_count} rows containing NaNs or Infs.")

# SYMMETRY AUGMENTATION: Teaching the model reversibility
# ΔΔG(A→B) = -ΔΔG(B→A)
rev_df = df.copy()
rev_df['ddG_target'] = -rev_df['ddG_target']

# Flip the 'Delta' columns (Mutant - WildType becomes WildType - Mutant)
delta_cols = [c for c in df.columns if c.startswith('d_')]
for col in delta_cols:
    rev_df[col] = -rev_df[col]

full_df = pd.concat([df, rev_df], ignore_index=True)
X = full_df.drop(columns=['pdb_id', 'mutation', 'ddG_target'])
y = full_df['ddG_target']

logger.info(f"📊 Dataset: {clean_count} clean original rows -> {len(full_df)} augmented rows.")
logger.info(f"🧬 Initial Feature Count: {X.shape[1]}")

# --- 2. Stepwise Feature Selection (RFECV) ---
logger.info("\n✂️ [2/6] Starting Recursive Feature Elimination (Stepwise Greedy)...")
# This pruned set will help avoid the "Curse of Dimensionality"
selector_base = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# verbose=1 will print a message for every 5 features removed
selector = RFECV(estimator=selector_base, step=5, cv=5, 
                 scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
selector.fit(X, y)

X_selected = X.iloc[:, selector.support_]
logger.info(f"✅ Selection Complete. Kept {X_selected.shape[1]} features.")

# --- 3. Hyperparameter Tuning (Grid Search) ---
logger.info("\n🔍 [3/6] Tuning Hyperparameters via GridSearchCV...")
param_grid = {
    'n_estimators': [1000],          # Forest size
    'max_features': ['sqrt', 0.3, 0.5], # Breadth of feature search
    'min_samples_split': [2, 5],      # Depth control
}

grid_search = GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1), 
                           param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_selected, y)
best_model = grid_search.best_estimator_

logger.info(f"🏆 Best Parameters: {grid_search.best_params_}")

# --- 4. Official 10-Fold Evaluation ---
logger.info("\n🧪 [4/6] Running Official 10-Fold Cross-Validation...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(best_model, X_selected, y, cv=kf, n_jobs=-1)

# --- 5. Final Metrics Calculation ---
logger.info("📈 [5/6] Calculating Performance Metrics...")
p_r, _ = pearsonr(y, y_pred)
s_rho, _ = spearmanr(y, y_pred)
k_tau, _ = kendalltau(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

logger.info("\n" + "="*40)
logger.info("RESULTS: mmCSM-PPI REPLICA")
logger.info("="*40)
logger.info(f"Pearson R:  {p_r:.4f}")
logger.info(f"Spearman ρ: {s_rho:.4f}")
logger.info(f"Kendall τ:  {k_tau:.4f}")
logger.info(f"RMSE:       {rmse:.4f} kcal/mol")
logger.info("="*40)

# --- 6. Saving Artifacts & Plotting ---
logger.info("\n💾 [6/6] Saving Artifacts...")
joblib.dump(best_model, 'models/mmcsm_final_model.pkl')
joblib.dump(selector, 'models/feature_selector.pkl')

# Save Regression Plot
plt.figure(figsize=(10, 8))
sns.regplot(x=y, y=y_pred, scatter_kws={'alpha':0.3, 'color':'navy'}, line_kws={'color':'red'})
plt.title(f"mmCSM-PPI (Pearson R = {p_r:.3f})")
plt.xlabel("Experimental ΔΔG (kcal/mol)")
plt.ylabel("Predicted ΔΔG (kcal/mol)")
plt.savefig('performance_plot.png')

# Save Feature Importance Plot
importances = pd.Series(best_model.feature_importances_, index=X_selected.columns)
top_20 = importances.sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 6))
top_20.plot(kind='barh', color='skyblue')
plt.title("Top 20 Predictive Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')

logger.info(f"✅ Done. Results saved to {log_filename} and plots.")