# Credit Scoring Pipeline

A dynamic retrainable credit scoring pipeline that automatically detects features, engineers new variables, trains multiple models, and optimizes decision thresholds for credit default prediction.

## Features

### 🤖 Machine Learning Pipeline
- **Automatic Feature Detection**: Uses keyword patterns to identify consumption, repayment, loan, and repeat borrower features
- **Feature Engineering**: Creates aggregated statistics (mean, median, std) and derived features like utilization ratios
- **Multiple Model Support**: RandomForest (baseline), LightGBM, and XGBoost with automatic model selection
- **Imbalanced Data Handling**: SMOTE oversampling for balanced training
- **Threshold Optimization**: F2-score optimization for better recall/precision balance
- **Model Calibration**: Probability calibration for reliable confidence scores
- **Versioned Artifacts**: Timestamped model saves with complete metadata

### 🌐 Web Frontend
- **Interactive Prediction Interface**: Real-time credit risk predictions
- **Smart Input Fields**: Dropdowns for categorical features, number inputs for numerical
- **Model Dashboard**: Live metrics display (F2-score, recall, precision, threshold)
- **Sample Data Loading**: Test with realistic data from training set
- **Feature Breakdown**: Visual separation of numerical vs categorical features
- **API Endpoints**: RESTful JSON API for programmatic access

### 📊 Integration Analysis
- **Data Utilization**: 15/51 features used (29.4% of available data)
- **Pipeline Integration**: 100% of ML components utilized
- **Frontend Integration**: 100% of model features exposed in UI
- **Overall Integration Score**: 76.5%

## Project Structure

```
d:\ml\
├── beneficiary_credit_features_with_targets.csv    # Training data (52 columns)
├── train_pipeline_dynamic.py                      # Main training script
├── app.py                                          # Flask web frontend
├── analyze_integration.py                         # Integration analysis tool
├── requirements.txt                                # Python dependencies
├── README.md                                       # This file
├── templates/                                      # Web UI templates
│   ├── index.html                                 # Main prediction interface
│   └── error.html                                 # Error handling page
└── credit_scoring_pipeline_outputs/               # Generated artifacts
    ├── best_balanced_model_v20251015TXXXXXX.pkl   # Trained model
    └── training_metrics_log.csv                   # Training history (optional)
```

## Setup Instructions

### 1. Environment Setup

**Option A: Using existing venv (recommended)**
```powershell
# The workspace already has a configured venv at D:\ml\.venv
# Activate it (optional, scripts use full path)
D:\ml\.venv\Scripts\Activate.ps1

# Verify installation
D:\ml\.venv\Scripts\python.exe --version
```

**Option B: Fresh virtual environment**
```powershell
# Create new venv
python -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Install Dependencies

**Core dependencies (required):**
```powershell
D:\ml\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Enhanced dependencies (optional, for better performance):**
```powershell
D:\ml\.venv\Scripts\python.exe -m pip install lightgbm xgboost
```

**Start the web frontend:**
```powershell
D:\ml\.venv\Scripts\python.exe app.py
# Then open http://localhost:5000 in your browser
```

## Usage

### 🌐 Web Interface (Recommended)

**Start the web frontend:**
```powershell
D:\ml\.venv\Scripts\python.exe app.py
```

Then open http://localhost:5000 in your browser for:
- Interactive model dashboard with live metrics
- Real-time predictions with proper categorical/numerical inputs
- Sample data loading and testing
- Model performance visualization

### 🤖 Training a Model

**Basic training with existing CSV:**
```powershell
D:\ml\.venv\Scripts\python.exe train_pipeline_dynamic.py
```

**Expected output:**
```
Running retrain_from_csv() ...
✅ Model trained & saved at: credit_scoring_pipeline_outputs\best_balanced_model_vXXXXXXXXTXXXXXX.pkl
{
  "model": "rf|lgbm|xgb",
  "cv_recall": 0.854,
  "tuned_threshold": 0.385,
  "tuned_f2": 0.905,
  "tuned_precision_class1": 0.655,
  "tuned_recall_class1": 1.0,
  "tuned_f1_class1": 0.792,
  "train_time_utc": "2025-10-15T13:13:36.470490Z"
}
```

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load saved model
model_data = joblib.load("credit_scoring_pipeline_outputs/best_balanced_model_vXXXXXXXX.pkl")
pipeline = model_data["pipeline"]
threshold = model_data["threshold"]
features = model_data["features"]

# Make predictions on new data
new_data = pd.read_csv("new_applicants.csv")
probabilities = pipeline.predict_proba(new_data[features])[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Results: 0 = likely to repay, 1 = likely to default
```

### Programmatic Retraining

```python
from train_pipeline_dynamic import retrain_from_csv, update_with_new_data

# Retrain on existing data
result = retrain_from_csv()
print(f"New model: {result['artifact']}")

# Add new data and retrain
new_df = pd.read_csv("additional_data.csv")
result = update_with_new_data(new_df, save_csv=True)
```

## Data Requirements

The CSV file should contain:

**Required columns:**
- `default_flag`: Target variable (1/0, "yes"/"no", "true"/"false")

**Feature columns (detected automatically):**
- **Consumption features**: Keywords like "energy", "electricity", "recharge", "mobile", "bill", "utility"
- **Repayment features**: Keywords like "repay", "overdue", "delinquent", "paid", "installment"
- **Loan features**: Keywords like "loan", "disbursal", "principal", "balance", "outstanding"
- **Repeat borrower features**: Keywords like "repeat", "num_past_loans", "num_loans"
- **Categorical features**: Low-cardinality categorical variables (< 30 unique values)

## Model Selection Logic

1. **LightGBM**: Preferred if available (fast, good performance)
2. **XGBoost**: Secondary choice if LightGBM unavailable
3. **RandomForest**: Fallback baseline (always available)

Models are compared using cross-validated recall, and the best performer is selected automatically.

## Performance Metrics

The pipeline optimizes for:
- **F2-score**: Weighted towards recall (important for credit risk)
- **Calibrated probabilities**: Reliable confidence estimates
- **Threshold tuning**: Optimal decision boundary for business needs

## Troubleshooting

**"ModuleNotFoundError" errors:**
```powershell
D:\ml\.venv\Scripts\python.exe -m pip install --upgrade pip
D:\ml\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**LightGBM/XGBoost installation issues on Windows:**
```powershell
# Try pre-compiled wheels
D:\ml\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
D:\ml\.venv\Scripts\python.exe -m pip install lightgbm xgboost
```

**CSV format issues:**
- Ensure `default_flag` column exists
- Check for consistent column names (no extra spaces)
- Verify target values are 1/0 or "yes"/"no"

## Model Outputs

Each training run creates:
- **Model artifact**: `best_balanced_model_v{timestamp}.pkl`
- **Metrics**: JSON performance summary
- **Features used**: List of selected features
- **Optimal threshold**: Tuned decision boundary

## Integration Analysis

Run the integration analysis tool to see how much of your data pipeline is utilized:

```powershell
D:\ml\.venv\Scripts\python.exe analyze_integration.py
```

**Current Integration Metrics:**
- 📊 **Data Utilization**: 15/51 features (29.4%)
  - 7 engineered numerical features (repayment stats, consumption stats, utilization)
  - 8 original categorical features (gender, education, location, etc.)
- ⚙️ **Pipeline Utilization**: 10/10 components (100%)
- 🌐 **Frontend Integration**: 15/15 model features (100%)
- 🎯 **Overall Score**: 76.5%

**Unused Features (36 available):**
Age, assets, bank data, UPI transactions, weather risk, poverty index, and more demographic/financial features that could potentially improve model performance.

## Next Steps

- **Feature Expansion**: Integrate more of the 36 unused features to improve model accuracy
- **Model Monitoring**: Track performance over time via the web dashboard
- **Feature Importance**: Analyze which features drive predictions most
- **A/B Testing**: Compare model versions using the web interface
- **Automated Retraining**: Schedule periodic model updates
- **Production Deployment**: Scale the Flask app for production use