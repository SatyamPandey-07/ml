"""
ML Model Frontend - Flask Web App
Loads trained credit scoring model and provides web interface for predictions
"""

import os
import json
import glob
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Global variables for model
MODEL_DATA = None
PIPELINE = None
FEATURES = None
THRESHOLD = None
METRICS = None

def load_latest_model():
    """Load the most recent trained model"""
    global MODEL_DATA, PIPELINE, FEATURES, THRESHOLD, METRICS
    
    model_dir = "credit_scoring_pipeline_outputs"
    if not os.path.exists(model_dir):
        return False
        
    # Find latest model file
    model_files = glob.glob(os.path.join(model_dir, "best_balanced_model_*.pkl"))
    if not model_files:
        return False
        
    latest_model = max(model_files, key=os.path.getctime)
    
    try:
        MODEL_DATA = joblib.load(latest_model)
        PIPELINE = MODEL_DATA["pipeline"]
        FEATURES = MODEL_DATA["features"]
        THRESHOLD = MODEL_DATA["threshold"]
        METRICS = MODEL_DATA["metrics"]
        print(f"✅ Loaded model: {latest_model}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/')
def index():
    """Main page showing model info and prediction form"""
    if MODEL_DATA is None:
        return render_template('error.html', 
                             error="No trained model found. Please train a model first.")
    
    # Get feature types from the pipeline
    preproc = PIPELINE.named_steps["preprocessor"]
    numeric_features = preproc.transformers[0][2]  # num transformer features
    categorical_features = preproc.transformers[1][2]  # cat transformer features
    
    return render_template('index.html', 
                         metrics=METRICS, 
                         features=FEATURES,
                         numeric_features=numeric_features,
                         categorical_features=categorical_features,
                         threshold=THRESHOLD)

@app.route('/model-info')
def model_info():
    """API endpoint returning model details as JSON"""
    if MODEL_DATA is None:
        return jsonify({"error": "No model loaded"}), 404
    
    return jsonify({
        "metrics": METRICS,
        "features": FEATURES,
        "threshold": float(THRESHOLD),
        "feature_count": len(FEATURES),
        "model_type": METRICS.get("model", "unknown")
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on input data"""
    if MODEL_DATA is None:
        return jsonify({"error": "No model loaded"}), 404
    
    try:
        # Get form data
        input_data = {}
        for feature in FEATURES:
            value = request.form.get(feature, '')
            if value == '':
                input_data[feature] = 0.0  # Default value for missing inputs
            else:
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    input_data[feature] = value  # Keep as string for categorical
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Make prediction
        probabilities = PIPELINE.predict_proba(df)[:, 1]
        probability = float(probabilities[0])
        prediction = int(probability >= THRESHOLD)
        
        # Prepare result
        result = {
            "probability": round(probability, 4),
            "prediction": prediction,
            "prediction_text": "Likely to Default" if prediction == 1 else "Likely to Repay",
            "threshold": float(THRESHOLD),
            "confidence": "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/sample-data')
def sample_data():
    """Generate sample data for testing"""
    try:
        # Read a few rows from the training data for examples
        df = pd.read_csv("beneficiary_credit_features_with_targets.csv")
        samples = []
        for _, row in df[FEATURES].head(3).iterrows():
            sample = {}
            for feature in FEATURES:
                sample[feature] = row[feature]
            samples.append(sample)
        
        # Get unique values for categorical features
        preproc = PIPELINE.named_steps["preprocessor"]
        categorical_features = preproc.transformers[1][2]  # cat transformer features
        categorical_options = {}
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                categorical_options[cat_feature] = sorted(df[cat_feature].unique().tolist())
        
        return jsonify({
            "samples": samples,
            "features": FEATURES,
            "categorical_options": categorical_options
        })
    except Exception as e:
        return jsonify({"error": f"Could not load sample data: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting ML Model Frontend...")
    
    # Load model on startup
    if load_latest_model():
        print(f"📊 Model Info:")
        print(f"   - Type: {METRICS.get('model', 'unknown')}")
        print(f"   - Features: {len(FEATURES)}")
        print(f"   - Threshold: {THRESHOLD:.3f}")
        print(f"   - CV Recall: {METRICS.get('cv_recall', 0):.3f}")
        print(f"   - F2 Score: {METRICS.get('tuned_f2', 0):.3f}")
        print(f"🌐 Starting web server at http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please train a model first.")
        print("Run: python train_pipeline_dynamic.py")