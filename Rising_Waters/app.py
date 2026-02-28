"""
Flask Web Application for Flood Prediction

Routes:
- "/" : Main page with prediction form
- "/predict" : POST endpoint for predictions
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import sys

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None


def load_model_and_scaler():
    """Load the trained model and scaler from saved files"""
    global model, scaler, feature_names
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'flood_model.pkl')
    
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please train the model first by running: python train_model.py")
        return False
    
    try:
        # Load model
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Load scaler (we'll need to pickle it along with the model)
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully!")
        
        # Feature names (in order)
        feature_names = [
            'rainfall_mm',
            'river_level_m',
            'soil_moisture_percent',
            'temperature_c',
            'humidity_percent',
            'elevation_m',
            'drainage_capacity_index'
        ]
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def get_flood_risk_level(probability):
    """
    Determine flood risk level based on probability
    
    Args:
        probability (float): Predicted probability of flood
    
    Returns:
        tuple: (risk_level, color, description)
    """
    if probability < 0.33:
        return "Low", "#27ae60", "Low risk of flooding. Conditions are favorable."
    elif probability < 0.66:
        return "Moderate", "#f39c12", "Moderate risk of flooding. Be cautious."
    else:
        return "High", "#e74c3c", "High risk of flooding. Take preventive measures!"


@app.route('/')
def index():
    """
    Render the main prediction form page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle flood prediction request
    
    Expected JSON format:
    {
        "rainfall_mm": float,
        "river_level_m": float,
        "soil_moisture_percent": float,
        "temperature_c": float,
        "humidity_percent": float,
        "elevation_m": float,
        "drainage_capacity_index": float
    }
    
    Returns:
        JSON with prediction results
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Extract features in order
        features = []
        for feature_name in feature_names:
            if feature_name not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing field: {feature_name}'
                }), 400
            
            try:
                feature_value = float(data[feature_name])
                features.append(feature_value)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid data type for {feature_name}'
                }), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if scaler:
            features_array = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        flood_probability = probability[1]  # Probability of flood (class 1)
        risk_level, color, description = get_flood_risk_level(flood_probability)
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'flood_risk_probability': round(float(flood_probability), 4),
            'risk_level': risk_level,
            'color': color,
            'description': description,
            'no_flood_probability': round(float(probability[0]), 4),
            'input_data': {
                'rainfall_mm': features[0],
                'river_level_m': features[1],
                'soil_moisture_percent': features[2],
                'temperature_c': features[3],
                'humidity_percent': features[4],
                'elevation_m': features[5],
                'drainage_capacity_index': features[6]
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌊 RISING WATERS - FLOOD PREDICTION WEB APPLICATION")
    print("="*60)
    
    # Load model and scaler
    if load_model_and_scaler():
        print("✅ Application ready to serve predictions!")
        print("\n🚀 Starting Flask server...")
        print("📍 Server running at: http://localhost:5000")
        print("📍 Open browser and go to: http://localhost:5000")
        print("\n(Press Ctrl+C to stop the server)\n")
        
        # Run the app
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please train the model first.")
        sys.exit(1)
