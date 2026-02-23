from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the scaler and the best regression model
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/mlr_model.pkl')

@app.route('/')
def home():
    # Pass an empty dictionary on the initial load so the template doesn't complain
    return render_template('index.html', inputs={})

@app.route('/predict', methods=['POST'])
def predict():
    # Capture the exact form data to send back to the user
    user_inputs = request.form.to_dict()
    
    try:
        # Extract features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        
        # Scale and Predict
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)[0]
        
        return render_template('index.html', 
                               prediction_text=f'Predicted Median House Value: ${prediction * 100000:,.2f}',
                               inputs=user_inputs) # Injecting the state back!
                               
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error in processing input: {e}',
                               inputs=user_inputs)

if __name__ == "__main__":
    app.run(debug=True)