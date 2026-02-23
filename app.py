from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/mlr_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', inputs={})

@app.route('/predict', methods=['POST'])
def predict():
    user_inputs = request.form.to_dict()
    
    try:        
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)[0]
        
        return render_template('index.html', 
                               prediction_text=f'Predicted Median House Value: ${prediction * 100000:,.2f}',
                               inputs=user_inputs)
                               
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error in processing input: {e}',
                               inputs=user_inputs)

if __name__ == "__main__":
    app.run(debug=True)