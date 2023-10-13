import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for clarity

# Load the model and scaler
model = pickle.load(open('artifacts/model.pkl', 'rb'))
scaler = pickle.load(open('artifacts/scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Ensure that the form data keys match your model's input features
            data = {key: [value] for key, value in request.form.items() if key != 'Cover_Type'}
            df = pd.DataFrame(data)
            
            # Standardize the input features using the loaded scaler
            df_scaled = scaler.transform(df)
            
            # Make predictions using the model
            prediction = model.predict(df_scaled)
            result = f"Predicted Cover Type: {prediction[0]}"
            return render_template('index.html', result=result)
        except Exception as e:
            return render_template('index.html', result=f"Prediction failed. Error: {str(e)}")
    return render_template('index.html', result="Prediction failed")


if __name__ == '__main__':
    app.run(debug=True)
