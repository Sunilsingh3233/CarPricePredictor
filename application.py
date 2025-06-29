from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
with open('LinearRegresssionModel.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['name']
le_company = data['le_company']
le_fuel = data['le_fuel']

# Load dataset for dropdown options
df = pd.read_csv('cleaned_car.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(df['company'].unique())
    fuel_types = sorted(df['fuel_type'].unique())
    car_names = sorted(df['name'].unique())

    if request.method == 'POST':
        company = request.form.get('company')
        fuel = request.form.get('fuel')
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kms_driven'))

        # Preprocess inputs
        company_encoded = le_company.transform([company])[0]
        fuel_encoded = le_fuel.transform([fuel])[0]

        # Predict
        features = [[company_encoded, year, kms_driven, fuel_encoded]]
        predicted_price = model.predict(features)[0]

        return render_template('index.html', 
                               companies=companies, 
                               fuel_types=fuel_types, 
                               car_names=car_names, 
                               price=round(predicted_price, 2))

    return render_template('index.html', 
                           companies=companies, 
                           fuel_types=fuel_types, 
                           car_names=car_names, 
                           price=None)

if __name__ == '__main__':
    app.run(debug=True)
