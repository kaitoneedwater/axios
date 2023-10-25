import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('House_data.csv')
model = joblib.load("DecisionTree.pkl")


@app.route('/')
def index():
    cities = sorted(data['city'].unique())
    countries = sorted(data['country'].unique())
    return render_template('index.html', countries=countries, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('bedrooms')
    bathrooms = request.form.get('bathrooms')
    sqft_living = request.form.get('sqft_living')
    sqft_lot = request.form.get('sqft_lot')
    floors = request.form.get('floors')
    waterfront = request.form.get('waterfront')
    view = request.form.get('view')
    condition = request.form.get('condition')
    sqft_above = request.form.get('sqft_above')
    sqft_basement = request.form.get('sqft_basement')
    yr_built = request.form.get('yr_built')
    yr_renovated = request.form.get('yr_renovated')
    city = request.form.get('city')
    country = request.form.get('country')

    input_data = pd.DataFrame([[bedrooms,bathrooms,sqft_living ,sqft_lot ,floors,waterfront,view ,condition ,sqft_above ,sqft_basement ,yr_built,yr_renovated,city,country]],columns=['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'waterfront','view','condition','sqft_above','sqft_basement','yr_built', 'yr_renovated', 'city','country'])

    input_data['city'] = pd.factorize(input_data['city'])[0]
    input_data['country'] = pd.factorize(input_data['city'])[0]

    prediction = model.predict(input_data)[0]
    print(prediction)
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
