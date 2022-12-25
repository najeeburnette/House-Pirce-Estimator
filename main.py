import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

app = Flask(__name__)

df = pd.read_csv("Housing.csv")

# List of variables to map
varlist = ['mainroad', 'guestroom', 'basement',
           'hotwaterheating', 'airconditioning', 'prefarea']


# Binarizing yes/no category columns
def binary_map(x):
    return x.map({'yes': 1, "no": 0})


df[varlist] = df[varlist].apply(binary_map)

# Furnishing columm using label encoding
le = LabelEncoder()
df.furnishingstatus = le.fit_transform(df.furnishingstatus)

# Implementaiton of Regression Model RFR
x = df.iloc[:, 1:]
y = df.iloc[:, :1]

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(x.values, y.values.ravel())


def value_estimaiton(area, bedrooms, bathrooms, stories, mainroad,
                     guestroom, basement, hotwater, aircon,
                     parking, highdemand, furnishing):
    y_pred = regressor.predict(np.array([area, bedrooms, bathrooms, stories,
                                         mainroad, guestroom, basement, hotwater,
                                         aircon, parking, highdemand, furnishing]).reshape(1, 12))

    estimate_int = y_pred.astype(np.int64)
    estimate_str = str(estimate_int[0])
    return estimate_str
    # print(regressor.score(x.values,y.values))


@app.route('/')
@app.route('/home')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/estimator', methods=["POST", "GET"])
def estimator():
    if request.method == "POST":
        area = request.form["area"]
        bedrooms = request.form["bedrooms"]
        bathrooms = request.form["bathrooms"]
        stories = request.form["stories"]
        mainroad = request.form["mainroad"]
        guestroom = request.form["guestroom"]
        basement = request.form["basement"]
        hotwater = request.form["hotwater"]
        aircon = request.form["aircon"]
        parking = request.form["parking"]
        highdemand = request.form["highdemand"]
        furnishing = request.form["furnishing"]
        final_estimation = value_estimaiton(area, bedrooms, bathrooms, stories, mainroad,
                                            guestroom, basement, hotwater, aircon, parking,
                                            highdemand, furnishing)
        return render_template('estimator.html', estimated_value=final_estimation)
        # area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwater,aircon, parking, highdemand,furnishing
    else:
        return render_template('estimator.html')


if __name__ == "__main__":
    app.run(debug=True)
