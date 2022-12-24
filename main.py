import numpy as np
import pandas as pd
import io
from flask import Response
from flask import Flask
from flask import Flask, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn import metrics

app = Flask(__name__)


@app.route('/')
@app.route('/home')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/estimator')
@app.route('/estimator.html')
def estimator():
    return render_template('estimator.html')


if __name__ == "__main__":
    app.run(debug=True)

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

'''
#Scatter Plot

scatter_df = pd.DataFrame().assign(Price=df["price"],
                                   Area=df["area"],
                                   Furnishingstatus=df['furnishingstatus'])

fig, ax = plt.subplots(figsize=(10,10))

ax.ticklabel_format(style = 'plain')
scatter = ax.scatter(x=scatter_df["Price"],
                     y=scatter_df["Area"],
                     c=scatter_df["Furnishingstatus"])
          
ax.set_title('Furnishing by Price and Area', fontsize=25)
ax.set_xlabel('Price', fontsize=20)
ax.set_ylabel('Area (sq.ft)', fontsize=20)
ax.legend(*scatter.legend_elements(),title="Furnishing Status")


#Line Plot
line_df = pd.DataFrame().assign(Price=df["price"],
                                   Bedrooms=df["bedrooms"]
                                   )
line_df["Price"] = line_df["Price"].div(1000000).round(2)
line_avg = line_df.groupby(['Bedrooms'], as_index=False).mean()

fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Average Price by Number of Bedrooms', fontsize=25)
ax.set_xlabel('Number of Bedrooms', fontsize=20)
ax.set_ylabel('Average House Price (millions)', fontsize=20)
plt.plot(line_avg.Bedrooms, line_avg.Price)
plt.show

#Histogram
hist_df = pd.DataFrame().assign(Price=df["price"], Area=df["area"])
                            
hist_df["Price"] = hist_df["Price"].div(1000000).round(2)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_title('Square Footage of Houses', fontsize=25)
ax.set_xlabel('Area(sq.ft)', fontsize=20)
ax.set_ylabel('House Price (millions)', fontsize=20)
hist_df["Area"].plot.hist(bins=20)
plt.show                     
'''

# Implementaiton of Regression Model RFR
x = df.iloc[:, 1:]
y = df.iloc[:, :1]

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(x.values, y.values.ravel())

# test the output by changing values
y_pred = regressor.predict(np.array([3500, 3, 2, 1, 1, 0, 0, 1, 1, 2, 0, 0]).reshape(1, 12))

'''
estimate_int = y_pred.astype(np.int64)
estimate_str = str(estimate_int[0])
print("Your estimated price is : " + estimate_str)
print(regressor.score(x.values,y.values))
'''
