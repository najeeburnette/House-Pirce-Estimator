import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Housing.csv")

# List of variables to map
varlist =  ['mainroad', 'guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'prefarea']

# Binarizing yes/no category columns
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

df[varlist] = df[varlist].apply(binary_map)


# Furnishing columm using label encoding
le = LabelEncoder()
df.furnishingstatus = le.fit_transform(df.furnishingstatus)


# Implementaiton of Regression Model RFR
x= df.iloc [:,1:]
y= df.iloc [:,:1]


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x,y.values.ravel())

# test the output by changing values
y_pred = regressor.predict(np.array([3300, 2, 2, 1, 1, 0, 0, 1, 1, 2, 0, 0]).reshape(1, 12)) 
estimate = y_pred[0]
print("Your estimated price is : " + estimate)
