from main import df
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

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
