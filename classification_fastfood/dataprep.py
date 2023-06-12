import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv("/kaggle/input/fastfood-nutrition/fastfood.csv")
df.head()
df.columns

# Get unique restaurants
df.restaurant.unique()

# Group by restaurant and calculate the average calories
restaurant_calories = df.groupby('restaurant').calories.mean().reset_index()
print(restaurant_calories)

import matplotlib.pyplot as plt

# Create a bar chart of the average calories per restaurant
plt.figure(figsize=(10,6))
plt.bar(restaurant_calories['restaurant'], restaurant_calories['calories'])
plt.xlabel('Restaurant')
plt.ylabel('Average Calories')
plt.title('Average Calories per Restaurant')

# Show the chart
plt.show()

# Create a pie chart of the average calories per restaurant
plt.figure(figsize=(10,6))
plt.pie(restaurant_calories['calories'], labels=restaurant_calories['restaurant'], autopct='%1.1f%%', startangle=90)
plt.axis('equal')

plt.title('Average Calories per Restaurant')
plt.show()

