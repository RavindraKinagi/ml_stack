import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
crime = pd.read_csv('../input/crime-classifcication/Crime1.csv', usecols=['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address'])

# Convert 'Dates' column to proper datetime format
crime['Dates'] = pd.to_datetime(crime['Dates'])

# Data Audit
crime.info()
crime.dtypes
crime.describe()

# Most common category of crime
crime_category = crime.groupby('Category')['Category'].count().sort_values(ascending=False)
plt.figure(figsize=(10, 8))
crime_category.plot(kind='barh')
plt.xlabel('Count')
plt.title('Number of times each Crime Category took place')
plt.show()
print('The most common crime category is LARCENY/THEFT')

# Day of the week with the most crimes
plt.figure(figsize=(10, 8))
sns.countplot(crime['DayOfWeek'])
plt.title('Day of the week on which most crimes take place')
plt.show()
print('Most crimes take place on Saturday')

# Most famous district in terms of crime
plt.figure(figsize=(12, 8))
sns.countplot(crime['PdDistrict'])
plt.show()
print('The SOUTHERN district is famous in terms of crimes')

# Descriptions in LARCENY/THEFT cases
larceny_descript = crime.loc[crime['Category'] == 'LARCENY/THEFT', 'Descript'].value_counts()
larceny_descript

# Resolutions in LARCENY/THEFT cases
category_resolution = crime.groupby(['Category', 'Resolution'])['Category'].count()
larceny_cases = category_resolution['LARCENY/THEFT']
larceny_cases.plot(kind='bar')
plt.show()
print('There was no resolution for the majority of LARCENY/THEFT cases')

# Day of the week with maximum LARCENY/THEFT cases
category_day = crime.groupby(['Category', 'DayOfWeek'])['Category'].count()
larceny_day = category_day['LARCENY/THEFT']
plt.figure(figsize=(10, 8))
larceny_day.plot(kind='bar')
plt.ylabel('Count')
plt.show()

# Specific address in SOUTHERN district where LARCENY/THEFT crimes take place
larceny_address = crime.groupby(['Category', 'PdDistrict', 'Address'])['Address'].count()['LARCENY/THEFT']['SOUTHERN'].sort_values(ascending=False)
larceny_address
print('There is no specific address identified for LARCENY/THEFT crimes in the SOUTHERN district')

# Perform any additional preprocessing steps based on the analysis

# Drop unnecessary columns
crime.drop(['Descript', 'Resolution'], axis=1, inplace=True)

# Encode categorical features if necessary

# Split the data into features and target
X = crime.drop('Category', axis=1)
y = crime['Category']

