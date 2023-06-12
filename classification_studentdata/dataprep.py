# 1. Data Acquisition
import pandas as pd

# Load Data for Analysis
df = pd.read_csv('../input/student-performance-data/student_data.csv')

# 2. Data Cleansing
# Check for Missing Values
df.isnull().sum()

# There are no missing values in this dataset

# Checking for Outliers
plt.figure(figsize=(15,8))
df.boxplot(color='b', sym='r+')

# Remove outliers using IQR method
Q1 = df.loc[:, df.columns != 'failures'].quantile(0.25)
Q3 = df.loc[:, df.columns != 'failures'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
df2 = df[~((df < lower_bound) | (df > upper_bound))]

plt.figure(figsize=(15,8))
df2.boxplot(color='g', sym='rx')
sns.boxplot(x='absences', data=df2)

df3 = df2[df2['absences'] < 16]
sns.boxplot(x='absences', data=df3)

# 3. Data Exploration
# Data Statistics
df.describe().T

# Visualizing Data Correlation
plt.figure(figsize=(20,20))
sns.heatmap(df2.corr(), vmin=-1, cmap="plasma_r", annot=True)

# 4. Data Analysis
# Q_1: Does gender affect the average score?
plt.figure(figsize=(6,5))
sns.barplot(x='sex', y='grades average', data=df2, errwidth=1, saturation=1, palette='Blues_d') 
plt.title('sex vs grades average \n')
plt.show()

# Q_2: Does age affect final grade?
b = sns.swarmplot(x='age', y='G3', hue='sex', data=df2)
b.axes.set_title('Does age affect final grade?\n', fontsize=20)
b.set_xlabel('Age', fontsize=20)
b.set_ylabel('Final Grade', fontsize=20)
plt.show()

# Q_3: Does the cohabitation status of parents affect the student's grades?
plt.figure(figsize=(6,6))
f = df2.loc[df2['Pstatus'] == 'A'].count()[0]
m = df2.loc[df2['Pstatus'] == 'T'].count()[1]
plt.style.use('ggplot')
plt.pie([f, m], labels=['A', 'T'], explode=[0.1, 0.1], startangle=0, labeldistance=1.2, autopct='%.2f %%')
plt.title('Pstatus\n\n A  vs  T ')
plt.show()

plt.figure(figsize=(6,6))
sns.barplot(x='Pstatus', y='grades average', data=df2, errwidth=1, saturation=1, palette='Blues_d') 
plt.title('Pstatus vs grades average')

# Q_4: Does travel time affect student grades?
plt.figure(figsize=(8,5))
sns.boxplot(df3['traveltime'], df3['grades average'], color='g')
plt.show()

# Q_5: Does the education of the father and mother affect the student's grades?
plt.figure(figsize=(10,8))
order_by = df.groupby('Fedu')['G1'].median().sort_values(ascending=False).index
sns.boxplot(x=df['Fedu'], y=df['G1'], order=order_by)
plt.xticks(rotation=90)
plt.title('

