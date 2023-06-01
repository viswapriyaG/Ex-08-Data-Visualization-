# Ex-08-Data-Visualization-

## AIM
To Perform Data Visualization on the given dataset and save the data to a file. 

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature generation and selection techniques to all the features of the data set
### STEP 4
Apply data visualization techniques to identify the patterns of the data.


# CODE
### Developed by: Viswa Priya G
### Register no: 212221220061

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Superstore.csv',encoding="latin-1")
df
df.head()
df.info()

#removing unnecessary data variables

df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Customer ID',axis=1,inplace=True)
df.drop('Customer Name',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Postal Code',axis=1,inplace=True)
df.drop('Product ID',axis=1,inplace=True)
df.drop('Product Name',axis=1,inplace=True)
df.drop('Order Date',axis=1,inplace=True)
df.drop('Ship Date',axis=1,inplace=True)
print("Updated dataset")
df

df.isnull().sum()

#detecting and removing outliers

plt.figure(figsize=(8,8))
plt.title("Data with outliers")
df.boxplot()
plt.show()

plt.figure(figsize=(8,8))
cols = ['Sales','Quantity','Discount','Profit']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

#data visualization
# Which segment has highest sales?
sns.lineplot(x="Segment",y="Sales",data=df,marker='o')
plt.title("Segment vs Sales")
plt.xticks(rotation = 90)
plt.show()

sns.barplot(x="Segment",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()

# Which city has highest profit?
df.shape
df1 = df[(df.Profit >= 60)]
df1.shape
plt.figure(figsize=(30,8))
states=df1.loc[:,["City","Profit"]]
states=states.groupby(by=["City"]).sum().sort_values(by="Profit")
sns.barplot(x=states.index,y="Profit",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("City")
plt.ylabel=("Profit")
plt.show()

# Which ship mode is profitable?
sns.barplot(x="Ship Mode",y="Profit",data=df)
plt.show()

sns.lineplot(x="Ship Mode",y="Profit",data=df)
plt.show()

sns.violinplot(x="Profit",y="Ship Mode",data=df)

sns.pointplot(x=df["Profit"],y=df["Ship Mode"])

# Sales of the product based on region
states=df.loc[:,["Region","Sales"]]
states=states.groupby(by=["Region"]).sum().sort_values(by="Sales")
sns.barplot(x=states.index,y="Sales",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("Region")
plt.ylabel=("Sales")
plt.show()

#pie chart

df.groupby(['Region']).sum().plot(kind='pie', y='Sales',figsize=(6,9),pctdistance=1.7,labeldistance=1.2)

# Find the relation between sales and profit
df["Sales"].corr(df["Profit"])

df_corr = df.copy()
df_corr = df_corr[["Sales","Profit"]]
df_corr.corr()

#pair plot

sns.pairplot(df_corr, kind="scatter")
plt.show()

# Heatmap

df4=df.copy()

#encoding

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder
oe=OrdinalEncoder()

df4["Ship Mode"]=oe.fit_transform(df[["Ship Mode"]])
df4["Segment"]=oe.fit_transform(df[["Segment"]])
df4["City"]=le.fit_transform(df[["City"]])
df4["State"]=le.fit_transform(df[["State"]])
df4['Region'] = oe.fit_transform(df[['Region']])
df4["Category"]=oe.fit_transform(df[["Category"]])
df4["Sub-Category"]=le.fit_transform(df[["Sub-Category"]])

#scaling

from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=pd.DataFrame(sc.fit_transform(df4),columns=['Ship Mode', 'Segment', 'City', 'State','Region',
                                               'Category','Sub-Category','Sales','Quantity','Discount','Profit'])

#Heatmap

plt.subplots(figsize=(12,7))
sns.heatmap(df5.corr(),cmap="PuBu",annot=True)
plt.show()

# Find the relation between sales and profit based on the following category.

Segment

grouped_data = df.groupby('Segment')[['Sales', 'Profit']].mean()
#Create a bar chart of the grouped data
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data['Sales'], label='Sales')
ax.bar(grouped_data.index, grouped_data['Profit'], bottom=grouped_data['Sales'], label='Profit')
ax.set_xlabel('Segment')
ax.set_ylabel('Value')
ax.legend()
plt.show()

City

grouped_data = df.groupby('City')[['Sales', 'Profit']].mean()
#Create a bar chart of the grouped data
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data['Sales'], label='Sales')
ax.bar(grouped_data.index, grouped_data['Profit'], bottom=grouped_data['Sales'], label='Profit')
ax.set_xlabel('City')
ax.set_ylabel('Value')
ax.legend()
plt.show()

States

grouped_data = df.groupby('State')[['Sales', 'Profit']].mean()
#Create a bar chart of the grouped data
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data['Sales'], label='Sales')
ax.bar(grouped_data.index, grouped_data['Profit'], bottom=grouped_data['Sales'], label='Profit')
ax.set_xlabel('State')
ax.set_ylabel('Value')
ax.legend()
plt.show()

Segment and Ship mode

grouped_data = df.groupby(['Segment', 'Ship Mode'])[['Sales', 'Profit']].mean()
pivot_data = grouped_data.reset_index().pivot(index='Segment', columns='Ship Mode', values=['Sales', 'Profit'])
#Create a bar chart of the grouped data
fig, ax = plt.subplots()
pivot_data.plot(kind='bar', ax=ax)
ax.set_xlabel('Segment')
ax.set_ylabel('Value')
plt.legend(title='Ship Mode')
plt.show()

Segment,ship mode and region

grouped_data = df.groupby(['Segment', 'Ship Mode','Region'])[['Sales', 'Profit']].mean()
pivot_data = grouped_data.reset_index().pivot(index=['Segment', 'Ship Mode'], columns='Region', values=['Sales', 'Profit'])
sns.set_style("whitegrid")
sns.set_palette("Set1")
pivot_data.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.xlabel('Segment - Ship Mode')
plt.ylabel('Value')
plt.legend(title='Region')
plt.show()


# OUPUT
Initial dataset

![op1](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/a1cbf54a-f482-4243-942a-2e42140df3c5)
![op2](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/19f87952-08e5-4b32-be25-948e0e52221c)
![op3](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/efe1aaa7-507a-498d-8159-8ee34f2a6ad5)
![op4](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/8a3bf347-efcf-4d5c-97b2-5e266240019e)

Updated dataset
![op5](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/4c68ecf8-afc5-4764-bfe5-59e522a6a80a)
![op6](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/f666b89c-3fde-43f5-bd04-b5eaba2280e8)

![op7](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/ace91567-dd85-4725-b06d-1c9fc2fb734b)
![op8](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/2cb6e339-5d9b-40b9-9010-e7a32c655d0b)

Which segment has highest sales?
![op9](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/de5ff641-4ea9-4c71-9475-efaf02002820)

![op10](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/cbba0805-72f3-4c2e-af1e-2c6d105cb7ed)

Which city has highest profit?
![op11](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/3b4dc8fe-5cee-4d1c-8cf3-6cc992628f35)

Profitable ship mode
![op12](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/7c049251-c193-4463-b0aa-af72b7aa77a0)
![op13](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/49abbbd0-0488-4c59-a4fd-ffb85e19de42)
![op14](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/3ddbb46d-5d7b-450b-9b4a-8a4be747eb0f)
![op15](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/be3aa34a-9658-451b-a568-7c870b03820e)

Sales of the product based on region
![op16](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/6d932e87-edbf-42c3-b45b-5a3d7ba04397)
![op17](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/e3807c68-1d93-48da-a55d-8c92ac553c95)

Relation between sales and profit
![op18](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/61f86614-8bb7-4d7e-917b-ed37ccc1bfdb)

Heatmap
![op19](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/1f9cc61d-6219-499e-8e71-7aa72a1cf13e)

Relation Between Sales And Profit Based On The Following Category

1.Segment
![op20](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/919be293-c9fc-425d-b200-97284c965093)

2.City
![op21](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/533380ac-f003-4072-a4d7-135a36475766)

3.States
![op22](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/016a6df5-a11e-404c-b91b-325f3e4b6ce6)

4.Segment and ship mode
![op23](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/82181155-5cc5-48cb-b89d-ef0dec9644aa)

5.Segment,ship mode and region
![op24](https://github.com/viswapriyaG/Ex-08-Data-Visualization-/assets/131427787/70709c99-06ae-4778-924a-d08f1589d83c)

# RESULT:
Thus using the given dataset data visualization is performed for following questions.
