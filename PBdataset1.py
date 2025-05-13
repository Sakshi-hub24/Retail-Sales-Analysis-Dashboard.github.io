import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('PowerBi/retail_sales_merged.csv')
print(df)
#🧹 1. Data Cleaning & Preparation 
#•	Remove duplicates and handle missing values.
print(df.isnull().sum())
print(df.describe())
print(df.info())
df.drop_duplicates(keep='first', inplace=True)

#•	Standardize column names (e.g., lowercase, underscore-separated).
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)

#•	Ensure correct data types (e.g., convert Date to datetime format).
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(df)


#🔄 2. Data Transformation
#•	Create derived columns:
df['total_sales']=df['quantity'] * df['unit_price']
print(df.columns)
#o	Extract Year and Month from Date
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
print(df.columns)

#o	Categorize Age into bins (e.g., 18–25, 26–35, etc.)
bins = [0, 18, 25, 35, 45, 55, 65]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
print(df['age_group'].value_counts())
print(df.columns)

#•	Create a Region Lookup table
# region_lookup = {
#     'North': ['New York', 'Boston', 'Chicago'],
#     'South': ['Miami', 'Atlanta', 'Dallas'],
#     'East': ['Washington', 'Philadelphia', 'Baltimore'],
#     'West': ['Los Angeles', 'San Francisco', 'Seattle']
# }

#📊 3. Exploratory Data Analysis (Python)
#•	Top 5 products by total sales.
top_5_products = df.groupby('product_name')['total_sales'].sum().nlargest(5)
print(top_5_products)
top_5_products.plot(kind='bar', title='Top 5 Products by Total Sales', ylabel='Total Sales', xlabel='Product Name', color='purple')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#•	Monthly sales trend.
monthly_sales = df.groupby(['year', 'month'])['total_sales'].sum().reset_index()
monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
monthly_sales.set_index('date', inplace=True)
monthly_sales.plot(kind='line', title='Monthly Sales Trend', ylabel='Total Sales', xlabel='Date', color='orange')
plt.legend()
plt.show()

#•	Sales performance by region.
sales_by_region = df.groupby('region')['total_sales'].sum()
print(sales_by_region)
sales_by_region.plot(kind='bar', title='Sales Performance by Region', ylabel='Total Sales', xlabel='Region', color='green')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#• Customer demographics analysis (e.g., age group and gender).
# Sales by age group and gender
sales_by_age_gender = df.groupby(['age_group', 'gender'])['total_sales'].sum().unstack()
print(sales_by_age_gender)
sales_by_age_gender.plot(kind='bar', title='Sales by Age Group and Gender', ylabel='Total Sales', xlabel='Age Group', color=['pink', 'lightblue','red'])
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.show()

df.to_csv('cleaned_retail_sales.csv', index=False)
print(df)




