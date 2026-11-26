import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data" / "melb_data(Uncleaned).csv"

# show all columns when printing
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.max_columns', None)

df = pd.read_csv(DATA_PATH)

print("\n")
print("\n========== Dataset Head ==========\n")
print(df.head())

print("\n")
print("\n========== Dataset Shape ==========\n")
print(df.shape)

print("\n")
print("\n========== Dataset Info ==========\n")
df.info()


print("\n")
print("\n==========Dataset Description ==========\n")
print(df.describe())

#=============================================
# Missing Values Analysis

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df) * 100)

missing_table = pd.DataFrame({
    "Missing Count" : missing_count,
    "Missing %" : missing_percent.round(2)
})
missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending = False)

print("\n========== Missing Values ==========\n")
print(missing_table)

#=============================================
# Duplicate Analysis


duplicates_count = df.duplicated().sum()
print("\n========== Duplicates ==========\n")
print(f"Number of duplicated Rows: {duplicates_count} \n")


#=============================================
# Unique Values 

unique_values = df.nunique()
print("\n========== Unique Values ==========\n")
print(unique_values.to_string())

#=============================================
# Columns Types 

numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns

print("\n========== Columns Types ==========\n")
print("Numeric Columns:", numeric_cols.tolist())
print("Categorical Columns:", categorical_cols.tolist())


#=============================================
# Target Variable Analysis 

print("\n========== Target Variable Analysis ==========\n")

print("\nPrice Statistics:")
print(df["Price"].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], bins=50, kde=True)

plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


#=============================================
# Outliers 

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Price"])

plt.title("Price Outliers")
plt.xlabel("Price")
plt.tight_layout()
plt.show()




