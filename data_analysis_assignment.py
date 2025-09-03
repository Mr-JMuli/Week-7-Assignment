# Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame

    print(" Dataset loaded successfully!\n")
except FileNotFoundError:
    print(" Error: Dataset not found. Please check the file path.")
except Exception as e:
    print(f" An unexpected error occurred: {e}")

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head(), "\n")

# Check dataset info
print("Dataset Info:")
print(df.info(), "\n")

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# Clean dataset (Iris has no missing values, but we’ll add handling)
df = df.dropna()


# Task 2: Basic Data Analysis

# Basic statistics
print("Basic Statistics:")
print(df.describe(), "\n")

# Grouping: average petal length per species
grouped = df.groupby("target")["petal length (cm)"].mean()
print("Average Petal Length per Species:")
print(grouped, "\n")

# Replace target numbers with species names for clarity
df["species"] = df["target"].map({i: name for i, name in enumerate(iris.target_names)})


# Task 3: Data Visualization

sns.set(style="whitegrid")  # for better style

# 1. Line chart – show cumulative mean of sepal length as a trend
df_sorted = df.sort_values("sepal length (cm)")
plt.figure(figsize=(8,5))
plt.plot(df_sorted["sepal length (cm)"].reset_index(drop=True).expanding().mean(), label="Cumulative Mean")
plt.title("Line Chart: Cumulative Mean of Sepal Length")
plt.xlabel("Sample Index (sorted)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart – average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram – distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=20, edgecolor="black")
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot – sepal length vs petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Findings / Observations

print(" Observations:")
print("- The dataset has no missing values.")
print("- Average petal length varies significantly by species (Setosa < Versicolor < Virginica).")
print("- Sepal width is normally distributed, slightly skewed left.")
print("- Strong positive correlation observed between sepal length and petal length.")
