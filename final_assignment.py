import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

pd.set_option("display.max_columns", 120)

# FOR COLAB
'''
from google.colab import drive
drive.mount('/content/drive')
cd "/content/drive/MyDrive/Final Assignment"
'''

import os
from pathlib import Path

current_dir = Path(os.getcwd())
client_path = current_dir / "telecom" / "Client.csv"
record_path = current_dir / "telecom" / "Record.csv"

for p in (client_path, record_path):
    print(f"{'OK ' if p.exists() else 'MISSING '} {p}")

client = pd.read_csv(client_path)
record = pd.read_csv(record_path)

print(f"Client: {client.shape}")
print(f"Record: {record.shape}")

# Merge on Customer_ID. Both tables have one row per customer, so this is a 1:1 join.
df = record.merge(client, on='Customer_ID', how='inner')
print(f"Merged: {df.shape}")
df.head()

# Data Overview
data_summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isna().sum(),
    'Missing %': (df.isnull().mean() * 100).round(1),
    'Unique Values': df.nunique()
})

print("DATASET SIZE: ", df.shape)

print("\nCOLUMNS BY TYPE: ")
print(data_summary['Data Type'].value_counts())

print("\nCATEGORICAL COLUMNS: ")
print(data_summary[data_summary['Data Type'] == 'object'])

# Count of missing values per column
missing_counts = df.isna().sum()

# Take the 20 columns with the most missing values
top_missing = missing_counts.sort_values(ascending=False).head(20)

# Plot
sns.barplot(x=top_missing.values, y=top_missing.index)
plt.xlabel('Number of missing values')
plt.title('Top 20 columns by number of missing values')
plt.show()

