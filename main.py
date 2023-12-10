import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your dataset
positive = pd.read_csv('/path/to/your/positive/data.csv')
negative = pd.read_csv('/path/to/your/negative/data.csv')

print(positive.shape)
print(negative.shape)

#Convert Boolean data to Numeric in your dataset
def convert_boolean_columns_to_numeric(data, columns_to_check):

    # Create a copy of the original DataFrame to avoid modifying it directly
    converted_data = data.copy()

    # Use applymap to replace True with 1 and False with 0 in the specified columns
    converted_data[columns_to_check] = converted_data[columns_to_check].applymap(lambda x: 1 if x else 0)

    return converted_data

# Convert the specified columns to numeric values, check the negative and positive data
columns_to_check = ['X', 'Y', 'C']

converted_P = convert_boolean_columns_to_numeric(positive, columns_to_check)
converted_N = convert_boolean_columns_to_numeric(negative, columns_to_check)

# Combine positive and negative datasets
from sklearn.utils import shuffle

# Set a random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Use pandas.condcat to Combine positive and negative datasets
combined_data = pd.concat([converte_P, converte_N], axis=0)

# Shuffle the combined dataset
combined_data = shuffle(combined_data, random_state=seed_value)
combined_data = combined_data.reset_index(drop=True)

print(combined_data.shape)

# Use the to_csv method to write the DataFrame to a CSV file
combined_data.to_csv("fullData.csv", index=False)

#Scaling the data before training a deep learning model is crucial to ensure that
#all input features contribute equally to the model's learning process, preventing
#certain features from dominating due to their larger scales, and promoting faster
#and more stable convergence during training.

# Min-Max Scaling

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("/path/to/your/fullData.csv")

# Separate the column names from the data
column_names = data.columns
rawdata = data.iloc[1:]  # Exclude the first row (column names)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler on the rowdata
minmax = scaler.fit_transform(rawdata)

# Choose a feature to visualize (e.g., 'X')
feature_to_plot = 'X'  # Adjust the column name according to your dataset

print(column_names.shape)
print(data.shape)
print('raw',rawdata.shape)
print('minmax', minmax.shape)

# Plot the original and scaled data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data[feature_to_plot], bins=20, color='blue', alpha=0.7, label='Original')
plt.title('Original Data')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(minmax[:, data.columns.get_loc(feature_to_plot)], bins=20, color='green', alpha=0.7, label='Scaled')
plt.title('Scaled Data (MinMax)')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Z-score Standardization (Standard Scaling)
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the rowdata
Z = scaler.fit_transform(rawdata)

print(column_names.shape)
print(data.shape)
print('raw',rawdata.shape)
print('z', Z.shape)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the histogram of the original data
ax1.hist(original_column, bins=20, color='blue', alpha=0.5)
ax1.set_title('Original Data Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Plot the histogram of the scaled data
ax2.hist(Z[:, 1], bins=20, color='green', alpha=0.5)  # Assuming you want to visualize the second column after scaling
ax2.set_title('Scaled Data Distribution')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Robust Scaling (RobustScaler)
from sklearn.preprocessing import RobustScaler

# Create a RobustScaler object
scaler = RobustScaler()

# Fit the scaler on the training data and transform both training and test data
robust_scaled = scaler.fit_transform(rawdata)

print(robust_scaled.shape)

# Convert the scaled data (numpy array) back to a DataFrame
robust_scaled_df = pd.DataFrame(robust_scaled, columns=column_names)

# Choose a feature to visualize (e.g., 'X')
feature_to_plot = 'X'  # Adjust the column name according to your dataset

# Plot the original and scaled data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data[feature_to_plot], bins=20, color='blue', alpha=0.7, label='Original')
plt.title('Original Data')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(robust_scaled_df[feature_to_plot], bins=20, color='green', alpha=0.7, label='Scaled')
plt.title('Scaled Data (RobustScaler)')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

#you can save your scaled data now
np.savetxt('robust.csv', robust_scaled, delimiter=',', newline='')
np.savetxt('minmax.csv', minmax, delimiter=',', newline='')
np.savetxt('Z.csv', Z, delimiter=',', newline='')








