import pandas as pd
import numpy as np

# Load the dataset
file_path = 'new data.csv'
data = pd.read_csv(file_path)

# Function to add noise to numeric columns
def add_noise(data, noise_level=0.01):
    augmented_data = data.copy()
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            noise = np.random.normal(0, noise_level, size=data[col].shape)
            augmented_data[col] = data[col] + noise
    return augmented_data

# Separate features and target
X = data.drop('Gene_type', axis=1)
y = data['Gene_type']

# Initial class distribution
initial_class_distribution = y.value_counts()

# Manual oversampling and noise addition
augmented_data = data.copy()
iterations = 5  # Number of times to repeat the process
for _ in range(iterations):
    for class_label, count in initial_class_distribution.items():
        samples_to_add = data[data['Gene_type'] == class_label].sample(count, replace=True)
        augmented_data = pd.concat([augmented_data, add_noise(samples_to_add)])

# Ensure Gene_type contains only 0 or 1
augmented_data['Gene_type'] = augmented_data['Gene_type'].apply(lambda x: 1 if x >= 0.5 else 0)

# Save the augmented data to a CSV file
output_file_path = 'augmented_data_multiple_rounds.csv'
augmented_data.to_csv(output_file_path, index=False)

# Confirm the file has been saved
print(f"Augmented data saved to {output_file_path}")
