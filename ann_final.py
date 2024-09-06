import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam,Adamax,Nadam,SGD,Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load the data
data = pd.read_csv('augmented_data_multiple_rounds.csv')

# Separate features and target variable
X = data.drop(['Gene_type'], axis=1)
y = data['Gene_type'].astype(int)  # Ensure y is binary (0 or 1)
gene_ids = X.pop('gene_id')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Split the dataset into training and test sets, including gene_ids
X_train, X_test, y_train, y_test, gene_ids_train, gene_ids_test = train_test_split(
    X_scaled, y, gene_ids, test_size=0.2, random_state=42, stratify=y)

# Define the input dimension
input_dim = X_train.shape[1]

# Build the ANN model
model = Sequential([
    Input(shape=(input_dim,)),  # Explicit input layer defining input shape
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
opti = Adam()
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=150, batch_size=32, verbose=1, callbacks=[early_stopping])

# Evaluate the model on the training data
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {accuracy:.2f}")

# Save the predictions and results to a CSV file
all_prob_predictions = model.predict(X_train).flatten()
full_results_df = pd.DataFrame({
    'Gene_ID': gene_ids_train,
    'Actual_Label': y_train,
    'Predicted_Probability': all_prob_predictions
})
full_results_df.to_csv('full_dataset_predictions.csv', index=False)
print("Predicted probabilities saved to 'full_dataset_predictions.csv'.")

# Save the trained ANN model
model.save('ann_model1.keras')
