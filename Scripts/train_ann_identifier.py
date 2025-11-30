import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

print("--- 1. Loading and Preparing Data ---")

# Load the data from Excel
data = pd.read_excel('ann_training_data.xlsx')

# Define features (X) and target (y)
features = ['y_k', 'y_k_1', 'y_k_2']
target = 'u_k_1'
X = data[features]
y = data[target]

print("--- 2. Scaling the Data ---")
scaler = StandardScaler()

# --- THIS IS THE FIX ---
# Fit the scaler on the NumPy array (.values), not the DataFrame (X)
# This trains the scaler WITHOUT feature names.
X_scaled = scaler.fit_transform(X.values)
# ----------------------

# Save the new scaler
scaler_filename = 'ann_identifier_scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Data scaler (trained on NumPy) saved to '{scaler_filename}'")


print("\n--- 3. Building and Training ANN Identifier ---")
ann_identifier = MLPRegressor(
    hidden_layer_sizes=(20,),     # Increased neurons slightly
    activation='relu',            # CHANGED: 'relu' is better for control tasks than 'logistic'
    solver='adam',
    max_iter=5000,                # CHANGED: Give it more time to learn
    random_state=42,
    verbose=True
)

print("Starting ANN training...")
ann_identifier.fit(X_scaled, y)
print("Training complete.")

# Save the new model
model_filename = 'ann_identifier_model.pkl'
joblib.dump(ann_identifier, model_filename)
print(f"\nTrained ANN model saved to '{model_filename}'")