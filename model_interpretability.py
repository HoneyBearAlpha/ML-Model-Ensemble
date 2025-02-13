import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)  # Replace 'target' with your target column
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Explain model predictions using SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Plot SHAP values
shap.summary_plot(shap_values, X_test)

# Instructions:
# - To use a different model, replace RandomForestClassifier with your model.
# - Ensure SHAP is installed: pip install shap
