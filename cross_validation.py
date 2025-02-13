import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)  # Replace 'target' with your target column
y = data['target']

# Define the model
model = RandomForestClassifier(random_state=42)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f'Cross-validation accuracy scores: {scores}')
print(f'Mean accuracy: {np.mean(scores):.2f}')
