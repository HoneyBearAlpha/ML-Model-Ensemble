import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)  # Replace 'target' with your target column
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TPOT
tpot = TPOTClassifier(generations=5, population_size=50, cv=5, random_state=42, verbosity=2)

# Fit TPOT
tpot.fit(X_train, y_train)

# Score TPOT
score = tpot.score(X_test, y_test)
print(f'TPOT accuracy: {score:.2f}')

# Export the best pipeline
tpot.export('tpot_best_pipeline.py')

# Instructions:
# - Adjust generations and population_size for longer or shorter runs.
# - Ensure TPOT is installed: pip install tpot
