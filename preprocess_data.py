import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Define preprocessing steps
numeric_features = ['num_feature1', 'num_feature2']  # Replace with actual numeric features
categorical_features = ['cat_feature1', 'cat_feature2']  # Replace with actual categorical features

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
data_preprocessed = preprocessor.fit_transform(data)

# Save the preprocessed data
pd.DataFrame(data_preprocessed).to_csv('preprocessed_data.csv', index=False)
