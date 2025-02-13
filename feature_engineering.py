import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
polynomial_features = poly.fit_transform(data)

# Convert to DataFrame
poly_df = pd.DataFrame(polynomial_features, columns=poly.get_feature_names(data.columns))

# Save the new feature engineered dataset
poly_df.to_csv('feature_engineered_data.csv', index=False)
