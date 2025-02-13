import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Visualize data distribution
sns.pairplot(data, hue='target')  # Replace 'target' with your target column
plt.show()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()

# Example model evaluation visualization
# Assuming y_true and y_pred are available from model predictions
y_true = data['target']  # Replace with actual true labels
y_pred = data['predictions']  # Replace with actual predictions

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
report = classification_report(y_true, y_pred)
print(report)

# Instructions:
# - To visualize other features, modify the pairplot and heatmap accordingly.
# - To use actual model predictions, replace y_true and y_pred with your data.
