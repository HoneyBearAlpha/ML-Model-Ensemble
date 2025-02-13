# Versioning dataset and model using DVC

# Instructions:
# 1. Initialize DVC in your repository:
#    dvc init

# 2. Add your dataset to DVC:
#    dvc add your_dataset.csv

# 3. Commit the changes:
#    git add .
#    git commit -m "Add dataset to DVC"

# 4. Push the dataset to remote storage (e.g., S3, Google Drive):
#    dvc remote add -d myremote <remote-url>
#    dvc push

# 5. Train your model and save it:
#    python train_model.py
#    dvc add model.pkl

# 6. Commit the model:
#    git add .
#    git commit -m "Add model to DVC"

# 7. Push the model to remote storage:
#    dvc push

# Note: Replace 'your_dataset.csv' and 'model.pkl' with your actual dataset and model files.
