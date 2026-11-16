import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"  # dataset stored in the repository
TARGET_COL = "ProdTaken"

# Load the dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Remove unnecessary columns: the customer identifier and the index column saved in the CSV
df = df.drop(columns=["CustomerID"])
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Fix the "Fe Male" typo so it is treated as the same category as "Female"
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Split into training and testing sets (stratified because of class imbalance)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits locally; the workflow passes them to the training job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
print("Train/test splits saved successfully.")
