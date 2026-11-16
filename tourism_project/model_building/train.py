import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# MLflow server started by the workflow ("Start MLflow Server" step)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")  # same file app.py loads

# Load the train/test splits produced by prep.py
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")
print("Train/test splits loaded successfully.")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]   #  All numerical feature names

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]   #  All categorical feature names

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],        # Number of boosting trees. More trees can improve performance but increase training time.
    'xgbclassifier__max_depth': [3, 4],           # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
    'xgbclassifier__colsample_bytree': [0.5, 0.7],    # Fraction of features sampled when building each tree.
    'xgbclassifier__colsample_bylevel': [0.5, 0.7],   # Fraction of features sampled at each tree level.
    'xgbclassifier__learning_rate': [0.05, 0.1],       # Step size used during boosting. Smaller values may improve generalization but require more trees.
    'xgbclassifier__reg_lambda': [0.5, 1.0],          # L2 regularization strength. Higher values help reduce overfitting.
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   #  The model pipeline by chaining preprocessor and xgb_model

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    # Experiment with different threshold values and observe their impact on
    # precision, recall, F1-score, and overall model performance.
    classification_threshold = 0.45    # Classification threshold between 0 and 1. Lower thresholds typically increase recall and decrease precision and vice versa.

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save the best model where the workflow commits it and app.py loads it
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print(f"Best model saved to {MODEL_PATH}")
