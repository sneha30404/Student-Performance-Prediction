# random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from preprocess import preprocess_data
import pandas as pd

def train_random_forest(file_path, n_estimators=100, max_depth=None, cv_folds=5):
    """
    Train and evaluate a Random Forest model with cross-validation and feature importance analysis.

    Parameters:
        file_path (str): Path to the dataset file.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree. Defaults to None (no limit).
        cv_folds (int): Number of cross-validation folds.

    Returns:
        mse (float): Mean Squared Error of the model.
        r2 (float): R-squared score of the model.
        feature_importances (pd.Series): Importance of each feature.
    """
    # Preprocess the data (loading, encoding, scaling)
    X_scaled, y = preprocess_data(file_path)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model with specified hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Cross-validation scores (R²)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
    mean_cv_r2 = cv_r2_scores.mean()

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Feature importance analysis
    feature_importances = pd.Series(model.feature_importances_, index=['Medu', 'Fedu', 'goout', 'Walc', 'failures', 'studytime', 'absences', 
                                                                    'freetime', 'health', 'Dalc', 'famrel', 'romantic', 'G1']).sort_values(ascending=False)

    # Print additional metrics
    print(f"Cross-Validated R² (mean): {mean_cv_r2:.4f}")
    print(f"Feature Importances:\n{feature_importances}")

    return mse, r2, feature_importances

if __name__ == "__main__":
    # Define the path to the dataset
    file_path = 'student-mat.csv'

    # Train Random Forest model and evaluate
    mse, r2, feature_importances = train_random_forest(file_path)
    print(f"Random Forest Model - MSE: {mse}, R2: {r2}")
