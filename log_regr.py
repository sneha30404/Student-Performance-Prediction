# lin_regr.py
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def train_linear_regression(X, y):
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate Linear Regression
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Linear Regression:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R2: {r2:.2f}\n")

    # Ridge Regression
    ridge_model = Ridge(alpha=30.0)  # You can tune the alpha parameter
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    # Evaluate Ridge Regression
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    print("Ridge Regression:")
    print(f"  MSE: {ridge_mse:.2f}")
    print(f"  R2: {ridge_r2:.2f}")
    
    return mse, r2, ridge_mse, ridge_r2

if __name__ == "__main__":
    # Load preprocessed data
    file_path = 'student-mat.csv'
    X_scaled, y = preprocess_data(file_path)

    # Train Linear Regression model and evaluate
    mse, r2, ridge_mse, ridge_r2 = train_linear_regression(X_scaled, y)
    print(f"Linear Regression Model - MSE: {mse}, R2: {r2}")
    print(f"Linear Regression Model with Ridge Regularisation - MSE: {ridge_mse}, R2: {ridge_r2}")
