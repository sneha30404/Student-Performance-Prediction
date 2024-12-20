# evaluation.py
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from preprocess import preprocess_data

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        mse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = {"MSE": mse, "R2": r2}

    return results

if __name__ == "__main__":
    file_path = 'student-mat.csv'
    X_scaled, y = preprocess_data(file_path)

    results = compare_models(X_scaled, y)

    for model, result in results.items():
        print(f"{model} - MSE: {result['MSE']}, R2: {result['R2']}")
