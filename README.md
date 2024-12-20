# Student Grade Prediction

This project predicts student grades (`G3`) based on various features such as parental education, study time, number of failures, and more. Machine learning models including **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest** are used for making predictions.

## Project Structure

The project is organized into separate Python files, each responsible for a specific task. The key components of the project are:

### Files

- **`preprocess.py`**: Contains functions for loading, cleaning, and scaling the dataset.
- **`log_regr.py`**: Implements the **Linear Regression** model, trains it, and evaluates its performance.
- **`knn_model.py`**: Implements the **K-Nearest Neighbors (KNN)** model, trains it, and evaluates its performance.
- **`random_forest.py`**: Implements the **Random Forest** model, trains it, and evaluates its performance.
- **`evaluation.py`**: Compares the performance of all three models and prints evaluation metrics (MSE, R²).
- **`visualize.py`**: Provides functions for visualizing the results, including correlation heatmaps and model performance comparison.

### **Project Workflow**

1. **Preprocessing**: 
    - The `preprocess.py` file handles the loading and preprocessing of the dataset. It cleans the data by selecting only relevant columns and scales the features using `StandardScaler`.
    - The columns used in the model are:
        - `Medu`, `Fedu`, `goout`, `Walc`, `failures`, `studytime`, `absences`, `G1`, `G2`, and `G3` (with `G3` as the target).

2. **Model Training**: 
    - Models are defined in separate files (`log_regr.py`, `knn_model.py`, `random_forest.py`). Each file:
        - Loads the preprocessed data from `preprocess.py`.
        - Trains a corresponding machine learning model.
        - Evaluates the model's performance using **Mean Squared Error (MSE)** and **R²**.

3. **Model Evaluation**: 
    - The `evaluation.py` file compares the performance of all three models (Linear Regression, KNN, Random Forest) and prints out their **MSE** and **R²**.

4. **Visualization**: 
    - The `visualize.py` file provides functions to visualize model performance and feature correlations:
        - A **heatmap** of feature correlations is plotted to show relationships between features and the target.
        - A **bar chart** is used to compare the performance of the models (MSE and R²).

---

## Setup and Installation

### Prerequisites

To run this project, you'll need to have Python installed along with the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required dependencies using pip. Create a virtual environment and run the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
