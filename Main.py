import pandas as pd
from sklearn.impute import SimpleImputer
from pycaret.classification import (
    setup as classification_setup,
    compare_models as classification_compare_models,
    tune_model as classification_tune_model,
    predict_model as classification_predict_model,
    save_model as classification_save_model,
)
from pycaret.regression import (
    setup as regression_setup,
    compare_models as regression_compare_models,
    tune_model as regression_tune_model,
    predict_model as regression_predict_model,
    save_model as regression_save_model,
)

# Step 1: Load Data and Automate Preprocessing

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)  # Modify for other file formats
        return data
    except Exception as e:
        print(e)
        return None

def automate_preprocessing(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(exclude=['object']).columns

    for col in categorical_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    for col in numerical_columns:
        data[col].fillna(data[col].median(), inplace=True)

    return data

# Step 2: User Interaction for Column Selection

def get_user_input(data):
    print("Available columns:")
    for col in data.columns:
        print(col)

    target_column = input("Enter the target variable: ")
    columns_to_drop = input("Enter columns to drop (comma-separated): ").split(',')

    return target_column, columns_to_drop

# Step 3: Apply Data Imputation, PyCaret, and Hyperparameter Tuning

def apply_imputation_strategy(data, categorical_strategy, numerical_strategy):
    for col in data.columns:
        if data[col].dtype == 'object':
            if categorical_strategy == 'most_frequent':
                data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            if numerical_strategy == 'mean':
                data[col].fillna(data[col].mean(), inplace=True)
            elif numerical_strategy == 'median':
                data[col].fillna(data[col].median(), inplace=True)
            elif numerical_strategy == 'mode':
                data[col].fillna(data[col].mode()[0], inplace=True)

    return data

def run_pycaret(data, target_column, task_type):
    if task_type == 'classification':
        setup_function = classification_setup
        compare_function = classification_compare_models
        tune_function = classification_tune_model
        predict_function = classification_predict_model
        save_function = classification_save_model
    else:
        setup_function = regression_setup
        compare_function = regression_compare_models
        tune_function = regression_tune_model
        predict_function = regression_predict_model
        save_function = regression_save_model

    setup_function(data, target=target_column)
    best_model = compare_function()

    # Tune the best model
    tuned_model = tune_function(best_model)

    print("The best model is:", best_model)

    # Save the tuned model for future use
    save_function(tuned_model, 'best_model')

    # Load the saved model when making predictions on new data
    new_data = load_data("path_to_new_data.csv")
    if new_data is not None:
        new_data = automate_preprocessing(new_data)
        predictions = predict_function(tuned_model, data=new_data)
        print("Predictions on new data:")
        print(predictions)

# Step 4: Main Execution Flow

if __name__ == "__main__":
    file_path = input("Enter the path of the dataset: ")
    data = load_data(file_path)

    if data is not None:
        data = automate_preprocessing(data)
        target_column, columns_to_drop = get_user_input(data)

        # Automatically detect the task type based on the target column's data type
        if data[target_column].dtype == 'object' or set(data[target_column]) == {0, 1}:
            task_type = 'classification'
        else:
            task_type = 'regression'

        categorical_strategy = input("Categorical column imputation strategy (most_frequent or additional_class): ")
        numerical_strategy = input("Numerical column imputation strategy (mean, median, or mode): ")

        data = apply_imputation_strategy(data, categorical_strategy, numerical_strategy)
        data.drop(columns=columns_to_drop, inplace=True)

        run_pycaret(data, target_column, task_type)
