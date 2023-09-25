import pandas as pd
from sklearn.impute import SimpleImputer
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models

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

# Step 2: User Interaction for Column Selection and Task Type Detection

def get_user_input(data):
    print("Available columns:")
    for col in data.columns:
        print(col)

    target_column = input("Enter the target variable: ")
    columns_to_drop = input("Enter columns to drop (comma-separated): ").split(',')

    return target_column, columns_to_drop

def get_imputation_strategy():
    categorical_strategy = input("Categorical column imputation strategy (most_frequent or additional_class): ")
    numerical_strategy = input("Numerical column imputation strategy (mean, median, or mode): ")

    return categorical_strategy, numerical_strategy

def get_task_type():
    task_type = input("Choose the task type (Enter 'regression' or 'classification'): ").strip().lower()
    while task_type not in ['regression', 'classification']:
        print("Invalid input. Please enter 'regression' or 'classification'.")
        task_type = input("Choose the task type (Enter 'regression' or 'classification'): ").strip().lower()
    return task_type

# Step 3: Apply Data Imputation and PyCaret

def apply_imputation_strategy(data, categorical_strategy, numerical_strategy):
    for col in data.columns:
        if data[col].dtype == 'object':
            if categorical_strategy == 'most_frequent':
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif categorical_strategy == 'additional_class':
                data[col].fillna('missing', inplace=True)
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
    else:
        setup_function = regression_setup
        compare_function = regression_compare_models

    setup_function(data, target=target_column)
    compare_function()

# Step 4: Main Execution Flow

if __name__ == "__main__":
    file_path = input("Enter the path of the dataset: ")
    data = load_data(file_path)

    if data is not None:
        data = automate_preprocessing(data)
        target_column, columns_to_drop = get_user_input(data)
        categorical_strategy, numerical_strategy = get_imputation_strategy()
        
        task_type = get_task_type()

        data = apply_imputation_strategy(data, categorical_strategy, numerical_strategy)
        data.drop(columns=columns_to_drop, inplace=True)

        run_pycaret(data, target_column, task_type)
