import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from data_loading import converting_input_dtype_type, check_user_information, check_duplicate, update_file, data_mapping

def plot_feature_distribution(df, title):
    df.hist(bins=20, figsize=(20, 15))
    plt.suptitle(title, fontsize=16)
    plt.show()

def handle_missing_values(df):
    # Find missing values
    missing_data = df.isnull().sum()
    print("Missing values before imputation:\n", missing_data[missing_data > 0])
    
    plt.figure(figsize=(10, 6))
    missing_data.plot(kind='bar')
    plt.title('Missing Values Before Imputation')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.show()
    
    # Remving rows with null values
    df = df.dropna(thresh=len(df.columns) - 3, axis=0)
    
    # Fill missing values with mean for numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing values with mode for categorical columns
    categorical_columns = df.select_dtypes(include=[object]).columns
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Plot missing values after imputation
    missing_data_after = df.isnull().sum()
    missing_data_after = missing_data_after[missing_data_after > 0]
    if not missing_data_after.empty:
        plt.figure(figsize=(10, 6))
        missing_data_after.plot(kind='bar')
        plt.title('Missing Values After Imputation')
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.show()
    else:
        print("No missing values after imputation.")

def normalize_data(df):
    # Separate numerical and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=[object]).columns

    # Pipelines for preprocessing
    numeric_pipeline = Pipeline(steps=[
        ('no_scaling', 'passthrough') # No scaling applied
    ])
    """
    numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ]) """

    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='passthrough')

    # Apply transformations
    df_transformed = preprocessor.fit_transform(df)
    df_transformed = pd.DataFrame(df_transformed, columns=preprocessor.get_feature_names_out())

    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")

    return df_transformed

def clean_and_modify_data(df):
    # Remove noise and error
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Print columns in an ordered manner with numbers
    print("Current columns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    # Ask the user to enter numbers corresponding to columns they wish to remove
    columns_to_remove_indices = input("Enter the numbers corresponding to the columns to remove (comma-separated): ").split(',')
    columns_to_remove_indices = [int(index.strip()) - 1 for index in columns_to_remove_indices if index.strip().isdigit()]
    
    # Match the numbers with column names and remove them
    columns_to_remove = [df.columns[i] for i in columns_to_remove_indices if i in range(len(df.columns))]
    
    df = df.copy()
    df.drop(columns=columns_to_remove, inplace=True)
    print(f"Columns removed: {columns_to_remove}")

    return df

def clean_string_columns(df):
    for column in df.select_dtypes(include=[object]).columns:
        if df[column].str.contains(r'\d').any():
            df[column] = df[column].str.extract(r'(\d+)', expand=False).astype(float)
    return df

def check_bad_loan_column(df):
    """
    Check if the dataframe contains a column named 'bad_loan' or any column that contains 'bad' in its name.

    Parameters:
    df (pd.DataFrame): The DataFrame to check
    
    Returns:
    bool: True if the column exists, False otherwise
    list: A list of columns that match the criteria
    """
    matching_columns = [col for col in df.columns if 'bad_loan' in col or 'bad' in col]
    
    if len(matching_columns) > 0:
        return True, matching_columns
    else:
        return False, []
    
def preprocess_data(df):
    # Plot feature distribution before preprocessing
    plot_feature_distribution(df, "Feature Distribution Before Preprocessing")
    
    # Handle missing values
    handle_missing_values(df)
    
    # Clean and modify data
    df = clean_and_modify_data(df)
    
    # Clean string columns
    df = clean_string_columns(df)
    
    # Normalize data (including categorical)
    df = normalize_data(df)
    
    # Plot feature distribution after preprocessing
    plot_feature_distribution(df, "Feature Distribution After Preprocessing")
    
    flag, column_list = check_bad_loan_column(df)
    if flag == True:
        return df, column_list
    else:
        print("No columns containing 'bad_loan' or 'bad' were found in the DataFrame")
        exit()
        #return None, None
        #raise ValueError("No columns containing 'bad_loan' or 'bad' were found in the DataFrame")

"""def user_input(file_path, df):
    
    Collect user inputs, convert them to DataFrame, and check for eligibility and duplicates.

    Args:
        file_path (str): Path to the file where data will be updated.
        df (pd.DataFrame): Existing DataFrame to compare new entries against.

    Returns:
        bool: True if the input is valid and processed, False otherwise.
    
    from data_loading import converting_input_dtype_type, check_user_information, check_duplicate, update_file, data_mapping
    
    user_inputs = {}
    obj_cols, numeric_cols = data_mapping(df)
    for column in df.columns:
        if column in obj_cols:
            unique = df[column].unique()
            print(f"Possible values {unique}")
        while True:
            value = input(f"Enter the value for {column}-> ")
            if column in numeric_cols:
                try:
                    user_inputs[column] = float(value)
                    break
                except ValueError:
                    print(f"Invalid input for {column}, Please enter a numeric value.")
            elif column in obj_cols:
                user_inputs[column] = value
                break
    user_input_df = converting_input_dtype_type(user_inputs)
    print(user_input_df)
    if check_user_information(user_input_df, df):
        if check_duplicate(df, user_input_df):
            print("Duplicate data detected")
        else:
            print("New entry detected.")
            update_file(df, user_input_df, file_path)
            print(user_input_df)
            user_input_df = clean_and_modify_data(user_input_df)
        return True, user_input_df
    else:
        print("Not eligible for loan")
        return False, user_input_df """

"""
df = pd.read_csv("/Users/gauravlamba/Documents/Codes/Gaurav_project/loan.csv")
file_path = "/Users/gauravlamba/Documents/Codes/Gaurav_project/loan.csv"
# Display the DataFrame
print("Original DataFrame:")
print(df)

# Now you can use this DataFrame as input to your preprocessing functions
preprocessed_df = preprocess_data(df)
print("Preprocessed DataFrame:")
print(preprocessed_df) """