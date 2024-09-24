import pandas as pd
import numpy as np

# File paths for the main data and new entries
FILE_PATH = "D:\\Codes\\Complete_project_macbook\\loan.csv"
NEW_ENTRY_PATH = "D:\\Codes\\Complete_project_macbook\\new_entry.csv"

def load_data():
    """
    Load data from a specified file path or the default file path if not specified.

    Returns:
        pd.DataFrame: The loaded and optimized DataFrame.
        str: The file path used for loading data.
    """
    flag = 0
    try:
        # Get the file path and flag indicating new or existing file
        file_path, flag = chose_file(flag)
    except Exception as e:
        print(f"Error during file choice: {e}")
        return None, None
    
    try:
        # Load the DataFrame from the file
        if flag == 1:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(FILE_PATH)
        
        # Strip any leading or trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Optimize the DataFrame by reducing data type sizes
        optimized_df = load_and_optimize_data(df)
        return optimized_df, file_path if flag == 1 else FILE_PATH
    except FileNotFoundError as e:
        print(f"File not found {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error: {e}")
    except Exception as e:
        print(f"Error occurred during data loading and processing: {e}")

def load_and_optimize_data(df):
    """
    Optimize the DataFrame by converting numeric columns to smaller data types where possible.

    Args:
        df (pd.DataFrame): The DataFrame to optimize.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'float64':
            # Convert to float32 if within range
            if df[col].max() <= np.finfo(np.float32).max and df[col].min() >= np.finfo(np.float32).min:
                df[col] = df[col].astype('float32')
        elif dtype == 'int64':
            # Convert to int32 if within range
            if df[col].max() <= np.iinfo(np.int32).max and df[col].min() >= np.iinfo(np.int32).min:
                df[col] = df[col].astype('int32')
        elif dtype in ['int32', 'int64', 'float32', 'float64']:
            # Further optimization of numeric columns
            if df[col].max() <= np.finfo(np.float32).max and df[col].min() >= np.finfo(np.float32).min:
                df[col] = df[col].astype('float32')
            else:
                df[col] = df[col].astype('float64')
    return df

def chose_file(flag):
    """
    Choose between a new file or an existing file and return the file path and flag.

    Args:
        flag (int): Flag indicating the type of file (0 for existing, 1 for new).

    Returns:
        tuple: (file_path, flag)
    """
    #try:
    while True:
        print("Do you want to use a new file (Y) or an existing file (N)?")
        file = input().strip().lower()
        if file == 'y':
            flag = 1
            path = input("Enter the file path: ").strip()
            return path, flag
        elif file == 'n':
            return FILE_PATH, flag
        else:
            print("Invalid Option. Please enter a valid option.")
        #else:
        #    raise ValueError("Invalid option. Please choose 'new' or 'existing'.")
            
    #except Exception as e:
        #print(f"Error occurred during file selection: {e}")
        #raise 

def data_mapping(df):
    """
    Map and return the numeric and object columns from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: (obj_cols, numeric_cols) where obj_cols is a list of object column names and numeric_cols is a list of numeric column names.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    obj_cols = df.select_dtypes(include=['object']).columns
    return obj_cols, numeric_cols

def converting_input_dtype_type(user_inputs):
    """
    Convert user input dictionary to a DataFrame and convert numeric columns to float32.

    Args:
        user_inputs (dict): Dictionary containing user inputs.

    Returns:
        pd.DataFrame: DataFrame with user inputs and numeric columns converted to float32.
    """
    try:
        # Convert dictionary to DataFrame
        user_input_df = pd.DataFrame([user_inputs])
        # Convert numeric columns to float32
        for col in user_input_df.select_dtypes(include=[np.number]).columns:
            user_input_df[col] = user_input_df[col].astype('float32')
        return user_input_df
    except Exception as e:
        # Handle any exceptions that occur during conversion
        print(f"An error occurred while converting user inputs to DataFrame: {e}")
        return pd.DataFrame()

def check_duplicate(df, new_row):
    """
    Check if the new row is a duplicate of any row in the existing DataFrame.

    Args:
        df (pd.DataFrame): The existing DataFrame.
        new_row (pd.DataFrame): DataFrame containing the new row.

    Returns:
        bool: True if the new row is a duplicate, False otherwise.
    """
    if df.empty or new_row.empty:
        print("Either dataframe is empty or user input is empty")
        return True
    
    # Combine existing DataFrame with new row
    combined_df = pd.concat([df, new_row], ignore_index=True)
    # Check for duplicates in the combined DataFrame
    if combined_df.duplicated().iloc[-1]:
        return True
    return False

def update_file(df, user_input_df, file_path):
    """
    Update the existing file with new entries and optionally save a new entry file.

    Args:
        df (pd.DataFrame): The existing DataFrame to update.
        user_input_df (pd.DataFrame): DataFrame containing new entries.
        file_path (str): Path to the file to update.
    """
    try:
        print("Updating the original file with new entries...")
        
        # Append new entries to the existing DataFrame
        update_df = pd.concat([df, user_input_df], ignore_index=True)
        
        # Save the updated DataFrame back to the original file
        update_df.to_csv(file_path, index=False)
        print("Original file updated successfully.")
        
        # Optional: Save the new entry DataFrame to a separate file
        user_input_df.to_csv(NEW_ENTRY_PATH, index=False)
        print("New entry file updated successfully.")
        
    except Exception as e:
        print(f"An error occurred while updating files: {e}")

def check_user_information(user_input_df, reference_df):
    """
    Check if user inputs meet the requirements based on the reference DataFrame.

    Args:
        user_input_df (pd.DataFrame): DataFrame containing user inputs.
        reference_df (pd.DataFrame): DataFrame used as a reference for requirements.

    Returns:
        bool: True if all requirements are met, False otherwise.
    """
    # Columns to check against the reference DataFrame
    columns_to_check = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
    
    # Dictionary to store the min and max values for each column
    requirements = {}
    
    for col in columns_to_check:
        if col in reference_df.columns:
            col_min = reference_df[col].min() if col in ['loan_amnt', 'int_rate', 'annual_inc'] else None
            col_max = reference_df[col].max() if col in ['loan_amnt', 'int_rate', 'dti'] else None
            requirements[col] = {'min': col_min, 'max': col_max}
    
    unmet_requirements = 0
    
    # Check if user inputs meet the requirements
    for col, limits in requirements.items():
        if col in user_input_df.columns:
            if 'min' in limits and limits['min'] is not None and (user_input_df[col] < limits['min']).any():
                print(f"{col} is less than the minimum required value of {limits['min']}")
                unmet_requirements += 1
            if 'max' in limits and limits['max'] is not None and (user_input_df[col] > limits['max']).any():
                print(f"{col} is more than the maximum allowed value of {limits['max']}")
                unmet_requirements += 1
    
    if unmet_requirements > 2:
        print("Not eligible")
        return False
    
    print("Eligible")
    return True

def user_input(file_path, df):
    # Ask if the customer is new or existing
    customer_type = input("Are you a new customer or an existing customer? (Enter 'new' or 'existing'): ").strip().lower()
    
    if customer_type not in ['new', 'existing']:
        print("Invalid input. Please enter 'new' or 'existing'.")
        return False, None
    
    if customer_type == 'existing':
        while True:
            customer_id = input("Please enter your id: ")
            if customer_id in df['id'].values:
                print("ID found.")
                break
            else:
                print("Invalid id. Please try again.")
                return False, None
    else:
        # Check if the id is already in use
        while True:
            customer_id = input("Please enter a new id: ")
            if customer_id in df['id'].values:
                print("ID already exists. Enter a valid id.")
            else:
                print("ID is available.")
                break

    user_inputs = {'id': customer_id}
    obj_cols, numeric_cols = data_mapping(df)
    
    print("Numeric columns: ", numeric_cols)
    
    for column in df.columns:
        if column == 'id':  # Skip id as it is already handled
            continue
        
        while True:
            value = input(f"Enter the value for {column}-> ")
            
            # Validate the input based on the column type
            if column in numeric_cols:
                try:
                    if df[column].dtype == 'int64':
                        user_inputs[column] = int(value)
                    elif df[column].dtype == 'float64':
                        user_inputs[column] = float(value)
                    else:
                        print(f"Invalid input type for {column}. Expected numeric.")
                        continue
                    break
                except ValueError:
                    print(f"Invalid input for {column}. Please enter a valid numeric value.")
            elif column in obj_cols:
                if isinstance(value, str):
                    user_inputs[column] = value
                    break
                else:
                    print(f"Invalid input type for {column}. Expected a string.")
    
    # Convert user inputs to the proper data types
    user_input_df = converting_input_dtype_type(user_inputs)
    
    print(user_input_df)
    
    if check_user_information(user_input_df, df):
        if check_duplicate(df, user_input_df):
            print("Duplicate data detected")
            return False, None
        else:
            print("New entry detected.")
            update_file(df, user_input_df, file_path)
            print(user_input_df)
        return True, user_input_df
    else:
        print("Not eligible for loan")
        return False, user_input_df


""" 
df, file_path = load_data()
#data_mapping(df)
flag, user_input_df = user_input(file_path, df)
print(user_input_df)
print(df.info()) """