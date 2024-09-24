import pandas as pd
import numpy as np
import os

FRAUD_FILE = "D:\Codes\Complete_project_macbook\fraud_entry.csv"
LOAN_FILE = "D:\Codes\Complete_project_macbook\loan.csv"

import pandas as pd
import os

def create_initial_columns():
    # Create a DataFrame with the required columns
    columns = ['id', 'is_fraud', 'may_be_fraud', 'reason']
    data = {'id': [], 'is_fraud': [], 'may_be_fraud': [], 'reason': []}
    fraud_df = pd.DataFrame(data, columns=columns)
    return fraud_df

def generate_new_id(last_id=None):
    # Function to generate a new ID based on the given format
    if last_id is None:
        return "LEFDA0001"
    
    letter = last_id[4]
    number = int(last_id[5:])
    
    if number < 9999:
        new_number = number + 1
        new_id = f"LEFD{letter}{new_number:04d}"
    else:
        new_letter = chr(ord(letter) + 1)
        new_id = f"LEFD{new_letter}0001"
    
    return new_id

def check_and_create_fraud_file(fraud_file):
    if not os.path.exists(fraud_file):
        # If the file does not exist, create it with the required columns
        fraud_df = create_initial_columns()
        fraud_df.to_csv(fraud_file, index=False)
        print(f"File created: {fraud_file} with initial columns.")
    else:
        # If the file exists, load it and check for missing columns
        fraud_df = pd.read_csv(fraud_file)
        required_columns = ['id', 'is_fraud', 'may_be_fraud', 'reason']
        
        # Add missing columns if necessary
        for col in required_columns:
            if col not in fraud_df.columns:
                if col == 'reason':
                    fraud_df[col] = ''
                else:
                    fraud_df[col] = 0  # Set 0 for int or float type columns
        
        # Save the updated fraud_df
        fraud_df.to_csv(fraud_file, index=False)
        print(f"File checked and updated: {fraud_file} with missing columns added.")
    
    return fraud_df

def determine_criteria_columns(df, num_cols=5):
    # Select numerical columns only
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Print numerical columns
    print("Numerical columns available for criteria: ", numerical_cols.tolist())
    
    # If fewer numerical columns than num_cols, use all of them
    if len(numerical_cols) <= num_cols:
        return numerical_cols
    
    # Calculate variance for each numerical column
    variances = df[numerical_cols].var()
    
    selected_cols = variances.nlargest(num_cols).index
    # Print variance of each column
    print("\nVariance columns based on highest variance:\n", selected_cols.tolist())
    return selected_cols

def calculate_criteria(df, columns):
    min_values = df[columns].min()
    print("\nMinimum values for selected columns:\n", min_values)
    
    # Create criteria based on whether each value exceeds the minimum values
    criteria = df[columns].gt(min_values).any(axis=1)
    return criteria 

def convert_object_columns_to_float(df):
    conversion_map = {}
    
    # Iterate over object-type columns excluding 'id'
    for col in df.select_dtypes(include=['object']).columns:
        if col == 'id':
            continue
        
        # Generate a mapping from original strings to floats
        unique_values = df[col].unique()
        mapping = {value: i for i, value in enumerate(unique_values)}
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        # Save the mappings
        conversion_map[col] = {'to_float': mapping, 'to_original': reverse_mapping}
        
        # Replace the strings in the column with corresponding floats
        df[col] = df[col].map(mapping).astype(np.float32)
    
    return df, conversion_map

def update_fraud_file_based_on_loan(fraud_file, loan_file):
    # Read the loan file and fraud file
    loan_df = pd.read_csv(loan_file)
    fraud_df = pd.read_csv(fraud_file)
    
    # Convert object dtype column into numerical values through customm mapping
    loan_df, conversion_map = convert_object_columns_to_float(loan_df)
    
    # Determine criteria columns and calculate criteria
    criteria_columns = determine_criteria_columns(loan_df)
    criteria = calculate_criteria(loan_df, criteria_columns)
    
    # Update fraud_df based on loan_df
    fraud_df_index = fraud_df.index[:len(loan_df)]  # Ensure same length
    for i, idx in enumerate(fraud_df_index):
        if i >= len(loan_df):
            break
        
        bad_loan = loan_df.loc[i, 'bad_loan']
        meets_criteria = criteria[i]
        
        if not meets_criteria and bad_loan == 1:
            fraud_df.at[idx, 'is_fraud'] = 1
            fraud_df.at[idx, 'reason'] = 'Criteria does not meet and bad_loan is 1'
        elif meets_criteria and bad_loan == 1:
            fraud_df.at[idx, 'may_be_fraud'] = 1
            fraud_df.at[idx, 'reason'] = 'Criteria meets but bad_loan is 1'
        elif not meets_criteria and bad_loan == 0:
            fraud_df.at[idx, 'may_be_fraud'] = 1
            fraud_df.at[idx, 'reason'] = 'Criteria does not meet but bad_loan is 0'
        else:
            fraud_df.at[idx, 'reason'] = 'Criteria meets and bad_loan is 0'
    
    fraud_df.to_csv(fraud_file, index=False)
    print(f"Updated fraud file: {fraud_file} based on loan file.")

def ensure_matching_ids(loan_file, fraud_file):
    # Read the loan file and fraud file
    if os.path.exists(loan_file):
        loan_df = pd.read_csv(loan_file)
    else:
        raise FileNotFoundError(f"{loan_file} not found.")
    
    if os.path.exists(fraud_file):
        fraud_df = pd.read_csv(fraud_file)
    else:
        fraud_df = create_initial_columns()
    
    # Check if 'id' column exists in loan_df
    if 'id' in loan_df.columns:
        # Use existing IDs from loan_df
        ids = loan_df['id'].tolist()
        
        # Ensure fraud_df has the same number of IDs
        if len(ids) > len(fraud_df):
            last_id = fraud_df['id'].iloc[-1] if not fraud_df.empty else None
            while len(ids) > len(fraud_df):
                new_id = generate_new_id(last_id)
                fraud_df = pd.concat([fraud_df, pd.DataFrame({'id': [new_id], 'is_fraud': [0], 'may_be_fraud': [0], 'reason': ['']})], ignore_index=True)
                last_id = new_id
        
        # Update fraud_df with existing IDs
        fraud_df['id'] = pd.Series(ids)
    else:
        # If 'id' does not exist, generate new IDs
        last_id = fraud_df['id'].iloc[-1] if not fraud_df.empty else None
        ids = [generate_new_id(last_id) for _ in range(len(fraud_df))]
        
        # Ensure fraud_df and loan_df have the same number of IDs
        loan_df['id'] = ids
        fraud_df['id'] = ids
    
    # Save updated fraud_df and loan_df
    loan_df.to_csv(loan_file, index=False)
    fraud_df.to_csv(fraud_file, index=False)
    
    print(f"IDs ensured to match between {loan_file} and {fraud_file}.")

# Example usage:
fraud_file = FRAUD_FILE
loan_file = LOAN_FILE

# Ensure IDs match between loan and fraud files
ensure_matching_ids(loan_file, fraud_file)

# Check and create fraud file if needed
fraud_df = check_and_create_fraud_file(fraud_file)

# Update the fraud file based on the loan file
update_fraud_file_based_on_loan(fraud_file, loan_file)
