from data_loading import user_input, load_data
from data_preprocessing import preprocess_data
from model_training import train_loan_eligibility_model
from data_preprocessing import clean_and_modify_data

def main():
    # Load the data
    optimized_df, file_path = load_data()
    print(optimized_df)
    
    # Preprocess the data
    preprocessed_df, columns_list = preprocess_data(optimized_df)
    
    print(preprocessed_df)
    
    # Clean column names
    cleaned_df, columns_names = clean_column_names(preprocessed_df, columns_list)
    
    i = 0
    print("Columns containing 'bad' in their name are: ")
    for col in columns_names:
        print(f"{i} {col}")
        i += 1
    column = input("Enter the column that you want to remove during training: ")
    
    # Train the model
    model = train_loan_eligibility_model(cleaned_df, column)
    
    # Get user input
    flag, user_input_df = user_input(file_path, optimized_df)
    
    if flag:
        # Apply the same cleaning and modifying function to the user input
        user_input_df = clean_and_modify_data(user_input_df)

        # Ensure the user_input_df has the same features as the model
        missing_cols = set(model.feature_names_in_) - set(user_input_df.columns)
        for col in missing_cols:
            user_input_df[col] = 0  # Adding missing columns with default value

        user_input_df = user_input_df[model.feature_names_in_]

        # Predict with the model
        user_input_predict = model.predict(user_input_df)
        
        if user_input_predict[0] == 1:
            print("Not eligible for loan")
        else:
            print("Eligible for loan")


def clean_column_names(df, column_names):
    df.columns = [col.replace('num__', '').replace('cat__', '') for col in df.columns]
    clean_column_names = [col.replace('num__', '').replace('cat__', '') for col in column_names]
    return df, clean_column_names

if __name__ == '__main__':
    main()
