# Loan Eligibility and Approval Prediction System

## Overview
This project is designed to predict the eligibility and approval status of loan applicants using machine learning models. It leverages a variety of key factors such as loan amount, interest rates, employment history, credit score, and other financial indicators. The system aids financial institutions in making accurate, data-driven decisions quickly and efficiently.

## Features
- Predicts loan eligibility based on applicant information
- Streamlines the loan approval process for financial institutions
- Uses machine learning algorithms for accurate prediction
- Provides detailed insights through data visualization

## Project Workflow
1. **Data Collection**: Gather historical loan data, including applicant details and loan outcomes.
2. **Data Preprocessing**: Clean and prepare the data for model training by handling missing values, encoding categorical features, and scaling numerical values.
3. **Model Training**: Train a RandomForestClassifier to predict the loan eligibility and approval status.
4. **Prediction**: Make predictions for new loan applications.
5. **Evaluation**: Measure model performance using accuracy, precision, recall, and other metrics.
6. - **New Data Entry and Fraud Detection**:
  - New Data Entry: The system creates a new_entry.csv file for each new data submission, generating a unique ID for every user. If a user ID already exists, the system reuses it to prevent duplication.
  - Fraud Detection: A separate fraud_entry.csv file is created to store data flagged as fraudulent. The system compares new entries against the dataset to detect inconsistencies and suspicious data.
  - Fraudulent Data Requirements: The system automatically determines the criteria for identifying fraudulent data by evaluating entries against pre-defined rules.

## Tech Stack
- **Programming Languages**: Python
- **Libraries**:
  - `numpy` and `pandas` for data manipulation
  - `scikit-learn` for machine learning
  - `matplotlib` and `seaborn` for visualization
- **Model**: RandomForestClassifier from `scikit-learn`

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
