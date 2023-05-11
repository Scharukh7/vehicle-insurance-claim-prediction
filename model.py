from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import mean_squared_error
from math import sqrt

# Define a PredictionModel class which will handle data preprocessing, model training and prediction.
class PredictionModel:

    # Initialize the PredictionModel with the path to the training data file.
    def __init__(self, train_file):
        self.train_file = train_file
        self.df = pd.read_csv(self.train_file)
        self.rename_columns()  # Rename columns to better understand their purpose
        self.imputers = {}  # Dictionaries to store imputers and encoders for each column
        self.encoders = {}
        # Initialize models to be trained
        self.models = {
            "linear_regression": LinearRegression(),
            "k_nearest_neighbors": KNeighborsRegressor(n_neighbors=5),
            "decision_tree": DecisionTreeRegressor(),
            "xgboost": XGBRegressor(random_state=42),  # Added XGBoost model
            "lightgbm": LGBMRegressor(random_state=42)  # Added LightGBM model
        }
        self.predictions = {}  # Dictionary to store predictions from each model

    # Method to rename columns of the dataframe
    def rename_columns(self):
        self.df = self.df.rename(columns={"Unnamed: 0": "ClaimNumber", "Unnamed: 1": "DateTimeOfAccident",
                                          "Unnamed: 3": "Age", "Unnamed: 4": "Gender",
                                          "Unnamed: 5": "MaritalStatus", "Unnamed: 6": "DependentChildren",
                                          "Unnamed: 8": "WeeklyWages", "Unnamed: 9": "PartTimeFullTime",
                                          "Unnamed: 10": "HoursWorkedPerWeek", "Unnamed: 12": "ClaimDescription",
                                          "Unnamed: 13": "InitialIncurredCalimsCost",
                                          "Unnamed: 14": "UltimateIncurredClaimCost"})

    # Method to preprocess the data. It imputes missing numerical values with the mean and encodes categorical variables.
    def preprocess_data(self, data):
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):  # Check if the data type of the column is numeric
                # If the data is numeric, replace missing values with the mean of the column
                imputer = SimpleImputer(strategy='mean')
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
                self.imputers[col] = imputer  # Store the imputer in the dictionary
            else:
                # If the data is not numeric, encode it using label encoding
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le  # Store the encoder in the dictionary
        return data

    # Method to train the models on the preprocessed data.
    def train_models(self):
        self.df = self.preprocess_data(self.df)

        # Define the features to be used in the model
        features = ['Age', 'Gender', 'MaritalStatus', 'DependentChildren', 'WeeklyWages', 'PartTimeFullTime',
                    'HoursWorkedPerWeek']
        X = self.df[features]  # Input features
        y = self.df['UltimateIncurredClaimCost']  # Output variable

        # Split the data into train and test sets
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store RMSE scores for each model
        self.rmse_scores = {} 

        # Train each model and calculate its RMSE on the test set
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)  # Train the model
            self.models[model_name] = model  # Store the trained model
            predictions = model.predict(X_test)  # Make predictions on the test set
            rmse = sqrt(mean_squared_error(y_test, predictions))  # Calculate the RMSE
            self.rmse_scores[model_name] = rmse  # Store the RMSE

    # Method to make predictions on given input data.
    def predict(self, input_data):
        input_df = pd.DataFrame.from_dict(input_data)

        # Preprocess the input data in the same way as the training data
        for col in input_df.columns:
            if col in self.imputers:
                input_df[col] = self.imputers[col].transform(input_df[col].values.reshape(-1, 1))

            if col in self.encoders:
                input_df[col] = self.encoders[col].transform(input_df[col])

        # Make a prediction with each model and store it
        for model_name, model in self.models.items():
            self.predictions[model_name] = model.predict(input_df)
        return self.predictions
    
    # Method to save the trained models, imputers and encoders.
    def save_models(self):
        for model_name in self.models:
            with open(f'C:/Users/xkens/Documents/vehicle-insurance-claim-prediction/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(self.models[model_name], f)
        with open('imputers.pkl', 'wb') as f:
            pickle.dump(self.imputers, f)

        with open('encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)

        # Save the RMSE scores
        with open('rmse_scores.pkl', 'wb') as f:
            pickle.dump(self.rmse_scores, f)

    # Method to load the trained models
    def load_models(self):
        for model_name in ["linear_regression", "k_nearest_neighbors", "decision_tree", "xgboost", "lightgbm"]:
            with open(f'C:/Users/xkens/Documents/vehicle-insurance-claim-prediction/{model_name}_model.pkl', 'rb') as f:
                self.models[model_name] = pickle.load(f)
        return self.models

# The main function of the script
if __name__ == "__main__":
    model = PredictionModel('C:/Users/xkens/Documents/vehicle-insurance-claim-prediction/train_SJC.csv')
    model.train_models()  # Train the models
    model.save_models()  # Save the trained models, imputers and encoders
    loaded_models = model.load_models()  # Load the trained models

    # Load the imputers and encoders
    with open('C:/Users/xkens/Documents/vehicle-insurance-claim-prediction/imputers.pkl', 'rb') as f:
        loaded_imputers = pickle.load(f)

    with open('C:/Users/xkens/Documents/vehicle-insurance-claim-prediction/encoders.pkl', 'rb') as f:
        loaded_encoders = pickle.load(f)
        
    # Load the RMSE scores
    with open('C:/Users/xkens/Documents/vehicle-insurance-claim-prediction/rmse_scores.pkl', 'rb') as f:
        loaded_rmse_scores = pickle.load(f)

    # Determine the model with the best (lowest) RMSE
    best_model = min(loaded_rmse_scores, key=loaded_rmse_scores.get)
    print(f"The model with the best RMSE is: {best_model} with RMSE: {loaded_rmse_scores[best_model]}")

    # To make a prediction with the loaded model, first define the input data
    input_data = {'Age': [sys.argv[1]], 'Gender': [sys.argv[2]], 'MaritalStatus': [sys.argv[3]],
                    'DependentChildren': [sys.argv[4]], 'WeeklyWages': [sys.argv[5]], 'PartTimeFullTime': [sys.argv[6]],
                    'HoursWorkedPerWeek': [sys.argv[7]]}
    input_df = pd.DataFrame.from_dict(input_data)

    # Preprocess the input data in the same way as the training data
    for col in input_df.columns:
        if col in loaded_imputers:
            input_df[col] = loaded_imputers[col].transform(input_df[col].values.reshape(-1, 1))

        if col in loaded_encoders:
            input_df[col] = loaded_encoders[col].transform(input_df[col])

    # Make a prediction with each loaded model
    for model_name, loaded_model in loaded_models.items():
        prediction = loaded_model.predict(input_df)
        print(f"Prediction from {model_name}: {prediction}")