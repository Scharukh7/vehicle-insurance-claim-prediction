# Vehicle Insurance Claim Prediction

This repository contains an application for predicting the potential cost of vehicle insurance claims. The application uses machine learning models trained on provided data to make predictions based on user input.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Parameters](#model-parameters)
- [File Descriptions](#file-descriptions)
- [Best Model](#best-model)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository
2. Install the requirements

## Usage

1. The `model.py` file contains the machine learning models and data preprocessing steps. Run this script to train the models using the following command:

```bash
python model.py 25 M M 2 1000 F 25
``` 
Where:

- 25 is Age (should be above 18)
- M is Gender (can be F for Female)
- M is Marital Status (can be S for Single)
- 2 is the number of Dependent Children
- 1000 is the Weekly Wage
- F is Full Time Employment (can be P for Part Time)
- 25 is the number of Hours Worked Per Week

This command will train the models on the dataset, save the models, imputers, and encoders, and print out the model with the best RMSE score and its RMSE value. It will also make a prediction using each model with the input data provided and print out the predictions.

2. The app.py file contains a GUI for making predictions using the trained models. Run this script to launch the GUI:

```bash
python app.py
```

n the GUI, fill in the fields with your data and click "Predict" to get a prediction of the potential claim cost in GBP from each model.

## Model Parameters
The models used in this application are:

- Linear Regression
- K Nearest Neighbors with n_neighbors=5
- Decision Tree Regressor
- XGBoost Regressor
- LightGBM Regressor

The input features used for training the models are Age, Gender, MaritalStatus, DependentChildren, WeeklyWages, PartTimeFullTime, HoursWorkedPerWeek.

## File Descriptions
- `model.py`: This script reads in the training data, trains the machine learning models, and saves the models, imputers, and encoders contains a method for making predictions with the trained models.
- `app.py`: This script creates a GUI for making predictions with the trained models. The GUI allows users to input their data and get a prediction of the potential claim cost.

## Best Model

The model with the best RMSE score is determined when the `model.py` script is run. The script prints out the name of the model with the best RMSE score and its RMSE value.

## Contributing

Contributions are welcome. Please open an issue to discuss your ideas or open a pull request with your changes.

## License

Please see the [LICENSE](./LICENSE) file for details.



























