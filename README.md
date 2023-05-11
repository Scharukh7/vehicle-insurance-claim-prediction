# Vehicle Insurance Claim Prediction

This repository contains an application for predicting vehicle insurance claims. The application uses machine learning models trained on provided data to make predictions based on user input.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository
2. Install the requirements


## Usage

1. The `model.py` file contains the machine learning models and data preprocessing steps. Run this script to train the models:

2. The `app.py` file contains a GUI for making predictions using the trained models. Run this script to launch the GUI:


3. In the GUI, fill in the fields with your data and click "Predict" to get a prediction from each model.

## Model Parameters

The models used in this application are:

- Linear Regression
- K Nearest Neighbors with `n_neighbors=5`
- Decision Tree Regressor

The input features used for training the models are `Age`, `Gender`, `MaritalStatus`, `DependentChildren`, `WeeklyWages`, `PartTimeFullTime`, `HoursWorkedPerWeek`.

## File Descriptions

- `model.py`: This script contains the machine learning models and data preprocessing steps. It trains the models and saves them for later use.

- `app.py`: This script contains a GUI for making predictions using the trained models. It loads the models and uses them to make predictions based on user input.

- `train_SJC.csv`: This CSV file contains the training data for the models.

- `*.pkl`: These files are created by `model.py` and contain the trained models, imputers, and encoders.

## Best Model

After training, the models are evaluated on their Root Mean Square Error (RMSE). The model with the lowest RMSE is considered the best model. The RMSE scores of the models can be found in the `rmse_scores.pkl` file.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started.































