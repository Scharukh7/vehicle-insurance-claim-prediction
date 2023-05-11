import tkinter as tk
import pickle
from tkinter import messagebox
from model import PredictionModel  # Import the model class from your file

class InsuranceClaimGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Initialize the model
        self.model = PredictionModel('C:/Users/xkens/Desktop/Vehicle-Insurance-Claim-Prediction-main/train_SJC.csv')

        # Load the models, imputers and encoders
        self.model.load_models()
        with open('C:/Users/xkens/Desktop/Vehicle-Insurance-Claim-Prediction-main/imputers.pkl', 'rb') as f:
            self.model.imputers = pickle.load(f)

        with open('C:/Users/xkens/Desktop/Vehicle-Insurance-Claim-Prediction-main/encoders.pkl', 'rb') as f:
            self.model.encoders = pickle.load(f)


        # Create the input fields
        self.fields = ['Age', 'Gender', 'MaritalStatus', 'DependentChildren', 'WeeklyWages', 'PartTimeFullTime', 'HoursWorkedPerWeek']
        self.entries = {}

        for field in self.fields:
            row = tk.Frame(self)
            lab = tk.Label(row, width=20, text=field+": ", anchor='w')
            ent = tk.Entry(row)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            lab.pack(side=tk.LEFT)
            ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries[field] = ent

        # Create the buttons
        b1 = tk.Button(self, text='Predict', command=self.predict)
        b1.pack(side=tk.LEFT, padx=5, pady=5)

        b2 = tk.Button(self, text='Quit', command=self.quit)
        b2.pack(side=tk.LEFT, padx=5, pady=5)

    def predict(self):
        # Get the input data from the GUI
        input_data = {field: [self.entries[field].get()] for field in self.fields}

        # Make a prediction and show it in a message box
        try:
            predictions = self.model.predict(input_data)
            for model_name, prediction in predictions.items():
                messagebox.showinfo("Prediction", f"Prediction from {model_name}: {prediction[0]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    InsuranceClaimGUI().mainloop()
