import pandas as pd

def load_data():
    # Load the dataset
    data = pd.read_csv("car_data.csv")

    # Function to assign risk
    def assign_risk(row):
        if row['Selling_Price'] < 2 and row['Owner'] > 0:
            return 'High'
        elif row['Selling_Price'] < 4:
            return 'Medium'
        else:
            return 'Low'

    # Create a new 'Risk' column
    data['Risk'] = data.apply(assign_risk, axis=1)

    return data