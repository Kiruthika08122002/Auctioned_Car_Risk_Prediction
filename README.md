# Auctioned Car Risk Prediction Using Machine Learning

This project aims to predict the risk level (High or Low) of auctioned cars based on historical data and car features using machine learning techniques.

## Project Description
The model uses a Random Forest Classifier trained on a dataset of car features to predict the risk based on the car's age and other factors. Feature encoding, scaling, and model evaluation have been performed to ensure reliable predictions.

## Files Included
- car_data.csv: Dataset used for training.
- model.py: Main script to train and evaluate the ML model.
- predict.py: Script to predict the risk level for new data.
- eda.py: Exploratory Data Analysis.
- preprocessing.py: Data cleaning and preparation.
- data.py, main.py: Support scripts for data loading and execution.
- labelencoders.pkl, scaler.pkl, riskmodel.pkl: Saved model and preprocessing objects.

## How to Run
1. Run model.py to train and save the model.
2. Use predict.py to get risk predictions on new car data.

## Results
The model achieved an accuracy of *100%* on the test data. Confusion matrix and classification report are included in the model.py output.

## Future Scope
- Improve generalization by testing on larger datasets.
- Deploy the model as a web app using Flask or Streamlit.
- Explore deep learning approaches for risk prediction.

## Author
Kiruthika K

## License
This project is for academic purposes under the AICTE AI-ML Virtual Internship Program.
