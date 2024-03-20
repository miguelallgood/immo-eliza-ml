
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tabulate import tabulate

class HousePricePredictor:
    """A class for predicting house prices using a trained machine learning model.

    This class loads a trained model and a dataset, preprocesses the data, and predicts house prices based on the provided features.

    Attributes:
        model_path (str): The path to the trained model file.
        data_path (str): The path to the dataset file.
        model: The trained machine learning model loaded from the model file.
        data (DataFrame): The dataset loaded from the data file.
    
    Methods:
        preprocess_data(): Preprocesses the dataset by selecting relevant features and scaling numeric features.
        predict_price(): Predicts house prices using the preprocessed dataset and the loaded model.
    """

    def __init__(self, model_path, data_path):
        """Initialize the HousePricePredictor with model and data paths."""
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(data_path)

    def preprocess_data(self):
        """Preprocess the dataset by selecting relevant features and scaling numeric features."""
        relevant_features = ['number_rooms', 'living_area', 'garden_area', 'number_facades', 'Longitude', 'Latitude']
        self.data = self.data[relevant_features]
        numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        scaler = StandardScaler()
        self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])

    def predict_price(self):
        """Predict house prices using the preprocessed dataset and the loaded model."""
        return self.model.predict(self.data)

if __name__ == "__main__":
    model_path = 'best_model.pkl'  # Path to the trained model file
    data_path = 'data/new_apartment_data.csv'  # Path to the new apartment data CSV file
    predictor = HousePricePredictor(model_path, data_path)
    predictor.preprocess_data()
    predicted_prices_scaled = predictor.predict_price()  # Predicted prices in scaled form
    # Inverse transform the predicted prices to the original scale
    scaler = StandardScaler()
    scaler.fit(pd.read_csv('data/cleaned_apartment.csv')[['price']])  # Assuming price column is in the original CSV
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled.reshape(-1, 1))

    # Load the new apartment data with features
    new_apartment_data = pd.read_csv(data_path)

    # Create a list of lists containing the features and the predicted price
    table_data = []
    for i, price in enumerate(predicted_prices.flatten(), start=1):
        features = new_apartment_data.iloc[i-1][['number_rooms', 'living_area', 'garden_area', 'number_facades', 'Longitude', 'Latitude']].tolist()  # Get features for the current property
        table_data.append([f"Property {i}", features, f"€{price:,.0f}"])

    # Print the table
    headers = ["Property", "Features", "Predicted Price"]
    print(tabulate(table_data, headers=headers))







