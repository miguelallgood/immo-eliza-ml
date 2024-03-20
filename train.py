
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class HousePricePredictor:
    """
    A class for predicting real state prices based on relevant features.

    Attributes:
    - data_path (str): Path to the CSV file containing the dataset.
    - data (DataFrame): DataFrame containing the dataset.
    - best_params (dict): Best hyperparameters found by GridSearchCV.
    - best_model (estimator): Best estimator from GridSearchCV.
    - X_test (DataFrame): Testing features.
    - y_test (Series): Testing target.
    - mse (float): Mean Squared Error of the model.
    - r2 (float): R-squared score of the model.

    Methods:
    - preprocess_data(): Preprocesses the dataset by selecting relevant features,
                        handling missing values, and scaling numerical features.
    - train_model(): Trains a RandomForestRegressor model using GridSearchCV to find
                     the best hyperparameters and saves the best model.
    - evaluate_model(): Evaluates the trained model using the testing data and calculates
                        Mean Squared Error (MSE) and R-squared score (R2).
    """
    def __init__(self, data_path):
        """
        Initialize the HousePricePredictor instance with the dataset path.

        Args:
        - data_path (str): Path to the CSV file containing the dataset.
        """
        self.data = pd.read_csv(data_path)

    def preprocess_data(self):
        """
        Preprocesses the dataset by selecting relevant features,
        handling missing values, and scaling numerical features.
        """
        relevant_features = ['number_rooms', 'living_area', 'garden_area', 'number_facades', 'Longitude', 'Latitude', 'price']
        self.data = self.data[relevant_features]
        self.data.dropna(subset=['Longitude', 'Latitude'], inplace=True)
        numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        scaler = StandardScaler()
        self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])

    def train_model(self):
        """
        Trains a RandomForestRegressor model using GridSearchCV to find
        the best hyperparameters and saves the best model.
        """
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        self.X_test = X_test
        self.y_test = y_test
        # Save the trained model
        joblib.dump(self.best_model, 'best_model.pkl')

    def evaluate_model(self):
        """
        Evaluates the trained model using the testing data and calculates
        Mean Squared Error (MSE) and R-squared score (R2).
        """
        y_pred = self.best_model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)

if __name__ == "__main__":
    # Example usage
    predictor = HousePricePredictor(r'C:\Users\migue\OneDrive\BeCode\GNT-Arai-6\projects\05-immoeliza-ml\data\cleaned_apartment.csv')
    predictor.preprocess_data()
    predictor.train_model()
    predictor.evaluate_model()
    print("Best Parameters:", predictor.best_params)
    print("Test Mean Squared Error:", predictor.mse)
    print("Test R^2 Score:", predictor.r2)
