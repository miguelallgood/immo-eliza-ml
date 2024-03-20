# Appartment Price Prediction

This project aims to predict appartment prices using machine learning techniques. It includes scripts for data preprocessing, model training, and prediction.

## Project Structure

- `data/`: Contains the dataset files.
- `train.py`: Script for training the machine learning model.
- `predict.py`: Script for making predictions using the trained model.
- `best_model.pkl`: Trained model saved in a pickle file format.

## Usage

### Data Preparation

Ensure that you have the dataset file (`cleaned_apartment.csv`) in the `data/` directory.

### Model Training

To train the machine learning model, run the following command: python train.py
This will preprocess the dataset, train the model using a Random Forest Regressor, and save the trained model to a file named `best_model.pkl`.

### Making Predictions

To make predictions using the trained model, run the following command: python predict.py
This script loads the trained model and a new dataset file (`new_apartment_data.csv`), preprocesses the data, and predicts apartment prices for the new dataset. The predicted prices are printed to the console.

## Predictions

To make predictions, you need to provide a CSV file containing data for the new properties. The CSV file should include columns for features such as the number of rooms, living area, garden area, number of facades, longitude, and latitude. After preprocessing the data and making predictions, the script prints the predicted prices to the console.

## Dependencies

- pandas
- scikit-learn
- joblib

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
