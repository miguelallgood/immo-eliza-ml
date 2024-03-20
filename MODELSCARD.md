**Model Card**

### Project Context
This model was developed as part of a project aimed at predicting real estate prices in Belgium using machine learning techniques. The decision to focus on apartment properties stems from the availability of a reliable dataset. While other types of properties exist, the dataset for apartment properties was the most comprehensive and reliable for training the model. 

The goal is to provide accurate price estimates for department properties based on various features such as the number of rooms, living area, garden area, number of facades, longitude, and latitude.

### Data
- **Input Dataset**: The input dataset consists of apartment data (`cleaned_apartment.csv`).
- **Target Variable**: The target variable is the price of the apartment.
- **Features**: The features used for prediction include:
  - Number of rooms
  - Living area
  - Garden area
  - Number of facades
  - Longitude
  - Latitude

### Model Details
Two models were tested during the development process, including Linear Regression and Random Forest Regression. After thorough evaluation, the Random Forest Regression model was chosen as the final model due to its superior performance in predicting property prices.

### Performance
The performance of the final model was evaluated using the following metrics:
- Mean Squared Error (MSE)
- R-squared Score (R^2)

The Random Forest Regression model achieved an MSE of 0.23643506074614673 and an R^2 score of 0.7802894918594732 on the test dataset, suggesting that the model is making relatively accurate predictions.

### Limitations
- The model's predictions may be affected by factors not captured in the dataset, such as economic conditions, apartment market trends, and property-specific attributes.
- The accuracy of the predictions may vary for properties with unique characteristics not well-represented in the training data.

### Usage
Dependencies:
- pandas
- scikit-learn
- joblib

Scripts:
- `train.py`: Script for training the machine learning model.
- `predict.py`: Script for making predictions using the trained model.

To train the model:
1. Ensure that the input dataset (`cleaned_apartment.csv`) is available.
2. Run the `train.py` script.

To generate predictions:
1. Provide a CSV file containing data for the new properties.
2. Run the `predict.py` script.

### Future Enhancements
In future versions, the predictor variables Longitude and Latitude will be automatically generated or built with location-aware software for user convenience.
It is planned to create a model that takes into account all property types in future releases.

### Maintainers
For questions or issues, please contact:
- [Miguel Bueno](mailto:bueno.reyes.miguel@gmail.com)

---
This model card serves as a summary of the model's purpose, performance, and limitations. It can be used to understand its characteristics and make informed decisions about its usage.
