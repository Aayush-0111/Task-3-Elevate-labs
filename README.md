# Task-3-Elevate-labs
<br>
 Housing Price Regression

A simple linear regression model to predict housing prices based on various features from a housing dataset.

## Project Overview

This project demonstrates how to build a linear regression model using Python's `scikit-learn` to predict housing prices. It includes:

- Data loading and preprocessing  
- Handling categorical variables with one-hot encoding  
- Splitting data into training and testing sets  
- Training a linear regression model  
- Evaluating model performance with MAE, MSE, RMSE, and R² metrics  
- Visualizing actual vs predicted housing prices  
- Displaying model coefficients  

## Dataset

- The dataset `Housing.csv` is expected to contain various features related to houses and a target column `price`.
- Categorical features like `furnishingstatus`, `mainroad`, and `guestroom` are automatically one-hot encoded.

## Requirements

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

You can install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
