# AUTHORS: Nick Mencis, Sydney Andrews, Sitara Brent
# CREATED: 20 September 2023
# S-STEM
# FALL 2023
# CODING: UTF-8

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load the dataset
data = pd.read_csv('total_stars.csv')

# 2. Create a linear regression model and predict temperatures for the whole dataset
X = data[['Mass', 'Radius', 'Luminosity']]
y = data['Temperature']

model = LinearRegression()
model.fit(X, y)
data['Predicted_Temperature'] = model.predict(X)

# 3. Update the original 'total_stars.csv' file with the Predicted_Temperature
data.to_csv('total_stars.csv', index=False)

# 4. Split the data into training and testing sets using the Predicted_Temperature
X_train, X_test, y_train, y_test = train_test_split(X, data['Predicted_Temperature'], test_size=0.2, random_state=42)

# 5. Retrain the model on the training data
model.fit(X_train, y_train)

# Predict the temperatures for the test set and calculate the RMSE
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')
# 2. Classify based on the temperature
def classify_spectral_type(temp):
    if temp > 30000:
        return 'O'
    elif temp > 10000:
        return 'B'
    elif temp > 7500:
        return 'A'
    elif temp > 6000:
        return 'F'
    elif temp > 5200:
        return 'G'
    elif temp > 3700:
        return 'K'
    else:
        return 'M'

# You can expand this for Luminosity Classification
def classify_luminosity(temp):
    # Dummy classification based on temperature, adjust as necessary
    if temp > 15000:
        return 'I'
    else:
        return 'V'

data['Spectral_Type'] = data['Predicted_Temperature'].apply(classify_spectral_type)
data['Luminosity_Classification'] = data['Predicted_Temperature'].apply(classify_luminosity)

# 3. Write to CSV
data.to_csv('ml_stars.csv', index=False)
