import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

sigma = 5.67e-8

# Read the CSV file
orig = pd.read_csv('total_stars - total_stars.csv')

# Convert 'Luminosity', 'Radius', and 'Mass' to numeric, coerce errors to NaN
orig['Luminosity'] = pd.to_numeric(orig['Luminosity'], errors='coerce')
orig['Radius'] = pd.to_numeric(orig['Radius'], errors='coerce')
orig['Mass'] = pd.to_numeric(orig['Mass'], errors='coerce')

# Drop NaN values (rows with invalid 'Luminosity', 'Radius', or 'Mass')
orig = orig.dropna(subset=['Luminosity', 'Radius', 'Mass'])

# Deletes almost empty columns
y = orig.copy()

# Calculate Temperature from Luminosity and Radius
y['Temperature'] = ((y['Luminosity'] / (4 * np.pi * (y['Radius']**2) * sigma))**(1/4))

# Adjusting Radius based on Luminosity and Temperature
y['Radius'] = np.sqrt(y['Luminosity'] / y['Luminosity'].iloc[0]) / (y['Temperature'] / y['Temperature'].iloc[0])**2
"""
# Assuming you have a 'spect' column to get the spec_class from
y['spec_class'] = y['spect'].str[0].str.upper()

y['color'] = np.where(y['spec_class'] == 'O', 'mediumblue',
            np.where(y['spec_class'] == 'B', 'dodgerblue',
            np.where(y['spec_class'] == 'A', 'paleturquoise',
            np.where(y['spec_class'] == 'F', 'white',
            np.where(y['spec_class'] == 'G', 'yellow',
            np.where(y['spec_class'] == 'K', 'orange',
            np.where(y['spec_class'] == 'M', 'red', 'NaN'))))))))

y = y.replace('NaN', np.nan).dropna(subset=['color'])
colors = y['color'].tolist()"""

y.to_csv("stars.csv")

# Training and testing the model
half_y = y[0:40]
X = half_y[['Temperature']].values
y_target = half_y['Temp_class'].values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_target, random_state=42)
model = tree.DecisionTreeClassifier(max_depth=4)
model.fit(Xtrain, ytrain)
y_predict = model.predict(Xtest)
accuracy = accuracy_score(y_predict, ytest)

print(f'Model Accuracy: {accuracy * 100:.2f}%')

Temperature = input('Temperature (K): ')
#lumin = input('Luminosity (Solar Units): ')
#radius = input('Radius (Solar Units): ')

pred_class = model.predict([[float(Temperature)]])
print('The predicted spectral class is:', pred_class[0])
