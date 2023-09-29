import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

orig =  pd.read_csv('total_stars.csv')
stars = orig.drop(columns=['hr', 'gl', 'bf', 'proper']) # Deletes almost empty columns
stars['Temp'] = 4600 * (1 / (0.92 * stars['ci'] + 1.7) + 1 / (0.92 * stars['ci'] + 0.62)) # Calculates Temp from B-V value
stars = stars.dropna(subset=['Temp', 'spect']) # Drops empty temp values
stars['Radius'] = np.sqrt(stars['lum'] / stars['lum'][0]) / (stars['Temp'] / stars['Temp'][0])**2 # Stars radius is solar Radii
stars['spec_class'] = stars['spect'].str[0].str.upper() # Puts the first letter of each classification into its own column

stars['color'] = (np.where(stars['spec_class'] == 'O', 'mediumblue', np.where(stars['spec_class'] == 'B',
                            'dodgerblue', np.where(stars['spec_class'] == 'A', 'paleturquoise',
                                        np.where(stars['spec_class'] == 'F', 'white', np.where(stars['spec_class'] == 'G',
                                                'yellow', np.where(stars['spec_class'] == 'K', 'orange',
                                                                np.where(stars['spec_class'] == 'M', 'red', 'NaN'))))))))

stars = stars.replace('NaN', np.nan).dropna(subset=['color']) #Drops everything that aren't in O, B, A, F, G, K, M classes
colors = stars['color'].tolist() #Makes list of colors
stars.to_csv("stars.csv") # Makes csv file of edited/cleaned data

# Plotting the HR Diagram
fig, ax = plt.subplots()

ax.scatter(stars['Temp'], stars['lum'], s = stars['Radius'] / 50, c = colors) #Had to divide by 50, otherwise the supergiants dwarf everything
ax.set_yscale('log')
ax.set_xlim([stars['Temp'].max() + 1000, stars['Temp'].min() - 1000]) # Reverses x axis
ax.set_xlabel('Temperature in Kelvin')
ax.set_ylabel('Luminosity in Solar Units')
ax.patch.set_facecolor('black') # Black background
ax.text(9000, 10000000, 'Supergiants', fontsize = 8, backgroundcolor = 'white') # Adds label
ax.text(4000, 100, 'Giants', fontsize = 8, backgroundcolor = 'white') # Adds label
ax.text(9000, 1, 'Main Sequence', fontsize = 8, backgroundcolor = 'white') # Adds label
temp_to_type = np.array([41000, 21000, 9500, 7240, 5920, 5300, 3850]) # Temps that correspond to each Spectral Class
classes = np.array(['O', 'B', 'A', 'F', 'G', 'K', 'M'])
ax2 = ax.secondary_xaxis("top")
ax2.set_xticks(temp_to_type)
ax2.set_xticklabels(classes)
ax2.set_xlabel('Spectral Type')
ax3 = ax.secondary_yaxis('right')
mags = np.array([-20, -10, -5, 0, 5, 10, 15, 20])
mag_to_lum = np.array([7870428422, 787043, 7870, 78.7, 0.787, 0.00787, 0.0000787, 0.000000787]) #Luminosities that correspond to Abs. Mags
ax3.set_yticks(mag_to_lum)
ax3.set_yticklabels(mags)
ax3.set_ylabel('Absolute Magnitude')
fig.savefig('HR_Diagram.png')

#Plotting 3D Map
fig2 = plt.figure()
ax4 = plt.axes(projection = '3d')
x = np.array(stars['x'].tolist()) # Makes list of x values
y = np.array(stars['y'].tolist()) # Makes list of y values
z = np.array(stars['z'].tolist()) # Makes list of z values

ax4.scatter3D(x, y, z, color = colors)
fig2.savefig('3D_Map.png')

#Plot of Absolute Magnitude vs. Radius
fig5, ax5 = plt.subplots()
ax5.scatter(x = stars['Radius'], y = stars['absmag'], color = colors)
ax5.set_xscale('log')
ax5.set_xlabel('Radius of the Star on a log Scale')
ax5.set_ylabel('Absolute Magnitude of the Star')
ax5.set_title('Absolute Magnitude vs. Radius')
ax5.set_ylim([20,-20]) # Flips y axis around since Mags are backwards
fig5.savefig('AbsMag_vs_Radius.png')

#Data Table of some averages, mins, and maxs
min_temp = stars.groupby('spec_class')['Temp'].min()
avg_temp = stars.groupby('spec_class')['Temp'].mean()
max_temp = stars.groupby('spec_class')['Temp'].max()

min_lum = stars.groupby('spec_class')['lum'].min()
avg_lum = stars.groupby('spec_class')['lum'].mean()
max_lum = stars.groupby('spec_class')['lum'].max()

min_rad = stars.groupby('spec_class')['Radius'].min()
avg_rad = stars.groupby('spec_class')['Radius'].mean()
max_rad = stars.groupby('spec_class')['Radius'].max()

min_mag = stars.groupby('spec_class')['absmag'].max() # Because magnitude scale is flipped so -10 mag is greater than 10 mag
avg_mag = stars.groupby('spec_class')['absmag'].mean()
max_mag = stars.groupby('spec_class')['absmag'].min()

spec_classes = classes.tolist()
temps = pd.DataFrame({'Minimum Temperature' : min_temp, 'Average Temperature' : avg_temp, 'Maximum Temperature' : max_temp}, index = spec_classes)
lumins = pd.DataFrame({'Minimum Luminosity' : min_lum, 'Average Luminosity' : avg_lum, 'Maximum Luminosity' : max_lum}, index = spec_classes)
radii = pd.DataFrame({'Minimum Radius' : min_rad, 'Average Radius' : avg_rad, 'Maximum Radius' : max_rad}, index = spec_classes)
absmags = pd.DataFrame({'Minimum Abs. Magnitude' : min_mag, 'Average Abs. Magnitude' : avg_mag, 'Maximum Abs. Magnitude' : max_mag}, index = spec_classes)

print(temps)
print(lumins)
print(radii)
print(absmags)

#Modeling and Predicting
half_stars = stars[0:57331] # Makes test set, splits original dataframe in half

X = half_stars[['Temp', 'lum', 'Radius']].values
y = half_stars['spec_class']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 42)
model = tree.DecisionTreeClassifier(max_depth = 4)
model.fit(Xtrain, ytrain)
y_predict = model.predict(Xtest)
accuracy = accuracy_score(y_predict, ytest)

temper = input('Temperature (K): ')
lumin = input('Luminosity (Solar Units): ')
radius = input('Radius (Solar Units): ')

pred_class = model.predict([[temper, lumin, radius]])
print('The predicted temperature class is: ' + str(pred_class[0]))