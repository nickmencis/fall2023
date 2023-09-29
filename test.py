import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_data(total_stars):
    df = pd.read_csv('total_stars.csv')

    required_columns = ['Star_name', 'Distance', 'Mass', 'Radius', 'Luminosity']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the CSV file.")

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Drop rows with special characters
    for col in df.columns:
        df = df[df[col].apply(lambda x: isinstance(x, (int, float, np.number)))]

    return df


def calculate_temperature(df):
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    df['Temperature'] = (df['Luminosity'] / (4 * np.pi * (df['Radius'] ** 2) * sigma)) ** (1 / 4)
    return df


def assign_spectral_type(temperature):
    if temperature > 30000:
        return 'O'
    elif 10000 < temperature <= 30000:
        return 'B'
    elif 7500 < temperature <= 10000:
        return 'A'
    elif 6000 < temperature <= 7500:
        return 'F'
    elif 5200 < temperature <= 6000:
        return 'G'
    elif 3700 < temperature <= 5200:
        return 'K'
    else:
        return 'M'


def assign_luminosity_class(luminosity):
    # This is a very rough approximation
    if luminosity > 100000:
        return 'I'
    elif 10000 < luminosity <= 100000:
        return 'II'
    elif 100 < luminosity <= 10000:
        return 'III'
    elif 10 < luminosity <= 100:
        return 'IV'
    else:
        return 'V'


def classify_stars(df):
    df['Spectral_Type'] = df['Temperature'].apply(assign_spectral_type)
    df['Luminosity_Class'] = df['Luminosity'].apply(assign_luminosity_class)

    X = df[['Distance', 'Mass', 'Radius', 'Luminosity', 'Temperature']]
    y = df['Spectral_Type'] + df['Luminosity_Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))

    return model, df


def main():
    df = load_data('total_stars.csv')
    df = calculate_temperature(df)
    model, df_with_classes = classify_stars(df)

    # You can save the model, or use it for other operations.
    # df_with_classes contains the classifications.


if __name__ == "__main__":
    main()
