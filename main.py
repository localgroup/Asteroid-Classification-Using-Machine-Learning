import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, accuracy_score)

# Read the dataset
df = pd.read_csv("nasa.csv")

# Convert 'Close Approach Date' to datetime
df['Close Approach Date'] = pd.to_datetime(df['Close Approach Date'])

# Convert 'Hazardous' column to binary (0 and 1)
df['Hazardous'] = df['Hazardous'].astype(int)


def haz():
    """Plotting function for Hazardous vs Est Dia in KM(max)"""

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Hazardous', y='Est Dia in KM(max)', data=df, estimator=np.mean)
    plt.show()


haz()


def heat_map():
    """Plotting heatmap for correlation analysis"""

    data_columns = ['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)',
                    'Miss Dist.(kilometers)', 'Orbit Uncertainity', 'Minimum Orbit Intersection',
                    'Eccentricity', 'Inclination', 'Perihelion Distance', 'Hazardous']

    df_ast = df[data_columns]

    corr = df_ast.corr()

    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, vmax=.3, center=0, square=True, linewidths=.5,
                annot=True, cbar=True)
    plt.show()


heat_map()


def box_plot():
    """Boxplot for Hazardous vs Relative Velocity km per hr"""

    sns.boxplot(data=df, x='Hazardous', y='Relative Velocity km per hr')
    plt.show()


box_plot()


def model_train():
    """Training the model"""

    selected_columns = ['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)',
                        'Miss Dist.(kilometers)', 'Orbit Uncertainity',
                        'Minimum Orbit Intersection', 'Eccentricity', 'Inclination',
                        'Perihelion Distance']

    features = df[selected_columns]
    target = df['Hazardous']

    features_train, features_temp, target_train, target_temp = train_test_split(features, target, test_size=0.2,
                                                                                random_state=42)
    features_val, features_test, target_val, target_test = train_test_split(features_temp, target_temp, test_size=0.5,
                                                                            random_state=42)

    model = RandomForestClassifier(n_estimators=65, random_state=42)
    model.fit(features_train, target_train)

    target_val_pred = model.predict(features_val)
    print("Validation Set:")
    print(classification_report(target_val, target_val_pred))

    target_test_pred = model.predict(features_test)
    print("Test Set:")
    print(classification_report(target_test, target_test_pred))

    return target_test, target_test_pred, model


target_test, target_test_pred, model = model_train()  # Distributing the variables into the global space!


#
def confuse(target_test, target_test_pred):
    """Confusion matrix function"""

    cm = confusion_matrix(target_test, target_test_pred)
    print(cm)
    cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    cmd.plot()
    plt.show()


confuse(target_test, target_test_pred)  # No need to pass (target_test, target_test_pred)
# as they are available in the global space now! But I am keeping it here to make a point!


def precise():
    """Precision-Recall curve function"""

    precision, recall, _ = precision_recall_curve(target_test, target_test_pred)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()


precise()  # No need to pass (target_test, target_test_pred) as they are available in the global space now!


def importance(model):
    """Calculates feature importance"""

    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.xlabel('Relative Importance')
    plt.show()


importance(model)  # No need to pass model as it is available in the global space now!
# But I am keeping it here to make a point!


def test_pre_trained_model():
    """Tests the pre-trained model with new data-set"""

    df_for_test = pd.read_csv("asteroids.csv")  # Reads the test data-set.
    # df_for_test.isna().any().sum()  # Shows the nan value
    df_for_test = df_for_test.replace(np.nan, 0)

    df_for_test['is_potentially_hazardous_asteroid'] = np.where(
        (df_for_test['is_potentially_hazardous_asteroid'] == False),
        0, 1)  # Replacing the bool values with equivalent 1 and 0.

    data_for_test = ['absolute_magnitude_h', 'estimated_diameter_max_km', 'estimated_diameter_min_km',
                     'kilometers_miss_distance', 'orbit_uncertainty', 'minimum_orbit_intersection',
                     'eccentricity', 'inclination', 'perihelion_distance', ]

    test_data = df_for_test[data_for_test]

    test_data = test_data.set_axis(['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)',
                                    'Miss Dist.(kilometers)', 'Orbit Uncertainity',
                                    'Minimum Orbit Intersection', 'Eccentricity', 'Inclination',
                                    'Perihelion Distance', ], axis=1)

    predictions = model.predict(test_data)
    return df_for_test, predictions


df_for_test, predictions = test_pre_trained_model()  # No need to pass model,
# as it is available in the global space now!


def accurate():
    """Calculates accuracy for test data-set"""
    accuracy = accuracy_score(df_for_test['is_potentially_hazardous_asteroid'], predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")


accurate()


def cm_for_test_data():
    """Generates confusion matrix for test data-set"""
    cm = confusion_matrix(df_for_test['is_potentially_hazardous_asteroid'], predictions)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.show()


cm_for_test_data()
