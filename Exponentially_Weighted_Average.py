

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


'''
Exponentially Weighted Average (EWA) example-

The formula used to compute EWA is-
V_t = (beta * V_t-1) + ((1 - beta) * theta_t)

Data source-
https://www.kaggle.com/sudalairajkumar/daily-temperature-of-major-cities
'''


# Read in data-
data = pd.read_csv("city_temperature.csv")

data.shape
# (2906327, 8)

data.columns.tolist()
''' 
['Region',
 'Country',
 'State',
 'City',
 'Month',
 'Day',
 'Year',
 'AvgTemperature']
'''

# Check for missing values-
data.isnull().sum()
'''
Region                  0
Country                 0
State             1450990
City                    0
Month                   0
Day                     0
Year                    0
AvgTemperature          0
dtype: int64
'''

# Use the first 1000 values of average temperature attribute/feature/column-
data_1000 = data.loc[:1000, 'AvgTemperature']

data_1000.shape
# (1001,)

# Check for missing value-
data_1000.isna().any()
# False


def EWA(data, beta):
    '''
    Function to compute Exponentially Weighted Average along with bias corrected
    values.
    Returns EWA values as numpy arrays.
    '''
    V_theta = 0.0
    V = [(beta * V_theta) + ((1 - beta) * x) for x in data]
    V = np.asarray(V)

    V_bias_correction = [((beta * V_theta) + ((1 - beta) * x)) / (1 - beta ** (t + 1)) for t, x in enumerate(data_1000)]
    V_bias_correction = np.asarray(V_bias_correction)

    return V, V_bias_correction


# Get EWA values for different beta-
data_ewa_098, data_ewa_098_bias_correction = EWA(data_1000, 0.98)
data_ewa_05, data_ewa_05_bias_correction = EWA(data_1000, 0.5)
data_ewa_09, data_ewa_09_bias_correction = EWA(data_1000, 0.9)




# Visualize plots-
plt.plot(np.asarray(data_1000), label = 'actual data')
plt.plot(data_ewa_09, label = 'EWA, beta = 0.9')
plt.plot(data_ewa_09_bias_correction, label = 'EWA bias correction, beta = 0.9')
plt.plot(data_ewa_098, label = 'EWA, beta = 0.98')
plt.plot(data_ewa_098_bias_correction, label = 'EWA bias correction, beta = 0.98')
plt.plot(data_ewa_05, label = 'EWA, beta = 0.5')
plt.plot(data_ewa_05_bias_correction, label = 'EWA bias correction, beta = 0.5')
plt.legend(loc = 'best')

plt.title("Exponentially Weighted Average - Visualizations")
plt.xlabel("days")
plt.ylabel("temperature")
plt.show()

