import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
from datetime import datetime

if __name__ == "__main__":
    # Read data
    deployment = pd.read_csv('Deployment.csv',
                             delimiter=',', usecols=[0, 1, 3, 4], skiprows=1,
                             names=['JobResult', 'CreationTime', 'Duration', 'ProductName'],
                             parse_dates=['CreationTime'])

    expansion = pd.read_csv('Expansion.csv',
                            delimiter=',', usecols=[0, 1, 3, 4], skiprows=1,
                            names=['JobResult', 'CreationTime', 'Duration', 'ProductName'],
                            parse_dates=['CreationTime'])

    update = pd.read_csv('Update.csv',
                         delimiter=',', usecols=[0, 1, 3, 4], skiprows=1,
                         names=['JobResult', 'CreationTime', 'Duration', 'ProductName'],
                         parse_dates=['CreationTime'])

    # No null value was found
    #deployment = deployment.set_index(deployment.CreationTime)
    deployment['Count'] = deployment.apply(lambda row: 1, axis=1)

    #Create new Dataframe with only CreationTime and Count
    data = deployment.loc[:, ['Count']]
    data = data.set_index(deployment.CreationTime)

    #Number of customer each unit day, time, week
    # week = data.resample('W').sum()
    # day = data.resample('D').sum()
    #We just care about hour for now
    hour = data.resample('H').sum()

    # Assigne Count result to all CreationTime
    hour_index = hour.index
    data_index = data.index
    for i in data_index:
        for j in hour_index:
            if i.hour == j.hour and i.day == j.day and i.month == j.month:
                data.at[i, 'Count'] = hour.get_value(j, 'Count')
                break

    rollMean = data.rolling(window=12).mean()
    rollStd = data.rolling(window=12).std()
    #Visualize the mean and std to see if the stationary model should be applied
    orig = plt.plot(data, color='blue', label='Original')
    mean = plt.plot(rollMean, color='red', label='Rolling Mean')
    std = plt.plot(rollStd, color='black', label='Rolling STD')
    plt.legend(loc='best')
    plt.show()
    #From the graph, we cannot apply stationary model for this time dataset


