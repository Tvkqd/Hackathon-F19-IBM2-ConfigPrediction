import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import itertools
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")


def stationary_check(timedata):
    ''' Visualize data to see if the mean and std are stable values '''
    rollMean = timedata.rolling(window=12).mean()
    rollStd = timedata.rolling(window=12).std()

    # Visualize the mean and std to see if the stationary model should be applied
    orig = plt.plot(timedata, color='blue', label='Original')
    mean = plt.plot(rollMean, color='red', label='Mean')
    std = plt.plot(rollStd, color='black', label='STD')
    plt.legend(loc='best')
    plt.show()

    #Print the p-value
    df = adfuller(timedata['Count'], autolag='AIC')
    print()
    print('p-value:', df[1])

def get_count(file_data):

    # Create new Dataframe with only CreationTime and Count
    data = file_data.loc[:, ['Count']]
    data = data.set_index(file_data.CreationTime)

    # Number of customer each unit day, time, week
    # week = data.resample('W').sum()
    # day = data.resample('D').sum()
    # We just care about hour for now
    hour = data.resample('H').sum()

    # Assigne Count result to all CreationTime
    hour_index = hour.index
    data_index = data.index
    for i in data_index:
        for j in hour_index:
            if i.hour == j.hour and i.day == j.day and i.month == j.month:
                data.at[i, 'Count'] = hour.get_value(j, 'Count')
                break
    return data

def forecast(data):
    # Scale data
    data_scaled = np.log(data)

    # Splitting train, test
    x = data_scaled['Count']
    x_train = x[117:]
    x_test = x[:116]
    indexs = x_test.index
    str_indexs = [i.strftime('%y-%m-%d %H:%M:%S') for i in indexs]
    #Drawing the future
    smodel = pm.auto_arima(data, start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    print(smodel.summary())

    # Forecast
    n_periods = 12
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='MS')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(3)
    plt.plot(data)
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)
    plt.show()
    #Reference: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

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
    # deployment = deployment.set_index(deployment.CreationTime)
    deployment['Count'] = deployment.apply(lambda row: 1, axis=1)
    de_data = get_count(deployment)
    forecast(de_data)

    expansion['Count'] = expansion.apply(lambda row: 1, axis=1)
    ex_data = get_count(expansion)
    forecast(ex_data)

    update['Count'] = update.apply(lambda row: 1, axis=1)
    up_data = get_count(update)
    forecast(up_data)
