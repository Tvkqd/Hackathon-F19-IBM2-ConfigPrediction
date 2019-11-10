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

    stationary_check(data)
    #From the graph, we cannot apply stationary model for this time dataset
    #Scale data
    data_scaled = np.log(data)

    #Splitting train, test
    x = data_scaled['Count']
    x_train = x[:475]
    x_test = x[476:]
    indexs = x_test.index
    str_indexs = [i.strftime('%y-%m-%d %H:%M:%S') for i in indexs]

    #Create model
    ######################################
    #Finding pqd
    # p = d = q = range(0,5)
    # pdq = list(itertools.product(p,d,q))
    #
    # mean = 1000
    # index = 0
    # for i in pdq:
    #     try:
    #         model = ARIMA(x_train, order=i)
    #         arima = model.fit()
    #         if arima.aic < mean:
    #             mean = arima.aic
    #             index = i
    #     except:
    #         continue
    # print(index)
    ########################################
    #Create model
    model = ARIMA(x_train, order=(6,0,2))
    arima = model.fit()
    plt.plot(x_train)
    plt.plot(arima.fittedvalues, color='red')
    plt.show()

    #Prediction
    pred = []
    for i in range(len(x_test)):
        #THIS IS AN INTERVAL, NEED TO DO BE PLOTTED
        output = arima.forecast()
        print(output)
        yhat = output[0]
        pred.append(yhat)
    error = mean_squared_error(x_test, pred)
    print('Test MSE: %.3f' % error)

    pred = pd.DataFrame(pred)
    pred['CreationTime'] = str_indexs
    pred['CreationTime'] = pd.to_datetime(pred['CreationTime'])
    pred = pred.set_index(pred.CreationTime)
    pred = pred.drop(['CreationTime'], axis=1)
    # NEED TO FIX THE PREDICTION

