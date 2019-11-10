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

    #stationary_check(data)
    #From the graph, we cannot apply stationary model for this time dataset
    #Scale data
    data_scaled = np.log(data)

    #Splitting train, test
    x = data_scaled['Count']
    x_train = x[117:]
    x_test = x[:116]
    indexs = x_test.index
    str_indexs = [i.strftime('%y-%m-%d %H:%M:%S') for i in indexs]

    # #Create model
    # ######################################
    # #Finding pqd
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
    # ########################################

    model = ARIMA(x_train, order=(6,0,2))
    arima = model.fit()
    print(arima.summary())
    plt.plot(x_train)
    plt.plot(arima.fittedvalues, color='red')
    plt.show()

    #Plotting the precdict and actual for test data
    fc, se, conf = arima.forecast(116, alpha=0.1)  # 90% conf
    fc_series = pd.Series(fc, index=x_test.index)
    lower_series = pd.Series(conf[:, 0], index=x_test.index)
    upper_series = pd.Series(conf[:, 1], index=x_test.index)
    # Plot
    plt.figure(1)
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(x_train, label='training')
    plt.plot(x_test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show(block= False)

    #Plotting future prediction
    arima.plot_predict(400,592)
    plt.figure(2)
    plt.show(block= False)

    #Drawing the future
    smodel = pm.auto_arima(data, start_p=1, start_q=1,
                           test='adf',
                           max_p=6, max_q=2, m=30,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    print(smodel.summary())

    # Forecast
    n_periods = 30
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

    # #Prediction
    # output = arima.forecast(steps=10)[0]
    # print(output)
    #
    # pred = []
    # for i in range(len(x_test)):
    #     #THIS IS AN INTERVAL, NEED TO DO BE PLOTTED
    #     output = arima.forecast(steps=10)
    #     print(output)
    #     yhat = output[0]
    #     pred.append(yhat)
    # error = mean_squared_error(x_test, output)
    # print('Test MSE: %.3f' % error)

    # pred = pd.DataFrame(pred)
    # pred['CreationTime'] = str_indexs
    # pred['CreationTime'] = pd.to_datetime(pred['CreationTime'])
    # pred = pred.set_index(pred.CreationTime)
    # pred = pred.drop(['CreationTime'], axis=1)
    # # NEED TO FIX THE PREDICTION

