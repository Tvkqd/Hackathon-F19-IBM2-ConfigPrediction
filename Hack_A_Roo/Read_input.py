#################################################################################################################
#  GROUP NAME: MEH                                                                                              #
#  MEMBERS: NAME                       STUDENT ID                                                               #
#        JAYDEN TRAN                    16213471                                                                #
#                                                                                                               #
#                                                                                                               #
#   Github link: https://github.com/Tvkqd/Hackathon-F19-IBM2-ConfigPrediction/tree/master/Hack_A_Roo            #
#   EVENT: UMKC HACK-A-ROO FALL 2019                                                                            #
#   PROJECT: IBM2-CONFIG PREDICTION                                                                             #
#   PURPOSE:  - Build a program can predict various provisioning, expansion/shrink,                             #
#               update and removal request execited for these services                                          #
#       PART I;  - Build a dashboard that helps visualize metrics                                               #
#                - Use dashboard to define an interval of time and view the above defined metrics               #
#       PART II: - Use mechine learning to build a prediction system                                            #
#                - Use one or more models that can assist with inventory management and                         #
#                  improve customer experience                                                                  #
#                - The system should provide following information:                                             #
#                      + The number of customer requests for provisioning.                                      #
#                      + Expansion/shrink and deletion should expect in a given time duration                   #
#                      + flexibility of defining the duration of time as a unit of days, time, and weeks.       #
#                      + Can configuration the requested in a given time duration, then able to show            #
#                        flexibility of defining the duration of time as a unit of days, time, and weeks.       #
#                      + Able to answer the request of run time of various operations as well as success and    #
#                        failure ratio. The program should be alble to predict how succeed given and amount     #
#                        of time already spent in the operation.                                                #
#################################################################################################################


################################
# IMPORT LIBRARY               #
################################
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io
import seaborn as sns
from datetime import datetime
from datetime import date
import calendar
import csv
import matplotlib.pyplot as plt

#####################################
#Building models                    #
#####################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler

##########################################################################################################
#   READ DATA                                                                                            #
##########################################################################################################
def read_input_file_Deloyment():
    df = pd.read_csv('Anonymized Deployment Job Data.csv'
                     , parse_dates = ['Job Creation Time', 'Job Completion Times']
                     , header = 0
                     , names = ['Job Result',
                                'Job Creation Time',
                                'Job Completion Times',
                                'Product Name',
                                'Cores',
                                'Storage (GB)']
                     )
    return df

def read_input_file_Expansion():
    df = pd.read_csv('Anonymized Expansion Job Data.csv'
                     , parse_dates = ['Job Creation Time', 'Job Completion Time']
                     , header = 0
                     , names = ['Job Result',
                                'Job Creation Time',
                                'Job Completion Time',
                                'Product Name',
                                'Cores',
                                'Storage',
                                'Customer ID']
                     )
    return df

def read_input_file_Update():
    df = pd.read_csv('Anonymized Update Job Data.csv'
                     , parse_dates = ['Job Creation Time', 'Job Completion Time']
                     , header = 0
                     , names = ['Job Result',
                                'Job Creation Time',
                                'Job Completion Time',
                                'Product Name',
                                'Component Being updated',
                                'Customer ID']
                     )
    return df
##########################################################################################################
#  END  READ DATA                                                                                        #
##########################################################################################################


##########################################################################################################
#   BUILDING DATA FUTURE                                                                                 #
##########################################################################################################
def conv_time (t):
    ###Convert time to minute if NEED###
    (h, m, s ) = t.split(':')
    result = int(h) * 3600 +int(m) * 60 + int(s)





##########################################################################################################
#   DATA PREPARATION                                                                                     #
##########################################################################################################
def res(input1, input2):
    ''' ??????????????????? nothing here'''
    newData = pd.DataFrame(input1, columns=['Job Result','Product Name'])

    newData.insert(2, column= 'Duration', value = input1['Job Completion Time'] - input1['Job Creation Time'])
    newData['Duration'] = pd.to_datetime(newData['Duration']).dt.time


    s = newData[newData['Job Result'] == 'SUCCESS']
    f = newData[newData['Job Result'] == 'FAILURE']
    c = newData[newData['Job Result'] == 'CANCELLED']

    fig = go.Figure(go.Bar(x = newData['Product Name'], y = s['Duration'] , name = 'Success'))
    fig.add_trace(go.Bar(x= newData['Product Name'], y = f['Duration'],  name = 'Fail'))
    fig.add_trace(go.Bar(x= newData['Product Name'], y = c['Duration'] ,  name = 'Canceled'))

    fig.update_layout(title = 'Duration base on Job Result', barmode = 'group',
                      yaxis = { 'tickformat': '%H.%M.%S', 'categoryorder': 'category ascending'},
                      xaxis = {'title': 'Product','categoryorder':'total descending'})
    #fig.show()


    ##print(newData)
    return 0

def count_job(input):
    '''Count Number of Job Creation Time'''
    df = input['Job Creation Time']



    return df.groupby(df.dt.date).count()

def time_stamp(input, target_time):

    ##input['Job Creation Time'].value_counts()

    result = 0

    for i in input['Job Creation Time']:
        if target_time == i:
            result = result + 1

    return print(result)

def CMatrix(cm, labels = ['default']):




    df = pd.DataFrame(data = cm, index = labels, columns = labels )
    df.index.name = 'TRUE'
    df.columns.name = 'PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)





    return df



##########################################################################################################
#   MAIN FUNCTION                                                                                        #
##########################################################################################################
def main():
    ##time_stamp(read_input_file_Deloyment(), datetime.date(2019, 7, 7) )

    #res(read_input_file_Update(), read_input_file_Expansion())
    #count_job(read_input_file_Deloyment())
    scaler = RobustScaler()
    A = scaler.fit_transform(input)
    B = input['Job Creation Time']

    a_train, a_test, b_train, b_test = train_test_split(A, B, test_size=0.15, random_state=123, stratify=y)

    pred_test = np.repeat(b_train.value_counts().idxmax(), b_test.size)
    matrix = pd.DataFrame(index=['accuracy', 'precision', 'recall'],
                          columns=['NULL', 'LogisticReg', 'ClassTree', 'NaiveBayes'])

    matrix.loc['recall', 'NULL'] = recall_score(b_tested=b_test, b_true=b_test)
    matrix.loc['precision', 'NULL'] = precision_score(b_tested = b_test, b_true = b_test)
    matrix.loc['accuracy', 'NULL'] = accuracy_score(b_tested = b_test, b_true = b_test)

    cm = confusion_matrix(b_tested = b_test, b_true = b_test)

    CMatrix(cm)

    return 0


##########################################################################################################
#   MAIN PROGRAM                                                                                         #
##########################################################################################################
main()