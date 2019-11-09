
import plotly.graph_objs as go
##import plotly
#from plotly import graph_objects as go
import pandas as pd

import numpy
import seaborn as sns
from datetime import datetime
from datetime import date
import calendar
import csv
import matplotlib.pyplot as plt

def read_input_file_Deloyment():
    df = pd.read_csv('Anonymized Deployment Job Data.csv'
                     , index_col = 'Job Result'
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
                                'Customer ID'
                                ]
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
                                'Customer ID'
                                ]
                     )

    return df


def conv_time (t):
    ###Convert time to minute if NEED###
    (h, m, s ) = t.split(':')
    result = int(h) * 3600 +int(m) * 60 + int(s)


def res(input1, input2):
    ###Draw a bar graph for Duration base on job result###
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
    fig.show()


    ##print(newData)
    return 0

def time_stamp(input, target_time):

    ##input['Job Creation Time'].value_counts()

    result = 0

    for i in input['Job Creation Time']:
        if target_time == i:
            result = result + 1

    return print(result)

##time_stamp(read_input_file_Deloyment(), datetime.date(2019, 7, 7) )



res(read_input_file_Update(), read_input_file_Expansion())