

import dash
import dash_core_components as dcc
import dash_html_components as html
import os
from dash.dependencies import Input, Output
import Read_input as grph
import plotly.graph_objs as go
import pandas as pd
import csv



stylesheets = ['style.css']
py_read_input = __import__('Read_input')


mainP = dash.Dash(__name__, external_stylesheets=stylesheets)

mainP.layout = html.Div(children=[
    html.H1(children="I DONT KNOW WHAT IS THIS CALL"),

    html.Div(children='''
    
    
        Group Meh...
        
        
    '''),

    dcc.Input(id = 'input', value = '', type = 'text'),

    ##html.Div(grph.res(grph.read_input_file_Expansion(), grph.read_input_file_Update())),
    html.Div(dcc.Input(id='input-box', type='text')),
    html.Button('Submit', id='button'),
    html.Div(id='container-button-basic',
             children='Enter a value and press submit'),

    html.Div([html.H1("Customer Requests for Provisioning", style={'textAlign': 'center'}),
    dcc.Dropdown(id='my-dropdown',options=[{'label': 'Product 1', 'value': 'Product 1'},{'label': 'Product 2', 'value': 'Product 2'},
                                           {'label': 'Product 3', 'value': 'Product 3'}],
        multi=True,value=['Product 1'],style={"display": "block","margin-left": "auto","margin-right": "auto","width": "60%"}),
    dcc.Graph(id='my-graph')
    ], className="container")
])

@mainP.callback(Output('my-graph', 'figure'),
              [Input('my-dropdown', 'value')])

def update_graph(selected_dropdown_value):
    dropdown = {"Product 1": "Product 1","Product 2": "Product 2","Product 3": "Product 3",}
    trace1 = []
    trace2 = []
    trace3 = []
    df = grph.read_input_file_Update()
    #df['Job Creation Time'] = pd.to_datetime(df['Job Creation Time'], infer_datetime_format=True).dt.date
    df['Job Creation Time'].groupby(df['Job Creation Time'].dt.date).count()

    print(len(df))
    for product in selected_dropdown_value:
        #print(df[df["Product Name"] == 'SUCCESS']["Job Result"])
        #print(df[df["Product Name"] == product])

        trace1 = [dict(
            x=df[df["Product Name"] == product]["Job Creation Time"],
            y=df[df["Job Result"] == 'SUCCESS']["Job Result"],
            autobinx=True,
            autobiny=True,
            marker=dict(color='rgb(255, 192, 203)'),
            name='Success',
            type='histogram',
            xbins=dict(
               # end =df[df["Product Name"] == product]["Job Completion Time"] ,
                size='M1',
              #  start = df[df["Product Name"] == 'SUCCESS']["Job Creation Time"],
            )
        )]
        trace2 = [dict(
            x=df[df["Product Name"] == product]["Job Creation Time"],
            y=df[df["Job Result"] == 'FAILURE']["Job Result"],
            autobinx=True,
            autobiny=True,
            marker=dict(color='rgb(0, 0, 0)'),
            name='Fail',
            type='histogram',
            xbins=dict(
                # end = '',
                size='M1',
                # start = ''
            )
        )]
        trace3 = [dict(
            x=df[df["Product Name"] == product]["Job Creation Time"],
            y=df[df["Job Result"] == 'CANCELLED']["Job Result"],
            autobinx=True,
            autobiny=True,
            marker=dict(color='rgb(152, 251, 152)'),
            name='Cancel',
            type='histogram',
            xbins=dict(
                # end = '',
                size='M1',
                # start = ''
            )
        )]


    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"{', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'})}
    return figure
if __name__ == '__main__':
    mainP.run_server(debug=True)


