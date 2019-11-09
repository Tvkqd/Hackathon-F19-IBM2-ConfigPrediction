

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import csv




stylesheets = ['style.css']

app = dash.Dash(__name__, external_stylesheets=stylesheets)

app.layout = html.Div(children=[
    html.H1(children="I DONT KNOW WHAT IS THIS CALL"),

    html.Div(children='''
        Group Meh...
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'EXAMPLE GRAPH NO THING SPECIAL'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)


