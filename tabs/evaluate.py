import pandas as pd #(version 0.24.2)
import datetime as dt
import dash         #(version 1.0.0)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly       #(version 4.4.1)
import plotly.express as px
from app import app, dbc

#df_pandas = pd.read_csv('\data\\train.csv')

# get relative data folder
import pathlib
#---------------------------------------------------------------
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
df_pandas = pd.read_csv(DATA_PATH.joinpath("train.csv"),parse_dates=['publish_date'])

#category_x = ['category_id', 'country_code']
value_y = ['views', 'dislikes','comment_count']
#-------------------------------------------------------------------------------------
style = {'padding': '1.5em'}
layout = dbc.Container([

    html.Br(),
    dbc.Row([
        dbc.Col(
            html.H2("Bar Chart for Country and Video Category",
                        className="text-center font-weight-normal text-primary"))
                        #style={"text-align": "center", "font-size":"100%", "color":"black"}))
    ]),
#-------------------------------------------------------------------------------------
        dbc.Row([
            dbc.Col([
                html.Label(['X-axis categories to compare:'],style={'font-weight': 'bold'}),
                dcc.RadioItems(id='xaxis_raditem',
                              options=[{'label': 'Video Category', 'value': 'category_id'},
                                        {'label': 'Country', 'value': 'country_code'}],
                              value='country_code',
                              className="font-weight-normal text-primary",
                              style={"width": "50%"}),
                             ],width=8),
                ]),
#-------------------------------------------------------------------------------------
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.Label(['Y-axis categories to compare:'],style={'font-weight': 'bold'}),
                dcc.Dropdown(id='yaxis_raditem', value='views',
                             options=[{'label':y, 'value':y} for y in value_y],
                             multi=False,
                             clearable=True,
                             persistence='string',
                             persistence_type='session')
                ], width=3),
                ]),


        #html.Div([
        #    html.Br(),
        #    html.Label(['Y-axis values to compare:'], style={'font-weight': 'bold'}),
        #    dcc.RadioItems(
        #        id='yaxis_raditem',
        #        options=[
        #                 {'label': 'views', 'value': 'views'},
        #                 {'label': 'dislikes', 'value': 'dislikes'},
        #        ],
        #        value='dislikes',
        #        style={"width": "50%"}
        #    ),
        #]),


        dbc.Row(dbc.Col(
            #dcc.Graph(id='the_graph'),
            dcc.Loading(children=[dcc.Graph(id="the_graph")], color="#119DFF", type="cube", fullscreen=False,),
                     )
                 )

], fluid=True)

#-------------------------------------------------------------------------------------
@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='xaxis_raditem', component_property='value'),
     Input(component_id='yaxis_raditem', component_property='value')]
)

def update_graph(x_axis, y_axis):
    dff = df_pandas
    #print(dff.shape)
    #print(dff[[x_axis,y_axis]][:1])

    barchart=px.histogram(
            data_frame=dff,
            x=x_axis,
            y=y_axis,
            title=y_axis+': by '+x_axis,
            facet_col='country_code',
            color='country_code',
            #barmode='group',
            )

    barchart.update_layout(xaxis={'categoryorder':'total ascending'},
                           title={'xanchor':'center', 'yanchor': 'top', 'y':0.9,'x':0.5,})

    return (barchart)
