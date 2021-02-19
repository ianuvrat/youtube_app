from dash import dash
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_extensions as de

from joblib import load
from app import app, dbc

country_code = ['IN','CA','US','GB']
category_id = [24.0,25.0,22.0,10.0]


style = {'padding': '1.5em'}
alert = dbc.Alert("Prectiction using XGBoost model is better!",
                  color="danger",
                  duration=3000)
                 # dismissable=False),  # use dismissable or duration=5000 for alert to close in x milliseconds
#-----------------------------------------------------------------------
layout = dbc.Container([

    html.Br(),
    dbc.Row([
       dbc.Col([
           html.H6("Estimate the number of likes your youtube video can get, based on different parameters." ,
                   className="text-dark")
       ])
    ]),
#-----------------------------------------------------------------------
     dbc.Row([
       dbc.Col([
           html.Div(id="the_alert", children=[]),
           html.Label(['Prediction Model:'], style={'font-weight': 'bold'}),
           dcc.RadioItems(
               id='xaxis_raditem',
               options=[{'label': 'XGBoost', 'value': 'XGBoost'},
                        {'label': 'LGBM', 'value': 'LGBM'}],
               value='XGBoost',
               labelClassName="m-1 text-center",
               style={"width": "50%"}),
              ], width={'size':4, 'offset':9},)
       ]),
#-----------------------------------------------------------------------
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div(id='prediction-content'),
                ],className="text-center bg-success text-white",width={'size':3, 'offset':4, 'order':10}),
            ]),
#-----------------------------------------------------------------------
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Markdown('###### Country'),
            dcc.Dropdown(
                id='country',
                options=[{'label': country, 'value': country} for country in country_code],
                value=country_code[2]),
                ], className="text-left",width={'size':3, 'offset':0, 'order':1}, style=style),

        dbc.Col([
            dcc.Markdown('###### Video Category'),
            dcc.Dropdown(
                id='catg',
                options=[{'label': category, 'value': category} for category in category_id],
                value=category_id[2]
                        ),
                ],className="text-left", width={'size':3, 'offset':6, 'order':1}, style=style),
        ]),
#-----------------------------------------------------------------------
    dbc.Row([
        dbc.Col([
            dcc.Markdown('###### Views'),
            daq.NumericInput(
                id='view',
                min=0,
                max=143408308.0,
                value=120000,
                size=120,
                label='Views',
                labelPosition='bottom'),
                ], className="text-left",width={'size':2, 'offset':0, 'order':1}),

        dbc.Col([
            dcc.Markdown('###### Dislikes'),
            daq.NumericInput(
                id='dislike',
                min=0,
                max=217017.0,
                value=1200,
                size=120,
                label='Dislikes',
                labelPosition='bottom'),
                ], className="text-left", width={'size':2, 'offset':2, 'order':3}),

        dbc.Col([
            dcc.Markdown('###### Comment Count'),
            daq.NumericInput(
                id='comment',
                min=0,
                max=692312.0,
                value=800,
                size=120,
                label='Comments',
                labelPosition='bottom'),
            ], className="text-left", width={'size':2, 'offset':3, 'order':2}, ),
        ]),


],fluid=True)
#-----------------------------------------------------------------------
@app.callback(
    [Output('prediction-content', 'children'),
     Output("the_alert", "children")],
    [Input(component_id='xaxis_raditem', component_property='value'),
     Input('country', 'value'),
     Input('catg', 'value'),
     Input('view', 'value'),
     Input('dislike', 'value'),
     Input('comment', 'value')])
#-----------------------------------------------------------------------
def predict(xaxis_raditem ,country, catg, view, dislike, comment):
    df = pd.DataFrame(
        columns=['category_id','views','dislikes', 'comment_count', 'country_code'],
        data=[[catg,view, dislike, comment,'country']]
    )
    print(xaxis_raditem)
    if xaxis_raditem == 'XGBoost':
        pipeline = load('model/pipeline_xg.joblib')
        y_pred_log = pipeline.predict(df)
        #print(y_pred_log)
        y_pred = round(y_pred_log[0])
        results = f'Predicted Youtube Likes = {y_pred} '
        #print(results)
        return results, dash.no_update

    elif xaxis_raditem == 'LGBM':
        pipeline = load('model/pipeline_lgb.joblib')
        y_pred_log = pipeline.predict(df)
        #print(y_pred_log)
        y_pred = round(y_pred_log[0])
        results = f'Predicted Youtube Likes =  {y_pred} '
        #print(results)
        return results,alert

