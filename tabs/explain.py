from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app, dbc

layout = dbc.Container([
        dcc.Markdown('### Final Results'),

        dbc.Row([
                dbc.Col([
                        html.H6("Feature Importance for predicting youtube likes (Descending Order)"),
                        html.Img(src='https://raw.githubusercontent.com/ianuvrat/datasets/main/download.png'),
                        ],className="text-center",width={'size':12, 'offset':0, 'order':1}),
                ]),
#-----------------------------------------------------------------------
html.Br(),
dbc.Row([
        dbc.Col([
                html.H6(""" Pearson Correlation Matrix """),
                html.Img(src='https://raw.githubusercontent.com/ianuvrat/datasets/main/corr_matrix.png'),
                ],className="text-center",width={'size':12, 'offset':0, 'order':1}),
        ]),
#-----------------------------------------------------------------------
html.Br(),
dbc.Row([
        dbc.Col([
                html.H6(""" Pearson Correlation Relation against Target """,
                             ),
                html.Img(src='https://raw.githubusercontent.com/ianuvrat/datasets/main/corr_bar.png',
                         )
                ],className="text-center",width={'size':12, 'offset':0, 'order':1})
        ]),
],fluid=True),





