import pandas as pd     #(version 1.0.0)
import plotly           #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px

import dash             #(version 1.9.1) pip install dash==1.9.1
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, dbc

#app = dash.Dash(__name__)

import pathlib
#---------------------------------------------------------------
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
dff = pd.read_csv(DATA_PATH.joinpath("train.csv"),parse_dates=['publish_date'])
#---------------------------------------------------------------
#dff = pd.read_csv('\data\\train.csv',parse_dates=['publish_date'])
#---------------------------------------------------------------
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
            id='datatable_id',
            data=dff.to_dict('records'),
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
                if i == "video_id" or i == "publish_date" or i == "title" or i == "channel_title" or i == "description" or i == "tags"
                else {"name": i, "id": i, "deletable": True, "selectable": True}
                for i in dff.columns
            ],
            editable=False,  # allow editing of data inside all cells
            filter_action="native",  # allow filtering of data by user ('native') or not ('none')
            sort_action="native",  # enables data to be sorted per-column by user or not ('none')
            sort_mode="multi",  # sort across 'multi' or 'single' columns
            column_selectable="multi",  # allow users to select 'multi' or 'single' columns
            row_selectable="multi",  # allow users to select 'multi' or 'single' rows
            row_deletable=False,  # choose if user can delete a row (True) or not (False)
            selected_columns=[],  # ids of columns that user selects
            selected_rows=[],  # indices of rows that user selects
            page_action='native',
            style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
            fixed_rows={'headers': True, 'data': 0},
            virtualization=False,

            style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['country', 'iso_alpha3']
            ],
            style_data={  # overflow cells' content into multiple lines
                'whiteSpace': 'normal',
                'height': 'auto'
            },
        )
        ]) ,
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='linedropdown',
                options=[
                         {'label': 'Views', 'value': 'views'},
                         {'label': 'Dislikes', 'value': 'dislikes'},
                         {'label': 'Comment Count', 'value': 'comment_count'},
                         {'label': 'Likes', 'value': 'likes'},
                ],
                value='likes',
                multi=False,
                clearable=False
            ),
        ],className="text-left",width={'size':2, 'offset':1, 'order':1}),

        dbc.Col([
        dcc.Dropdown(id='piedropdown',
            options=[
                     {'label': 'views', 'value': 'views'},
                     {'label': 'dislikes', 'value': 'dislikes'},
                     {'label': 'comment_count', 'value': 'comment_count'},
                     {'label': 'likes', 'value': 'likes'},
            ],
            value='likes',
            multi=False,
            clearable=False
        ),
        ],className="text-left",width={'size':3, 'offset':4, 'order':2}),

    ]),

    # dbc.Row([
    #     dbc.Col([
    #         dcc.Graph(id='linechart'),
    #     ],className="text-left",width={'size':6, 'offset':0, 'order':1}),

    dbc.Row([
        dbc.Col([
            dcc.Loading(children=[dcc.Graph(id="linechart")], color="#119DFF", type="cube", fullscreen=False)
                     ],className="text-left",width={'size':6, 'offset':0, 'order':1}),

            dbc.Col([
            dcc.Loading(children=[dcc.Graph(id="piechart")], color="#119DFF", type="cube", fullscreen=False)
                    ],className="text-left",width={'size':6, 'offset':0, 'order':1}),
            ]),



],fluid=True)

#------------------------------------------------------------------
@app.callback(
    [Output('piechart', 'figure'),
     Output('linechart', 'figure')],
    [Input('datatable_id', 'selected_rows'),
     Input('piedropdown', 'value'),
     Input('linedropdown', 'value')]
)


#------------------------------------------------------------------
def update_data(chosen_rows,piedropval,linedropval):
    if len(chosen_rows)==0:
        df_filterd = dff[dff['country_code'].isin(['CA','IN','GB','US'])]

    else:
        print(chosen_rows)
        df_filterd = dff[dff.index.isin(chosen_rows)]

    pie_chart=px.pie(
            data_frame=df_filterd,
            names='country_code',
            values=piedropval,
            hole=.3,
            #labels={'countriesAndTerritories':'Countries'}
            )


    #extract list of chosen countries
    list_chosen_countries=df_filterd['country_code'].tolist()
    #print(list_chosen_countries)
    #filter original df according to chosen countries
    #because original df has all the complete dates
    df_line = dff[dff['country_code'].isin(list_chosen_countries)]
    df_line = df_line.groupby(['publish_date', 'country_code'], as_index=False)[linedropval].mean()
    df_line = df_line.set_index('publish_date')
    df_line = df_line.groupby([pd.Grouper(freq="M"), 'country_code'])[linedropval].mean().reset_index()
   # print(df_line)
    a = df_line[df_line['country_code'].isin(list_chosen_countries)]
    print(a)

    line_chart = px.line(
            data_frame=a,
            x='publish_date',
            y=linedropval,
            color='country_code',
            #labels={'country_code':'Countries', 'publish_date':'date'},
            )
    line_chart.update_layout(uirevision='foo')

    return (pie_chart,line_chart)

#------------------------------------------------------------------

# if __name__ == '__main__':
#     app.run_server(debug=True)