from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app, server, dbc
from tabs import intro, predict, explain, evaluate, model, eda2

style = {'maxWidth': '960px', 'margin': 'auto'}

app.layout = dbc.Container([
    dcc.Markdown('# Youtube Likes Predictor',
                 className="text-center font-weight-normal text-primary"),
    dcc.Tabs(id='tabs', value='tab-intro', children=[
        dcc.Tab(label='Intro', value='tab-intro'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-evaluate'),
        dcc.Tab(label='EDA-2', value='tab-eda2'),
        dcc.Tab(label='Predict', value='tab-predict'),
        dcc.Tab(label='Feature Importance', value='tab-explain'),
        dcc.Tab(label='Model Results', value='tab-model'),
    ]),
    html.Div(id='tabs-content')
],fluid=True)

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-intro': return intro.layout
    elif tab == 'tab-predict': return predict.layout
    elif tab == 'tab-explain': return explain.layout
    elif tab == 'tab-evaluate': return evaluate.layout
    elif tab == 'tab-model': return model.layout
    elif tab == 'tab-eda2':return eda2.layout



if __name__ == '__main__':
    app.run_server(debug=True)