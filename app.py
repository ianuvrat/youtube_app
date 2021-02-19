import dash
import dash_auth
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI],

                 meta_tags=[{'name': 'viewport',
                           'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
app.config.suppress_callback_exceptions = True
server = app.server

##
auth = dash_auth.BasicAuth(
    app,
    {'admin': 'admin',
     'anuvrat.shukla@siom.in': 'admin'})
