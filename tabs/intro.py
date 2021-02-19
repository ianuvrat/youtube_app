from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_extensions as de

from app import app,dbc

url = "https://assets8.lottiefiles.com/packages/lf20_emod42y4.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

layout = [
        dcc.Markdown(""" ### Intro
        This web application takes user inputs and predicts number of likes a youtube video can get!!"""),
        html.Div(de.Lottie(options=options, width="25%", height="15%", url=url)),

        #html.Br(),
        # dbc.Row([
        #     dbc.Col([
        #         html.Img(src='https://raw.githubusercontent.com/ianuvrat/datasets/main/new.jpg',),
        #
        #             ],className="text-center",width={'size':12, 'offset':0, 'order':1}),
        #         ]),

        dcc.Markdown("""Created by Anuvrat Shukla""",
                     className="text-center font-weight-normal text-primary")
        ]