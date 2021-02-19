from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_extensions as de

from app import app

url = "https://assets4.lottiefiles.com/packages/lf20_e6MQKr.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

layout = [dcc.Markdown(""" ### Model Result""",
                       className="text-center font-weight-normal text-primary"),
        dcc.Markdown(""" 
        ###### o Metric Used - Root Mean Square Error
        
        ###### o Splitting Method - Stratified K Fold Method (5 Splits)""",
                               className="text-left font-weight-normal text-primary"),

          html.Div(de.Lottie(options=options, width="25%", height="15%", url=url)),


          dcc.Markdown(""" 
###### ------------- Fold 1 -------------
                                X_train: (20848, 5)
                                y_train: (20848,)
                                X_validation: (5213, 5)
                                y_validation: (5213,)

                                RMSE for validation set is 19290.508803963716
                                Model Fitment for validation set is 87 %"""),

          dcc.Markdown("""
###### ------------- Fold 2 -------------
                                X_train: (20849, 5)
                                y_train: (20849,)
                                X_validation: (5212, 5)
                                y_validation: (5212,)

                                RMSE for validation set is 31954.76870327114
                                Model Fitment for validation set is 81 %"""),

          dcc.Markdown("""
###### ------------- Fold 3 -------------
                                X_train: (20849, 5)
                                y_train: (20849,)
                                X_validation: (5212, 5)
                                y_validation: (5212,)

                                RMSE for validation set is 15519.85601969042
                                Model Fitment for validation set is 86 %"""),

          dcc.Markdown("""
###### ------------- Fold 4 -------------
                                X_train: (20849, 5)
                                y_train: (20849,)
                                X_validation: (5212, 5)
                                y_validation: (5212,)

                                RMSE for validation set is 21519.221378542166
                                Model Fitment for validation set is 87 %"""),

          dcc.Markdown("""
###### ------------- Fold 5 -------------
                                X_train: (20849, 5)
                                y_train: (20849,)
                                X_validation: (5212, 5)
                                y_validation: (5212,)

                                RMSE for validation set is 28187.026953349316
                                Model Fitment for validation set is 77 %"""),

          dcc.Markdown("""
##### ------------- Overall  -------------
                                 So, RMSE for oofs is 24048.462393272905
                                 & Overall Model Fitment for oofs is 83 % """),
          ]
