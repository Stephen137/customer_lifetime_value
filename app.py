import dash
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from dash import dcc

import plotly.express as px

import pandas as pd
import numpy as np

import pathlib
import pickle as pickle

# APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
LOGO = "https://stephen137.github.io/Green%20Man.jpg"

# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()

# DATA
with open(ART_PATH.joinpath("predictions_df.pkl"), 'rb') as file_obj:
    predictions_df = pickle.load(file_obj)

df = predictions_df \
    .assign(
        spend_actual_vs_pred = lambda x: x['spend_90_total'] - x['pred_spend'] 
    )

# LAYOUT

# Slider Marks
x = np.linspace(df['spend_actual_vs_pred'].min(), df['spend_actual_vs_pred'].max(), 10, dtype=int)
x = x.round(0)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("by Stephen Barrie", className="ml-2")),
                ],
                align="center",
                #no_gutters=True,
            ),
            href="https://www.linkedin.com/in/sjbarrie/",
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(
            id="navbar-collapse", navbar=True, is_open=False
        ),
    ],
    color="dark",
    dark=True,
)

app.layout = html.Div(
    children = [
        navbar, 
        dbc.Row(
            [
                dbc.Col(
                    [
                        
                        html.H4("Google Merchandise Store analytics dashboard"),
                        html.Div(
                            id="intro",
                            children="I created this interactive dashboard as the final step of a detailed analysis of the Google Analytics Sample dataset. Happy hovering! The dataset provides 12 months (August 2016 to August 2017) of obfuscated Google Analytics 360 data from the Google Merchandise Store , a real ecommerce store that sells Google-branded merchandise, in BigQuery. It’s a great way analyze business data and learn the benefits of using BigQuery to analyze Analytics 360",
                        ),
                        html.Br(),
                        html.Hr(),
                        html.H5("Actual vs predicted spend"),
                        html.P("The slider below represents the absolute difference between predicted and actual spend during the 90 day period, 4 May to 1 August 2017. The predictions were made using an XGBoost regression model. Use these business insights to allocate marketing resource to those customers that were predicted to spend but didn't (represented by blue dots in the interactive plot opposite)"),
                        html.H5("Open to Work"),
                        html.P("Now...time for some shameless advertising! I am an experienced chartered accountant, auditor and business analyst with a solid understanding of data engineering and machine learning techniques. Please reach out if you feel I may be a good fit for your team"),
                        dcc.Slider(
                            id    = 'spend-slider', 
                            value = df['spend_actual_vs_pred'].max(),
                            max   = df['spend_actual_vs_pred'].max(),
                            min   = df['spend_actual_vs_pred'].min(), 
                            marks = {i: '$'+str(i) for i in range(x[0],x[-1]) if i % 300 == 0}
                        ),
                        html.Br(),
                        html.Button("Download Segmentation", id="btn"), dcc.Download(id="download")
                    ],
                    width = 3,
                    style={'margin':'10px'}
                ),
                dbc.Col(
                    dcc.Graph(id='graph-slider'),
                    width = 8
                )
            ] 
        )
    ]
)

# CALLBACKS 
@app.callback(
    Output('graph-slider', 'figure'),
    Input('spend-slider', 'value'))
def update_figure(spend_delta_max):
    
    df_filtered = df[df['spend_actual_vs_pred'] <= spend_delta_max]

    fig = px.scatter(
        data_frame=df_filtered,
        x = 'frequency',
        y = 'pred_prob',
        color = 'spend_actual_vs_pred', 
        color_continuous_midpoint=0, 
        opacity=0.5, 
        color_continuous_scale='IceFire', 
        hover_name='fullVisitorId',
        hover_data=['spend_90_total', 'pred_spend'],
    ) \
        .update_layout(
            {
                'plot_bgcolor': PLOT_BACKGROUND,
                'paper_bgcolor':PLOT_BACKGROUND,
                'font_color': PLOT_FONT_COLOR,
                'height':700
            }
        ) \
        .update_traces(
            marker = dict(size = 12)
        )
    
    return fig

# Download Button
@app.callback(
    Output("download", "data"), 
    Input("btn", "n_clicks"), 
    State('spend-slider', 'value'),
    prevent_initial_call=True,
)
def func(n_clicks, spend_delta_max):
    
    df_filtered = df[df['spend_actual_vs_pred'] <= spend_delta_max]

    return dcc.send_data_frame(df_filtered.to_csv, "customer_segmentation.csv")

# Navbar
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)