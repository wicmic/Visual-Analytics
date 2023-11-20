# HELP AND CHEATSHEETS
# -------------------------------------------------------------------
# https://hackerthemes.com/bootstrap-cheatsheet/#mt-1
# Bootstrap Themes: https://bootswatch.com/flatly/

# IMPORT LIBRARIES
# -------------------------------------------------------------------
import pandas as pd
import os
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
import locale
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# IMPORT DATA
# -------------------------------------------------------------------

# Stocks
folder_path = 'D:\Python\Visual Analytics\Visual-Analytics\Data_stocks'  # Ordnerpfad allenfalls anpassen
data_frames = []

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        filename = os.path.splitext(file)[0]  # Dateiname
        data['Stock/Index'] = filename  # Hinzufügen der Dateinamens-Spalte
        data_frames.append(data)

df_stocks = pd.concat(data_frames,
                      ignore_index=True)  # enthält die aggregierten Daten mit der zusätzlichen Information "Titel" in der letzten Spalte
df_stocks.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# Datum formatieren und Infomationen hinzufügen
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], format='%Y-%m-%d')
df_stocks['Month'] = pd.to_datetime(df_stocks['Date']).dt.month
df_stocks['Year'] = pd.to_datetime(df_stocks['Date']).dt.year
df_stocks['MonthYear'] = df_stocks['Date'].dt.to_period('M')
df_stocks['Open'] = df_stocks['Open']
df_stocks['Close'] = df_stocks['Close']
df_stocks['High'] = df_stocks['High']
df_stocks['Low'] = df_stocks['Low']
df_stocks['Adj Close'] = df_stocks['Adj Close']
df_stocks['Volume'] = df_stocks['Volume'].round().astype(int)

# neue Measures hinzufügen
df_stocks['Return'] = (df_stocks['Close'] - df_stocks['Open'])
df_stocks['Return %'] = ((df_stocks['Return'] / df_stocks['Open']) * 100).round(1)
# Monatszahlen
df_stocks['Monthly Return'] = df_stocks.groupby(['Stock/Index', 'MonthYear'])['Return'].transform('sum')
df_stocks['Open Value Month'] = df_stocks.groupby(['Stock/Index', 'MonthYear'])['Open'].transform(
    'first')  # wird nur für die Kalkulation benötigt
df_stocks['Monthly Return %'] = ((df_stocks['Monthly Return'] / df_stocks['Open Value Month']) * 100).round(1)
# Jahreszahlen
df_stocks['Yearly Return'] = df_stocks.groupby(['Stock/Index', 'Year'])['Return'].transform('sum')
df_stocks['Open Value Year'] = df_stocks.groupby(['Stock/Index', 'Year'])['Open'].transform(
    'first')  # wird nur für die Kalkulation benötigt
df_stocks['Yearly Return %'] = ((df_stocks['Yearly Return'] / df_stocks['Open Value Year']) * 100).round(1)

# Covid
file_path = r'D:\Python\Visual Analytics\Visual-Analytics\Data_Covid\owid-covid-data.csv'

df_covid = pd.read_csv(file_path, delimiter=',')
# nicht benötigte Spalten löschen
columns_to_keep = ['iso_code', 'continent', 'location', 'date', 'new_cases', 'total_cases']
df_covid.drop(df_covid.columns.difference(columns_to_keep), axis=1, inplace=True)
# Spalten umbenennen
df_covid.columns = df_covid.columns.str.replace('iso_code', 'Country short')
df_covid.columns = df_covid.columns.str.replace('continent', 'Continent')
df_covid.columns = df_covid.columns.str.replace('location', 'Country')
df_covid.columns = df_covid.columns.str.replace('date', 'Date')
df_covid.columns = df_covid.columns.str.replace('total_cases', 'Cases Acc.')
df_covid.columns = df_covid.columns.str.replace('new_cases', 'Cases')

# Datum formatieren
df_covid['Date'] = pd.to_datetime(df_covid['Date'], format='%Y-%m-%d')


# FUNKTIONEN
# -------------------------------------------------------------------

def scatter_plot(df_filtered_stocks):
    # Kennzahlen erstellen
    df_volatility = df_filtered_stocks.groupby(['Stock/Index', 'Year'])['Return %'].std()
    df_rendite = df_filtered_stocks.groupby(['Stock/Index', 'Year'])['Return %'].mean()
    df_scatter = pd.DataFrame({'Rendite %': df_rendite, 'Volatilität %': df_volatility}).reset_index()

    # Scatterplot erstellen
    fig = px.scatter(
        df_scatter,
        x='Rendite %',
        y='Volatilität %',
        color='Year',
        title='Scatterplot für Rendite und Volatilität',
        labels={'Rendite %': 'Durchschnittliche Rendite', 'Volatilität %': 'Volatilität (Standardabweichung der Rendite)'}
    )

    return fig

def scatter_plot_cluster(df_filtered_stocks):
    # Kennzahlen erstellen
    df_volatility = df_filtered_stocks.groupby(['Stock/Index', 'Year'])['Return %'].std()
    df_rendite = df_filtered_stocks.groupby(['Stock/Index', 'Year'])['Return %'].mean()
    df_scatter = pd.DataFrame({'Rendite %': df_rendite, 'Volatilität %': df_volatility}).reset_index()
    df_scatter = df_scatter[['Rendite %', 'Volatilität %']]

    # Daten normalisieren
    scaler = StandardScaler()
    df_scatter_cluster = scaler.fit_transform(df_scatter)

    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(df_scatter_cluster)
        wcss.append(kmeans.inertia_)

    kmeans_model = KMeans(n_clusters=3, random_state=0)
    df_scatter['Cluster'] = kmeans_model.fit_predict(df_scatter_cluster)
    df_scatter

    fig = px.scatter(df_scatter, x='Rendite %', y='Volatilität %', color='Cluster',
                     color_continuous_scale='viridis',
                     labels={'Rendite %': 'avg. Return %', 'Volatilität %': 'Volatility %'},
                     )

    fig.update_layout(
        showlegend=True,
        legend=dict(title='Cluster', yanchor='top', y=0.99, xanchor='left', x=0.01),
        coloraxis_colorbar=dict(title='Cluster'),
    )

    return fig


# START APP
# -------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],

           # make it mobile-friendly
           meta_tags=[{'name': 'viewport',
                       'content': 'width=device-width, initial-scale=1.0'}]
           )
server = app.server


# LAYOUT SECTION: BOOTSTRAP
# --------------------------------------------------------------------
app.layout = html.Div([
    html.H1("Stock & Covid Dashboard"),

    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': stock, 'value': stock} for stock in df_stocks['Stock/Index'].unique()
        ],
        multi=True,  # Allow multiple selections
        value=None,
        placeholder='Select stock(s)'
    ),
    dcc.DatePickerRange(
        id='date-slider',
        min_date_allowed=df_stocks['Date'].min(),
        max_date_allowed=df_stocks['Date'].max(),
        start_date=df_stocks['Date'].min(),
        end_date=df_stocks['Date'].max(),
        display_format='YYYY-MM-DD',
        style={'height': '40px'}
    ),
    # Scatterplot
    dcc.Graph(id='scatter-plot'),
    # Scatterplot Cluster
    dcc.Graph(id='scatter-plot-cluster'),
])



# Callback-Funktionen
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('scatter-plot-cluster', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('date-slider', 'start_date'),
     Input('date-slider', 'end_date')]
)

def update_figures(selected_stocks, start_date, end_date):
    min_date = pd.to_datetime(start_date)
    max_date = pd.to_datetime(end_date)

    if not selected_stocks:   # wenn nichts ausgewählt
        df_filtered_stocks = df_stocks[(df_stocks['Date'] >= min_date) & (df_stocks['Date'] <= max_date)]
    else:
        df_filtered_stocks = df_stocks[(df_stocks['Stock/Index'].isin(selected_stocks)) &
                                       (df_stocks['Date'] >= min_date) &
                                       (df_stocks['Date'] <= max_date)]

    scatter_fig = scatter_plot(df_filtered_stocks)
    scatter_cluster_fig = scatter_plot_cluster(df_filtered_stocks)

    return scatter_fig, scatter_cluster_fig



# RUN THE APP
# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False, port=8055)