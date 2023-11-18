# HELP AND CHEATSHEETS
#-------------------------------------------------------------------
#https://hackerthemes.com/bootstrap-cheatsheet/#mt-1
#Bootstrap Themes: https://bootswatch.com/flatly/

# IMPORT LIBRARIES
#-------------------------------------------------------------------
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
#-------------------------------------------------------------------

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

df_stocks = pd.concat(data_frames, ignore_index=True) # enthält die aggregierten Daten mit der zusätzlichen Information "Titel" in der letzten Spalte
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
df_stocks['Return %'] = ((df_stocks['Return'] / df_stocks['Open']) *100).round(1)
# Monatszahlen
df_stocks['Monthly Return'] = df_stocks.groupby(['Stock/Index', 'MonthYear'])['Return'].transform('sum')
df_stocks['Open Value Month'] = df_stocks.groupby(['Stock/Index', 'MonthYear'])['Open'].transform('first') # wird nur für die Kalkulation benötigt
df_stocks['Monthly Return %'] = ((df_stocks['Monthly Return'] / df_stocks['Open Value Month']) *100).round(1)
# Jahreszahlen
df_stocks['Yearly Return'] = df_stocks.groupby(['Stock/Index', 'Year'])['Return'].transform('sum')
df_stocks['Open Value Year'] = df_stocks.groupby(['Stock/Index', 'Year'])['Open'].transform('first') # wird nur für die Kalkulation benötigt
df_stocks['Yearly Return %'] = ((df_stocks['Yearly Return'] / df_stocks['Open Value Year']) *100).round(1)

#Covid
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
#-------------------------------------------------------------------

def heatmap(filtered_stocks_df):
    # Kennzahlen erstellen
    df_heatmap = filtered_stocks_df.pivot(index='Stock/Index', columns='Date', values='Return %')

    # Heatmap erstellen
    fig = px.imshow(df_heatmap,
                    labels=dict(x='Date', y='Stock/Index', color='Return %'),
                    x=df_heatmap.columns,
                    y=df_heatmap.index,
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    zmin=-5,
                    zmax=5,
                    )

    # Layout
    fig.update_layout(
        title='Return % Heatmap for Stocks/Indices',
        width=800,
        xaxis_title='Date',
        xaxis=dict(rangeslider_visible=True,showspikes=True, spikethickness=2),
        yaxis_title='Stock/Index',
        coloraxis_colorbar=dict(thicknessmode="pixels", thickness=20, lenmode="pixels", len=250, yanchor="top", y=1, dtick=2),
        autosize=True,
    )

    return fig

def treemap(filtered_stocks_df):
    # Kennzahlen erstellen
    df_treemap = filtered_stocks_df.groupby('Stock/Index').agg(
        Open=('Open', 'first'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum')
    ).reset_index()

    df_treemap['Return'] = df_treemap['Close'] - df_treemap['Open']
    df_treemap['Return %'] = ((df_treemap['Return'] / df_treemap['Open']) * 100).round(1)

    # Treemap erstellen
    fig = px.treemap(df_treemap,
                     path=['Stock/Index'],
                     values='Return', # Volumen nehmen, geht aber noch nicht
                     color='Return %',
                     color_continuous_scale='RdYlGn',
                     color_continuous_midpoint=0,
                     title='Treemap: Volume and Return % by Stock/Index')

    # Layout
    fig.update_layout(
        margin = dict(t=50, l=25, r=25, b=25),
        font=dict(size=14, color='lightgray'),
    )

    fig.update_coloraxes(showscale=False) # Farbskala ausblenden

    # Texte
    return_text = ('Return: ' +df_treemap['Return'].round().apply(lambda x: '{:,.0f}'.format(x)).astype(str) + '\n(' + df_treemap['Return %'].astype(str) + '%)') # auf Volumen anpassen
    fig.update_traces(text=return_text, selector=dict(type='treemap'), textposition='middle center', insidetextfont=dict(size=20))

    return fig


def candle_chart(filtered_stocks_candle, filtered_covid_candle):

    # Candle-Chart erstellen
    fig = go.Figure()
    candlestick = go.Candlestick(x=filtered_stocks_candle['Date'],
                                 open=filtered_stocks_candle['Open'],
                                 high=filtered_stocks_candle['High'],
                                 low=filtered_stocks_candle['Low'],
                                 close=filtered_stocks_candle['Close'],
                                 name='Stock Price')
    fig.add_trace(candlestick)

    # Balkendiagramm für COVID-Cases hinzufügen
    bar_chart = go.Bar(x=filtered_covid_candle['Date'],
                       y=filtered_covid_candle['Cases'],
                       name='COVID-19 Cases',
                       marker=dict(color='blue'),
                       yaxis='y2')  # yaxis='y2' platziert die Daten auf der Sekundärachse
    fig.add_trace(bar_chart)

    # Layout
    fig.update_layout(title_text='Stock Price and COVID-19 Cases',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      yaxis2=dict(title='COVID-19 Cases', overlaying='y', side='right'),
                      xaxis_rangeslider_visible=True,
                      legend=dict(x=0.4, y=1.15, orientation='h'))  # x und y bestimmen die Position der Legende

    return fig

def candle_chart_trend(filtered_stocks_candle, filtered_covid_candle):

    # 20-Tage gleitenden Durchschnitt berechnen
    filtered_stocks_candle['20_MA'] = filtered_stocks_candle['Close'].rolling(window=20).mean()

    # Candle-Chart erstellen
    fig = go.Figure()

    # Candlestick
    candlestick = go.Candlestick(x=filtered_stocks_candle['Date'],
                                 open=filtered_stocks_candle['Open'],
                                 high=filtered_stocks_candle['High'],
                                 low=filtered_stocks_candle['Low'],
                                 close=filtered_stocks_candles['Close'],
                                 name='Stock Price')
    fig.add_trace(candlestick)

    # 20-Tage gleitenden Durchschnitt hinzufügen
    moving_avg = go.Scatter(x=filtered_stocks_candles['Date'],
                            y=filtered_stocks_candles['20_MA'],
                            mode='lines',
                            name='20-Day Moving Average',
                            line=dict(color='orange'))  # Hier die Linienfarbe auf Orange ändern
    fig.add_trace(moving_avg)

    # Balkendiagramm für COVID-Cases hinzufügen
    bar_chart = go.Bar(x=filtered_covid_candle['Date'],
                       y=filtered_covid_candle['Cases'],
                       name='COVID-19 Cases',
                       marker=dict(color='blue'),
                       yaxis='y2')  # yaxis='y2' platziert die Daten auf der Sekundärachse
    fig.add_trace(bar_chart)

    # Layout
    fig.update_layout(title_text='Stock Price with 20-Day Moving Average',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=True,
                      yaxis2=dict(title='COVID-19 Cases', overlaying='y', side='right'),
                      legend=dict(x=0.4, y=1.15, orientation='h'))  # Position der Legende

    return fig


 def scatterplot(filtered_stocks_df):
    # Kennzahlen erstellen
    df_volatility  = filtered_stocks_df.groupby(['Stock/Index', 'Year'])['Return %'].std()
    df_rendite  = filtered_stocks_df.groupby(['Stock/Index', 'Year'])['Return %'].mean()
    df_scatter = pd.DataFrame({'Rendite %': df_rendite, 'Volatilität %': df_volatility}).reset_index()

    # Scatterplot erstellen
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_scatter['Rendite %'], df_scatter['Volatilität %'], c=df_scatter['Year'], cmap='RdYlGn', alpha=0.7)
    plt.title('Scatterplot für Rendite und Volatilität')
    plt.xlabel('Durchschnittliche Rendite')
    plt.ylabel('Volatilität (Standardabweichung der Rendite)')

    # Legende für die Farbskala
    legend = plt.colorbar(scatter, label='Year')

    return plt

def scatterplot_cluster(filtered_stocks_df):
    # Kennzahlen erstellen
    df_volatility = filtered_stocks_df.groupby(['Stock/Index', 'Year'])['Return %'].std()
    df_rendite = filtered_stocks_df.groupby(['Stock/Index', 'Year'])['Return %'].mean()
    df_scatter = pd.DataFrame({'Rendite %': df_rendite, 'Volatilität %': df_volatility}).reset_index()
    df_scatter = df_scatter[['Rendite %', 'Volatilität %']]

    # Daten normalisieren
    scaler = StandardScaler()
    df_scatter_cluster = scaler.fit_transform(df_scatter)

    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(df_scatter_cluster)  # Datensatz hier hinterlegen
        wcss.append(kmeans.inertia_)

    # n_clusters berechnen
    kneedle = KneeLocator(range(1, 10), wcss, curve='convex', direction='decreasing')
    optimal_k = kneedle.knee

    # K-means-Modell
    kmeans_model = KMeans(n_clusters=3, random_state=0)
    df_scatter['Cluster'] = kmeans_model.fit_predict(df_scatter_cluster)

    fig = px.scatter(df_scatter, x='Rendite %', y='Volatilität %', color='Cluster',
                     color_continuous_scale='viridis',
                     title='Scatterplot für Rendite und Volatilität mit Clustering',
                     labels={'Rendite %': 'Durchschnittliche Rendite %', 'Volatilität %': 'Volatilität %'},
                     )

    fig.update_layout(
        showlegend=True,
        legend=dict(title='Cluster', yanchor='top', y=0.99, xanchor='left', x=0.01),
        coloraxis_colorbar=dict(title='Cluster'),
    )

    fig.show()



# START APP
#-------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],

                # make it mobile-friendly
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
server = app.server


# LAYOUT SECTION: BOOTSTRAP
#--------------------------------------------------------------------



# Callback-Funktionen
@app.callback(
    Output('scatter-plot', 'figure'),

    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date'),

)


def update_figures(start_date, end_date, stock_index, covid_country):
    min_date = pd.to_datetime(start_date)
    max_date = pd.to_datetime(end_date)

    filtered_stocks_df = df_stocks[(df_stocks['Date'] >= min_date) & (df_stocks['Date'] <= max_date)]
    filtered_covid_df = df_covid[(df_stocks['Date'] >= min_date) & (df_covid['Date'] <= max_date)]

    if stock_index:
        filtered_stocks_candle = filtered_stocks_df[filtered_stocks_df['Stock/Index'].isin(stock_index)]

    if covid_country:
        filtered_covid_candle = filtered_covid_df[filtered_covid_df['Country'].isin(covid_country)]




    return # Visuals einfügen




# RUN THE APP
#--------------------------------------------------------------------
if __name__=='__main__':
    app.run_server(debug=False, port=8055)