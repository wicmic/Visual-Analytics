# HELP AND CHEATSHEETS
# -------------------------------------------------------------------
# https://hackerthemes.com/bootstrap-cheatsheet/#mt-1
# Bootstrap Themes: https://bootswatch.com/flatly/
# Farbschemas aus https://plotly.com/python/builtin-colorscales/

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


# neue Measures hinzufügen
df_stocks['Return'] = (df_stocks['Close'] - df_stocks['Open'])
df_stocks['Return %'] = ((df_stocks['Return'] / df_stocks['Open']) * 100).round(1)
# Monatszahlen
df_stocks['Monthly Return'] = df_stocks.groupby(['Stock/Index', 'MonthYear'])['Return'].transform('sum')
df_stocks['Open Value Month'] = df_stocks.groupby(['Stock/Index', 'MonthYear'])['Open'].transform('first')  # wird nur für die Kalkulation benötigt
df_stocks['Monthly Return %'] = ((df_stocks['Monthly Return'] / df_stocks['Open Value Month']) * 100).round(1)
# Jahreszahlen
df_stocks['Yearly Return'] = df_stocks.groupby(['Stock/Index', 'Year'])['Return'].transform('sum')
df_stocks['Open Value Year'] = df_stocks.groupby(['Stock/Index', 'Year'])['Open'].transform('first')  # wird nur für die Kalkulation benötigt
df_stocks['Yearly Return %'] = ((df_stocks['Yearly Return'] / df_stocks['Open Value Year']) * 100).round(1)

# Covid
file_path = r'D:\Python\Visual Analytics\Visual-Analytics\Data_Covid\owid-covid-data.csv'

df_covid = pd.read_csv(file_path, delimiter=',')
# nicht benötigte Spalten löschen
columns_to_keep = ['iso_code', 'continent', 'location', 'date', 'new_cases']
df_covid.drop(df_covid.columns.difference(columns_to_keep), axis=1, inplace=True)
# Spalten umbenennen
df_covid.columns = df_covid.columns.str.replace('iso_code', 'Country short')
df_covid.columns = df_covid.columns.str.replace('continent', 'Continent')
df_covid.columns = df_covid.columns.str.replace('location', 'Country')
df_covid.columns = df_covid.columns.str.replace('date', 'Date')
df_covid.columns = df_covid.columns.str.replace('new_cases', 'New Cases')

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
        title='Average Return and Volatility per Day',
        labels={'Rendite %': 'avg. Return %', 'Volatilität %': 'Volatility %'},
        hover_data=['Stock/Index']
    )

    # Layout
    fig.update_layout(
        plot_bgcolor='black',  # Hintergrundfarbe
        paper_bgcolor='black',  # Hintergrundfarbe des gesamten Plots
        font=dict(color='darkgrey', family='Arial, sans-serif'),  # Schriftfarbe- und -art
        coloraxis_colorbar=dict(title='Jahr'),
        coloraxis=dict(colorscale='Greens'),
        xaxis=dict(
            showgrid=False,  # Gitterlinien ausblenden
            title='avg. Return %',
            zeroline=True,  # Nulllinie anzeigen
            zerolinecolor='darkgrey'  # Farbe der Nulllinie
        ),
        yaxis=dict(
            showgrid=False,  # Gitterlinien ausblenden
            title='Volatility %',
            zeroline=True,  # Nulllinie anzeigen
            zerolinecolor='darkgrey',  # Farbe der Nulllinie
            rangemode = 'tozero'  # startet bei 0
        )
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
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
        kmeans.fit(df_scatter_cluster)
        wcss.append(kmeans.inertia_)

    kmeans_model = KMeans(n_clusters=3, random_state=0)
    df_scatter['Cluster'] = kmeans_model.fit_predict(df_scatter_cluster)
    df_scatter

    fig = px.scatter(df_scatter, x='Rendite %', y='Volatilität %', color='Cluster',
                     color_continuous_scale='Greens',
                     labels={'Rendite %': 'avg. Return %', 'Volatilität %': 'Volatility %'},
                     title='Average Return and Volatility per Day',
                     )
    # Layout
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='black',  # Hintergrundfarbe
        paper_bgcolor='black',  # Hintergrundfarbe des gesamten Plots
        legend=dict(title='Cluster', yanchor='top', y=0.99, xanchor='left', x=0.01),
        coloraxis_colorbar=dict(title='Cluster'),
        font=dict(color='darkgrey', family='Arial, sans-serif'),
        coloraxis=dict(colorscale='Greens'),
        xaxis=dict(
            showgrid=False,  # Gitterlinien ausblenden
            title='avg. Return %',
            zeroline=True,  # Nulllinie anzeigen
            zerolinecolor='darkgrey'  # Farbe der Nulllinie
        ),
        yaxis=dict(
            showgrid=False,  # Gitterlinien ausblenden
            title='Volatility %',
            zeroline=True,  # Nulllinie anzeigen
            zerolinecolor='darkgrey',  # Farbe der Nulllinie
            rangemode='tozero'  # startet bei 0
        )

    )

    return fig

def treemap(df_filtered_date_only):
    # Kennzahlen erstellen
    df_treemap = df_filtered_date_only.groupby('Stock/Index').agg(
        Open=('Open', 'first'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum')
        ).reset_index()

    df_treemap['Return'] = df_treemap['Close'] - df_treemap['Open']
    df_treemap['Return %'] = ((df_treemap['Return'] / df_treemap['Open']) * 100).round(1)

    # Treemap erstellen
    fig = px.treemap(df_treemap,
                     path=['Stock/Index'],
                     values='Return',
                     color='Return %',
                     color_continuous_scale='RdYlGn',
                     color_continuous_midpoint=0)

    # Layout
    fig.update_layout(
        plot_bgcolor='black',  # Hintergrundfarbe des Plots
        paper_bgcolor='black',  # Hintergrundfarbe des gesamten Plots
        margin=dict(t=50, l=25, r=25, b=25),
        font=dict(size=14, color='darkgrey', family='Arial, sans-serif'),
    ) # graue Fläche am Rand geht nicht weg - weshalb?

    fig.update_coloraxes(showscale=False) # Farbskala ausblenden

    # Beschriftungen
    return_text = ('Return: ' +df_treemap['Return'].round().apply(lambda x: '{:,.0f}'.format(x)).astype(str) + '\n(' + df_treemap['Return %'].astype(str) + '%)')
    fig.update_traces(text=return_text, selector=dict(type='treemap'), textposition='middle center', insidetextfont=dict(size=20))

    return fig

def heatmap(df_filtered_date_only):
    # Kennzahlen erstellen
    df_heatmap = df_filtered_date_only.pivot(index='Stock/Index', columns='Date', values='Return %')

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
        #width=800,
        #height=600,
        plot_bgcolor='black',  # Hintergrundfarbe des Plots
        paper_bgcolor='black',  # Hintergrundfarbe des gesamten Plots
        font=dict(color='darkgrey'),  # Schriftfarbe
        #xaxis_title='Date',
        #yaxis_title='Stock/Index',
        coloraxis_colorbar=dict(thicknessmode="pixels", thickness=20, lenmode="pixels", len=250, yanchor="top", y=1, dtick=2),
        xaxis=dict(rangeslider_visible=True, showspikes=True, spikethickness=2),
        autosize=True,
    )

    return fig

def candle_chart_trend(df_filtered_stock, df_filtered_covid_date):
    # Kennzahlen erstellen
    df_candle_chart_stocks = df_filtered_stock
    df_candle_chart_covid = df_filtered_covid_date

    # 20-Tage gleitenden Durchschnitt berechnen
    df_candle_chart_stocks['20_MA'] = df_candle_chart_stocks['Close'].rolling(window=20).mean()

    # Candle-Chart erstellen
    fig = go.Figure()

    # Candlestick
    candlestick = go.Candlestick(x=df_candle_chart_stocks['Date'],
                                 open=df_candle_chart_stocks['Open'],
                                 high=df_candle_chart_stocks['High'],
                                 low=df_candle_chart_stocks['Low'],
                                 close=df_candle_chart_stocks['Close'],
                                 name='Stock Price')
    fig.add_trace(candlestick)

    # 20-Tage gleitenden Durchschnitt hinzufügen
    moving_avg = go.Scatter(x=df_candle_chart_stocks['Date'],
                            y=df_candle_chart_stocks['20_MA'],
                            mode='lines',
                            name='20-Day Moving Average',
                            line=dict(color='tan'))
    fig.add_trace(moving_avg)

    # Balkendiagramm für COVID-Cases hinzufügen
    bar_chart = go.Bar(x=df_candle_chart_covid['Date'],
                       y=df_candle_chart_covid['New Cases'],
                       name='COVID-19 Cases',
                       marker=dict(color='papayawhip'),
                       yaxis='y2')
    fig.add_trace(bar_chart)

    # Layout
    fig.update_layout(
        plot_bgcolor='black',  # Hintergrundfarbe des Plots
        paper_bgcolor='black',  # Hintergrundfarbe des gesamten Plots
        font=dict(color='darkgrey', family='Arial, sans-serif'),  # Schriftfarbe und -art
        #title_text='Stock Price with 20-Day Moving Average and COVID-19 Cases',
        xaxis_rangeslider_visible=True, # Slider einblenden
        legend=dict(x=0.2, y=1.1, orientation='h'), # Position der Legende

        xaxis = dict(
            showgrid=False,  # Gitterlinien ausblenden
            #title='Date',
            zeroline=True,  # Nulllinie anzeigen
            zerolinecolor='darkgrey'  # Farbe der Nulllinie
        ),
        yaxis = dict(
            showgrid=False,  # Gitterlinien ausblenden
            title='Stock Price',
            zeroline=True,  # Nulllinie anzeigen
            zerolinecolor='darkgrey',  # Farbe der Nulllinie
            rangemode='tozero'  # startet bei 0
        ),
        yaxis2 = dict(
            showgrid=False,  # Gitterlinien ausblenden
            title='COVID-19 Cases',
            overlaying='y',
            side='right',
            zeroline=False,  # Nulllinie nicht anzeigen
            # zerolinecolor='darkgrey',  # Farbe der Nulllinie
            rangemode='tozero'  # startet bei 0
        )
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
app.layout = html.Div( style={
        'backgroundColor': 'black',  # Set the background color to black
        'color': 'darkgrey',
        'fontFamily': 'Arial, sans-serif',
        },
    children=[
        html.H1("Stock & Covid Dashboard"),
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            dbc.Row(                            # row0
                className='mt-3',
                children=[

                    dbc.Col(
                        dcc.DatePickerRange(
                            id='date-slider',
                            min_date_allowed=df_stocks['Date'].min(),
                            max_date_allowed=df_stocks['Date'].max(),
                            start_date=df_stocks['Date'].min(),
                            end_date=df_stocks['Date'].max(),
                            display_format='YYYY-MM-DD',
                            style={
                                'height': '40px',
                                'backgroundColor': 'black',
                                'color': 'black',
                                'whiteSpace': 'normal',
                                'fontFamily': 'Arial, sans-serif',
                                'margin-bottom': '10px',
                            }
                        ),
                    ),

                    dbc.Col(),


                    dbc.Col(
                            dcc.Dropdown(
                                id='country-dropdown',
                                options=[
                                    {'label': country, 'value': country} for country in df_covid['Country'].unique()
                                ],
                                value='Switzerland',
                                placeholder='Select a country',
                                style={
                                    'backgroundColor': 'black',
                                    'color': 'black',
                                    'textAlign': 'left',
                                    'fontFamily': 'Arial, sans-serif',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    'margin-bottom': '10px',
                                }
                            ),
                    ),

                    dbc.Col(
                            dcc.Dropdown(
                                id='stock-dropdown2',
                                options=[
                                    {'label': stock, 'value': stock} for stock in df_stocks['Stock/Index'].unique()
                                ],
                                value='BTC-USD',
                                placeholder='Select a stock/index',
                                style={
                                    'backgroundColor': 'black',
                                    'color': 'black',
                                    'textAlign': 'left',
                                    'fontFamily': 'Arial, sans-serif',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    'margin-bottom': '10px',
                                }
                            ),
                    ),

                    dbc.Col(),

                    dbc.Col(
                            dcc.Dropdown(
                                id='stock-dropdown',
                                options=[
                                    {'label': stock, 'value': stock} for stock in df_stocks['Stock/Index'].unique()
                                ],
                                multi=True,  # Mehfachauswahl erlauben
                                value=None,
                                placeholder='Select stocks/indexes',
                                style={
                                    'backgroundColor': 'black',
                                    'color': 'black',
                                    'textAlign': 'left',
                                    'fontFamily': 'Arial, sans-serif',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    'margin-bottom': '10px',
                                    }
                            ),
                    ),
                ], style={'width': '100%'}),



            dbc.Row([                           # row1
                dbc.Col([                       # Col1
                    # Table Covid
                    dash_table.DataTable(
                        id='table-covid',
                        columns=[
                            {'name': 'Start', 'id': 'Start_Cases'},
                            {'name': 'End', 'id': 'End_Cases'},
                            {'name': 'Acc.', 'id': 'End_Acc'},
                            {'name': 'High', 'id': 'High', 'type': 'numeric', 'presentation': 'positive'},
                            {'name': 'Low', 'id': 'Low', 'type': 'numeric', 'presentation': 'negative'}
                        ],
                        style_table={'overflowX': 'auto', 'fontFamily': '-apple-system'},
                        style_header={
                            'backgroundColor': 'black',
                            'color': 'darkgrey'
                        },
                        style_data={
                            'backgroundColor': 'black',
                            'color': 'darkgrey'
                        },
                        style_as_list_view=True,
                        style_cell={'textAlign': 'left',
                                    'fontSize': '75%',
                                    'fontFamily': 'Arial, sans-serif',
                                    'whiteSpace': 'normal',
                                    'height': 'auto'},
                    ),
                ], width={'size': 3}),
                dbc.Col([               # Col2

                ], width={'size': 3}),
                dbc.Col([               # Col3

                ], width={'size': 5}),
            ], style={'width': '100%', 'margin-bottom': '15px',}),

            dbc.Row([                   #row2
                dbc.Col([               #Col1
                    # Table Stocks
                    dash_table.DataTable(
                        id='table-stocks',
                        columns=[
                            {'name': 'Stock/Index', 'id': 'Stock/Index'},
                            {'name': 'Start', 'id': 'Start'},
                            {'name': 'End', 'id': 'End'},
                            {'name': 'High', 'id': 'High', 'type': 'numeric', 'presentation': 'positive'},
                            {'name': 'Low', 'id': 'Low', 'type': 'numeric', 'presentation': 'negative'},
                            {'name': 'Volume (billions)', 'id': 'Volume'},
                            {'name': 'Return', 'id': 'Return'},
                            {'name': 'Return %', 'id': 'Return %'}
                        ],
                        style_table={'overflowX': 'auto', 'fontFamily': '-apple-system'},
                        style_header={
                            'backgroundColor': 'black',
                            'color': 'darkgrey'
                        },
                        style_data_conditional=[
                            {'if': {'filter_query': '{Return} >= 0'},
                            'backgroundColor': 'forestgreen',
                            'color': 'darkgrey'
                            },
                            {'if': {'filter_query': '{Return} < 0'},
                            'backgroundColor': 'darkred',
                            'color': 'black' # in grau nicht leserlich
                            }
                        ],
                        style_as_list_view=True,
                        page_current= 0,
                        page_size= 15,
                        style_cell ={'textAlign': 'left',
                                        'fontSize': '75%',
                                        'fontFamily': 'Arial, sans-serif',
                                        'whiteSpace': 'normal',
                                        'height': 'auto'},
                    ),
                ], width={'size': 3}),
                dbc.Col([               #Col2
                    # Candle-Chart
                    dcc.Graph(id='candle-plot-trend'),
                ], width={'size': 6}),
                dbc.Col([               #Col3
                    # Scatterplot
                    dcc.Graph(id='scatter-plot'),
                ], width={'size': 3}),
            ], style={'width': '100%', 'margin-bottom': '1px',}),



            dbc.Row([                   #row3
                dbc.Col([               #Col1
                    # Treemap
                    dcc.Graph(id='treemap-plot'),
                ], width={'size': 5}),


                dbc.Col([               #Col2
                    # Heatmap
                    dcc.Graph(id='heatmap-plot'),

                ], width={'size': 7}),
            ], style={'width': '100%'}),


                    # Scatterplot Cluster
                    dcc.Graph(id='scatter-plot-cluster'),





        ]
    )
])


# Callback-Funktionen
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('scatter-plot-cluster', 'figure'),
    Output('treemap-plot', 'figure'),
    Output('heatmap-plot', 'figure'),
    Output('candle-plot-trend', 'figure'),
    Output('table-stocks', 'data'),
    Output('table-covid', 'data'),
    [Input('stock-dropdown', 'value'),
     Input('stock-dropdown2', 'value'),
     Input('country-dropdown', 'value'),
     Input('date-slider', 'start_date'),
     Input('date-slider', 'end_date')]
)

def update_figures(selected_stocks, selected_stock, selected_country, start_date, end_date):
    min_date = pd.to_datetime(start_date)
    max_date = pd.to_datetime(end_date)


    # für Visuals, welche Stocks eingschränken können
    if not selected_stocks:   # wenn nichts ausgewählt
        df_filtered_stocks = df_stocks[(df_stocks['Date'] >= min_date) & (df_stocks['Date'] <= max_date)]
    else:
        df_filtered_stocks = df_stocks[(df_stocks['Stock/Index'].isin(selected_stocks)) &
                                       (df_stocks['Date'] >= min_date) &
                                       (df_stocks['Date'] <= max_date)]

    # für Visuals, welche nur ein Stock auswählen können
    if not selected_stock:   # wenn nichts ausgewählt
        df_filtered_stock = df_stocks[(df_stocks['Date'] >= min_date) & (df_stocks['Date'] <= max_date)]
    else:
        df_filtered_stock = df_stocks[(df_stocks['Stock/Index'].isin([selected_stock])) &
                                       (df_stocks['Date'] >= min_date) &
                                       (df_stocks['Date'] <= max_date)]


    # für Visuals, welche immer alle Stocks zeigen
    df_filtered_date_only = df_stocks[(df_stocks['Date'] >= min_date) & (df_stocks['Date'] <= max_date)]

    # für Covid-Visuals, Country muss ausgewählt sein
    df_filtered_covid_date = df_covid[(df_covid['Date'] >= min_date) & (df_covid['Date'] <= max_date) & (df_covid['Country'] == selected_country)]


    # Table Stocks
    df_table_stocks = df_filtered_date_only.groupby('Stock/Index').agg(
        Start=('Open', 'first'),
        End=('Close', 'last'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Volume=('Volume', 'sum'),
    ).reset_index()

    df_table_stocks['Return'] = (df_table_stocks['End'] - df_table_stocks['Start']).round(0)
    df_table_stocks['Return %'] = ((df_table_stocks['Return'] / df_table_stocks['Start']) * 100).round(2)
    df_table_stocks[['Start', 'End', 'High', 'Low']] = df_table_stocks[['Start', 'End', 'High', 'Low']].round(0)
    df_table_stocks['Volume'] = (df_table_stocks['Volume'] / 1000000000).round(1)       # in Milliarden / billions

    # Table Covid
    df_table_covid = df_filtered_covid_date.groupby('Country').agg(
        Start_Cases=('New Cases', 'first'),
        End_Cases=('New Cases', 'last'),
        Start_Acc=('New Cases', 'first'),
        End_Acc=('New Cases', 'sum'),
        High=('New Cases', 'max'),
        Low=('New Cases', 'min'),
        Cases=('New Cases', 'sum')
    ).reset_index()


    # Definition der Visuals
    scatter_fig = scatter_plot(df_filtered_stocks)
    scatter_cluster_fig = scatter_plot_cluster(df_filtered_stocks)
    treemap_fig = treemap(df_filtered_date_only)
    heatmap_fig = heatmap(df_filtered_date_only)
    candle_trend_fig = candle_chart_trend(df_filtered_stock, df_filtered_covid_date)

    # zurückgeben
    return scatter_fig, scatter_cluster_fig, treemap_fig, heatmap_fig, candle_trend_fig, df_table_stocks.to_dict('records'), df_table_covid.to_dict('records')



# RUN THE APP
# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False, port=8055)