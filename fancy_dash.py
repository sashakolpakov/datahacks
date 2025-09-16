import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import base64
import io
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Import the data collection module
from data_collect import main, Config

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Forex Advanced Analytics Dashboard"

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Forex Advanced Analytics Dashboard"), className="mb-2 mt-4")
    ]),
    
    dbc.Row([
        # Left sidebar for configuration
        dbc.Col([
            html.H4("Data Collection"),
            html.Hr(),
            
            # Currency pair selection
            html.Label("Select Currency Pair"),
            dcc.Dropdown(
                id="currency-pair-dropdown",
                options=[
                    {"label": pair, "value": pair} for pair in 
                    ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "GBPJPY", "GBPAUD"]
                ],
                value="GBPAUD",
                className="mb-3"
            ),
            
            # Date range selection
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id="date-range",
                start_date=(datetime.now() - timedelta(days=365*5)).date(),  # 5 years ago
                end_date=datetime.now().date(),
                className="mb-3"
            ),
            
            # Advanced options
            html.Div([
                html.H5("Data Options"),
                dbc.Checklist(
                    id="advanced-options",
                    options=[
                        {"label": "Include All Major Currencies", "value": "all_currencies"},
                        {"label": "Include Commodity Data", "value": "commodities"},
                        {"label": "Include Stock Indices", "value": "indices"},
                        {"label": "Include Special Indicators", "value": "special"}
                    ],
                    value=["all_currencies", "commodities", "indices", "special"],
                    className="mb-3"
                ),
            ]),
            
            # API Keys (collapsible)
            dbc.Button(
                "API Keys",
                id="collapse-button",
                className="mb-3",
                color="secondary",
            ),
            dbc.Collapse(
                dbc.Card(dbc.CardBody([
                    dbc.Input(id="fred-key", placeholder="FRED API Key", type="password", className="mb-2"),
                    dbc.Input(id="polygon-key", placeholder="Polygon.io API Key", type="password"),
                ])),
                id="collapse",
                is_open=False,
            ),
            
            # Submit button
            dbc.Button("Collect Data", id="collect-button", color="primary", className="mt-3"),
            
            # Loading spinner
            dcc.Loading(
                id="loading-data",
                type="default",
                children=html.Div(id="loading-output-data"),
            ),
            
            # Download button
            html.Div(
                dbc.Button("Download CSV", id="btn-download", 
                          color="success", className="mt-3", style={"display": "none"}),
                id="download-btn-container",
                className="d-grid gap-2 mt-3"
            ),
            dcc.Download(id="download-dataframe-csv"),
            
        ], width=3),
        
        # Main content area
        dbc.Col([
            # Initial instructions card
            html.Div(
                id="instructions-card",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Advanced Forex Analytics", className="card-title"),
                            html.P("This dashboard provides advanced analytics capabilities for forex data analysis:"),
                            html.Ul([
                                html.Li("Data Collection from multiple sources"),
                                html.Li("Basic visualizations and correlations"),
                                html.Li("Cluster Analysis to identify market regimes"),
                                html.Li("Anomaly Detection to spot unusual market conditions"),
                                html.Li("Principal Component Analysis to understand market drivers")
                            ]),
                            html.P("Configure your options in the sidebar and click 'Collect Data' to begin."),
                        ])
                    )
                ]
            ),
            
            # Results area (hidden initially)
            html.Div(
                id="results-container",
                style={"display": "none"},
                children=[
                    # Tabs for different views/analyses
                    dbc.Tabs([
                        # Tab 1: Basic Data & Time Series
                        dbc.Tab(label="Basic Data", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Price Chart", className="mt-3"),
                                    dcc.Graph(id="price-chart")
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Data Statistics", className="mt-3"),
                                    html.Div(id="basic-stats")
                                ], width=6),
                                dbc.Col([
                                    html.H5("Top Correlations", className="mt-3"),
                                    html.Div(id="correlations-display")
                                ], width=6)
                            ])
                        ]),
                        
                        # Tab 2: Cluster Analysis
                        dbc.Tab(label="Cluster Analysis", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Clustering Options", className="mt-3"),
                                    html.Label("Number of Clusters"),
                                    dcc.Slider(
                                        id="cluster-slider",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=4,
                                        marks={i: str(i) for i in range(2, 11)},
                                        className="mb-3"
                                    ),
                                    html.Label("Features for Clustering"),
                                    dcc.Dropdown(
                                        id="cluster-features",
                                        multi=True,
                                        className="mb-3"
                                    ),
                                    dbc.Button("Run Clustering", id="run-clustering", color="primary"),
                                    dcc.Loading(
                                        id="loading-clusters",
                                        type="default",
                                        children=html.Div(id="loading-output-clusters"),
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Div(id="cluster-results", className="mt-3")
                                ], width=9)
                            ])
                        ]),
                        
                        # Tab 3: Anomaly Detection
                        dbc.Tab(label="Anomaly Detection", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Anomaly Detection Options", className="mt-3"),
                                    html.Label("Contamination Rate (% of outliers)"),
                                    dcc.Slider(
                                        id="anomaly-slider",
                                        min=0.01,
                                        max=0.2,
                                        step=0.01,
                                        value=0.05,
                                        marks={i/100: f"{i}%" for i in range(1, 21, 4)},
                                        className="mb-3"
                                    ),
                                    html.Label("Features for Anomaly Detection"),
                                    dcc.Dropdown(
                                        id="anomaly-features",
                                        multi=True,
                                        className="mb-3"
                                    ),
                                    dbc.Button("Detect Anomalies", id="run-anomaly", color="primary"),
                                    dcc.Loading(
                                        id="loading-anomalies",
                                        type="default",
                                        children=html.Div(id="loading-output-anomalies"),
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Div(id="anomaly-results", className="mt-3")
                                ], width=9)
                            ])
                        ]),
                        
                        # Tab 4: PCA
                        dbc.Tab(label="Principal Components", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H5("PCA Options", className="mt-3"),
                                    html.Label("Number of Components"),
                                    dcc.Slider(
                                        id="pca-slider",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=3,
                                        marks={i: str(i) for i in range(2, 11)},
                                        className="mb-3"
                                    ),
                                    html.Label("Features for PCA"),
                                    dcc.Dropdown(
                                        id="pca-features",
                                        multi=True,
                                        className="mb-3"
                                    ),
                                    dbc.Button("Run PCA", id="run-pca", color="primary"),
                                    dcc.Loading(
                                        id="loading-pca",
                                        type="default",
                                        children=html.Div(id="loading-output-pca"),
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Div(id="pca-results", className="mt-3")
                                ], width=9)
                            ])
                        ]),
                    ]),
                ]
            ),
            
        ], width=9)
    ])
], fluid=True)

# Add callbacks for the advanced analytics features
@app.callback(
    [Output("cluster-results", "children"),
     Output("loading-output-clusters", "children")],
    [Input("run-clustering", "n_clicks")],
    [State("cluster-features", "value"),
     State("cluster-slider", "value")],
    prevent_initial_call=True
)
def run_clustering(n_clicks, selected_features, n_clusters):
    global df_store
    
    if not selected_features or len(selected_features) < 2:
        return html.Div([
            html.P("Please select at least 2 features for clustering", className="text-danger")
        ]), None
    
    # Prepare data for clustering
    cluster_data = df_store[selected_features].copy()
    cluster_data = cluster_data.dropna()
    
    if len(cluster_data) < n_clusters:
        return html.Div([
            html.P("Not enough data points for clustering after removing NaN values", className="text-danger")
        ]), None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels back to the data
    cluster_data['Cluster'] = cluster_labels
    
    # Create results display
    results = []
    
    # 1. Create a scatter plot using the first two features
    if len(selected_features) >= 2:
        scatter_fig = px.scatter(
            cluster_data, 
            x=selected_features[0], 
            y=selected_features[1],
            color='Cluster',
            title=f"Cluster Visualization ({selected_features[0]} vs {selected_features[1]})",
            color_continuous_scale=px.colors.qualitative.Bold
        )
        
        results.append(html.Div([
            html.H5("Cluster Visualization"),
            dcc.Graph(figure=scatter_fig)
        ]))
    
    # 2. Create a time series plot showing clusters
    if Config.PRIMARY_PAIR in df_store.columns:
        # Create a combined dataframe with original data and clusters
        time_cluster_data = df_store[[Config.PRIMARY_PAIR]].copy()
        time_cluster_data = time_cluster_data.join(
            pd.DataFrame({'Cluster': cluster_labels}, index=cluster_data.index)
        )
        
        # Create the figure
        time_fig = go.Figure()
        
        # Add price trace
        time_fig.add_trace(go.Scatter(
            x=time_cluster_data.index,
            y=time_cluster_data[Config.PRIMARY_PAIR],
            mode='lines',
            name=Config.PRIMARY_PAIR,
            line=dict(color='lightgrey')
        ))
        
        # Add colored bands for clusters
        for cluster in range(n_clusters):
            cluster_periods = time_cluster_data[time_cluster_data['Cluster'] == cluster]
            if not cluster_periods.empty:
                # Create bands by adding filled areas
                for i in range(len(cluster_periods) - 1):
                    if i == 0 or cluster_periods.index[i] != cluster_periods.index[i-1] + pd.Timedelta(days=1):
                        start_idx = cluster_periods.index[i]
                        
                        # Find end of continuous period
                        j = i
                        while j < len(cluster_periods) - 1 and cluster_periods.index[j+1] == cluster_periods.index[j] + pd.Timedelta(days=1):
                            j += 1
                        end_idx = cluster_periods.index[j]
                        
                        # Add a rectangle shape for this period
                        time_fig.add_shape(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=start_idx,
                            x1=end_idx,
                            y0=0,
                            y1=1,
                            fillcolor=px.colors.qualitative.Bold[cluster % len(px.colors.qualitative.Bold)],
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )
        
        time_fig.update_layout(
            title=f"{Config.PRIMARY_PAIR} with Cluster Regimes",
            xaxis_title="Date",
            yaxis_title="Rate",
            showlegend=True
        )
        
        results.append(html.Div([
            html.H5("Market Regimes over Time"),
            dcc.Graph(figure=time_fig)
        ]))
    
    # 3. Create a table with cluster statistics
    stats_data = []
    for cluster in range(n_clusters):
        cluster_points = cluster_data[cluster_data['Cluster'] == cluster]
        
        # Calculate statistics for each feature
        stats_row = {
            'Cluster': f"Cluster {cluster}",
            'Size': len(cluster_points),
            'Percentage': f"{len(cluster_points) / len(cluster_data) * 100:.1f}%"
        }
        
        # Add mean values for each feature
        for feature in selected_features:
            stats_row[f"{feature} (mean)"] = f"{cluster_points[feature].mean():.4f}"
        
        stats_data.append(stats_row)
    
    stats_df = pd.DataFrame(stats_data)
    stats_table = dbc.Table.from_dataframe(
        stats_df,
        striped=True,
        bordered=True,
        hover=True
    )
    
    results.append(html.Div([
        html.H5("Cluster Statistics"),
        stats_table
    ]))
    
    return html.Div(results), None

@app.callback(
    [Output("anomaly-results", "children"),
     Output("loading-output-anomalies", "children")],
    [Input("run-anomaly", "n_clicks")],
    [State("anomaly-features", "value"),
     State("anomaly-slider", "value")],
    prevent_initial_call=True
)
def run_anomaly_detection(n_clicks, selected_features, contamination):
    global df_store
    
    if not selected_features or len(selected_features) < 1:
        return html.Div([
            html.P("Please select at least 1 feature for anomaly detection", className="text-danger")
        ]), None
    
    # Prepare data for anomaly detection
    anomaly_data = df_store[selected_features].copy()
    anomaly_data = anomaly_data.dropna()
    
    if len(anomaly_data) < 10:  # Arbitrary minimum number of samples
        return html.Div([
            html.P("Not enough data points for anomaly detection after removing NaN values", className="text-danger")
        ]), None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(anomaly_data)
    
    # Apply Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    # Predict anomalies (-1 for outliers, 1 for inliers)
    anomaly_labels = isolation_forest.fit_predict(scaled_data)
    
    # Convert to binary (1 for anomalies, 0 for normal)
    is_anomaly = np.where(anomaly_labels == -1, 1, 0)
    
    # Add anomaly labels back to the data
    anomaly_data['Anomaly'] = is_anomaly
    
    # Create results display
    results = []
    
    # 1. Create time series with anomalies highlighted
    if Config.PRIMARY_PAIR in df_store.columns:
        # Add price and anomaly data together
        time_anomaly_data = df_store[[Config.PRIMARY_PAIR]].copy()
        time_anomaly_data = time_anomaly_data.join(
            pd.DataFrame({'Anomaly': is_anomaly}, index=anomaly_data.index)
        )
        
        # Create scatter plot highlighting anomalies
        fig = go.Figure()
        
        # Add line trace for the currency pair
        fig.add_trace(go.Scatter(
            x=time_anomaly_data.index,
            y=time_anomaly_data[Config.PRIMARY_PAIR],
            mode='lines',
            name=Config.PRIMARY_PAIR,
            line=dict(color='royalblue')
        ))
        
        # Add scatter points for anomalies
        anomalies = time_anomaly_data[time_anomaly_data['Anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies[Config.PRIMARY_PAIR],
            mode='markers',
            name='Anomalies',
            marker=dict(
                color='red',
                size=8,
                symbol='circle'
            )
        ))
        
        fig.update_layout(
            title=f"Anomaly Detection for {Config.PRIMARY_PAIR}",
            xaxis_title="Date",
            yaxis_title="Rate",
            legend_title="Legend",
            template="plotly_white"
        )
        
        results.append(html.Div([
            html.H5("Anomalies over Time"),
            dcc.Graph(figure=fig)
        ]))
    
    # 2. Feature contribution analysis
    # For each feature, compare the distribution between normal and anomaly points
    feature_comparisons = []
    
    for feature in selected_features:
        normal_values = anomaly_data[anomaly_data['Anomaly'] == 0][feature]
        anomaly_values = anomaly_data[anomaly_data['Anomaly'] == 1][feature]
        
        # Skip if we don't have enough anomalies
        if len(anomaly_values) < 2:
            continue
        
        # Create box plots comparing distributions
        box_fig = go.Figure()
        
        box_fig.add_trace(go.Box(
            y=normal_values,
            name='Normal',
            boxmean=True,
            marker_color='royalblue'
        ))
        
        box_fig.add_trace(go.Box(
            y=anomaly_values,
            name='Anomaly',
            boxmean=True,
            marker_color='red'
        ))
        
        box_fig.update_layout(
            title=f"Distribution of {feature} (Normal vs Anomaly)",
            yaxis_title=feature,
            showlegend=True
        )
        
        feature_comparisons.append(dcc.Graph(figure=box_fig))
    
    # Create a grid layout for feature comparisons
    feature_grid = []
    for i in range(0, len(feature_comparisons), 2):
        row_items = feature_comparisons[i:i+2]
        if len(row_items) == 2:
            feature_grid.append(
                dbc.Row([
                    dbc.Col(row_items[0], width=6),
                    dbc.Col(row_items[1], width=6)
                ])
            )
        else:
            feature_grid.append(
                dbc.Row([
                    dbc.Col(row_items[0], width=6)
                ])
            )
    
    if feature_grid:
        results.append(html.Div([
            html.H5("Feature Distributions (Normal vs Anomaly)"),
            html.Div(feature_grid)
        ]))
    
    # 3. Anomaly summary
    summary = html.Div([
        html.H5("Anomaly Detection Summary"),
        html.P([
            f"Total points analyzed: {len(anomaly_data)}",
            html.Br(),
            f"Anomalies detected: {sum(is_anomaly)} ({sum(is_anomaly)/len(anomaly_data)*100:.2f}%)",
            html.Br(),
            f"Contamination rate used: {contamination*100:.1f}%"
        ])
    ])
    
    results.append(summary)
    
    return html.Div(results), None

@app.callback(
    [Output("pca-results", "children"),
     Output("loading-output-pca", "children")],
    [Input("run-pca", "n_clicks")],
    [State("pca-features", "value"),
     State("pca-slider", "value")],
    prevent_initial_call=True
)
def run_pca_analysis(n_clicks, selected_features, n_components):
    global df_store
    
    if not selected_features or len(selected_features) < n_components:
        return html.Div([
            html.P(f"Please select at least {n_components} features for PCA", className="text-danger")
        ]), None
    
    # Prepare data for PCA
    pca_data = df_store[selected_features].copy()
    pca_data = pca_data.dropna()
    
    if len(pca_data) < 10:  # Arbitrary minimum number of samples
        return html.Div([
            html.P("Not enough data points for PCA after removing NaN values", className="text-danger")
        ]), None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create DataFrame with principal components
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=pca_data.index
    )
    
    # Create results display
    results = []
    
    # 1. Explained variance plot
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)
    
    variance_fig = go.Figure()
    
    # Add bar chart for individual variance
    variance_fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(n_components)],
        y=explained_variance,
        name='Individual Explained Variance (%)'
    ))
    
    # Add line chart for cumulative variance
    variance_fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(n_components)],
        y=cumulative_variance,
        name='Cumulative Explained Variance (%)',
        mode='lines+markers',
        line=dict(color='red')
    ))
    
    variance_fig.update_layout(
        title='Explained Variance by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance (%)',
        yaxis=dict(range=[0, 100]),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        template='plotly_white'
    )
    
    results.append(html.Div([
        html.H5("Explained Variance"),
        dcc.Graph(figure=variance_fig)
    ]))
    
    # 2. Feature importance (loadings) visualization
    loadings = pca.components_
    loadings_df = pd.DataFrame(
        loadings,
        columns=selected_features,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Heatmap of feature loadings
    heatmap_fig = px.imshow(
        loadings_df,
        labels=dict(x="Feature", y="Principal Component", color="Loading"),
        x=selected_features,
        y=[f'PC{i+1}' for i in range(n_components)],
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Feature Contributions to Principal Components"
    )
    
    heatmap_fig.update_layout(
        xaxis=dict(tickangle=45),
        template='plotly_white'
    )
    
    results.append(html.Div([
        html.H5("Feature Loadings"),
        dcc.Graph(figure=heatmap_fig)
    ]))
    
    # 3. If the currency pair is available, add a time series with PC1
    if Config.PRIMARY_PAIR in df_store.columns:
        # Join the first principal component with the currency data
        pc_time_df = df_store[[Config.PRIMARY_PAIR]].copy()
        pc_time_df = pc_time_df.join(pc_df[['PC1']])
        
        # Normalize both series for comparison
        pc_time_df['Normalized_Rate'] = (pc_time_df[Config.PRIMARY_PAIR] - pc_time_df[Config.PRIMARY_PAIR].mean()) / pc_time_df[Config.PRIMARY_PAIR].std()
        pc_time_df['Normalized_PC1'] = (pc_time_df['PC1'] - pc_time_df['PC1'].mean()) / pc_time_df['PC1'].std()
        
        # Create time series plot
        pc_time_fig = go.Figure()
        
        pc_time_fig.add_trace(go.Scatter(
            x=pc_time_df.index,
            y=pc_time_df['Normalized_Rate'],
            mode='lines',
            name=f'{Config.PRIMARY_PAIR} (normalized)'
        ))
        
        pc_time_fig.add_trace(go.Scatter(
            x=pc_time_df.index,
            y=pc_time_df['Normalized_PC1'],
            mode='lines',
            name='PC1 (normalized)',
            line=dict(dash='dash')
        ))
        
        pc_time_fig.update_layout(
            title=f"Comparing {Config.PRIMARY_PAIR} with Principal Component 1",
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            legend_title="Legend",
            template="plotly_white"
        )
        
        results.append(html.Div([
            html.H5("Principal Component Time Series"),
            dcc.Graph(figure=pc_time_fig)
        ]))
    
    # 4. Table of top contributing features to each PC
    top_contributors = []
    
    for i in range(n_components):
        pc_loadings = loadings_df.iloc[i].abs()
        top_features = pc_loadings.sort_values(ascending=False).head(5)
        
        contributors = {
            'Principal Component': f'PC{i+1}',
            'Explained Variance': f'{explained_variance[i]:.2f}%',
            'Top Features': ', '.join([f"{feat} ({loadings_df.iloc[i][feat]:.4f})" for feat in top_features.index])
        }
        
        top_contributors.append(contributors)
    
    contributors_df = pd.DataFrame(top_contributors)
    
    contributors_table = dbc.Table.from_dataframe(
        contributors_df,
        striped=True,
        bordered=True,
        hover=True
    )
    
    results.append(html.Div([
        html.H5("Top Contributing Features"),
        contributors_table,
        html.P("Note: Values in parentheses are the loadings (weights) for each feature.")
    ]))
    
    return html.Div(results), None

if __name__ == '__main__':
    app.run(debug=True)
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Store the dataframe in a global variable for access across callbacks
# In a production app, use dcc.Store instead
# Store the dataframe globally for access across callbacks
# In a production app, use dcc.Store instead
df_store = None

# Callback for the download button
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    global df_store
    if df_store is not None:
        return dcc.send_data_frame(df_store.to_csv, f"{Config.PRIMARY_PAIR}_data.csv", index=True)

# Main callback to collect data
@app.callback(
    [
        Output("loading-output-data", "children"),
        Output("results-container", "style"),
        Output("instructions-card", "style"),
        Output("price-chart", "figure"),
        Output("basic-stats", "children"),
        Output("correlations-display", "children"),
        Output("cluster-features", "options"),
        Output("cluster-features", "value"),
        Output("anomaly-features", "options"),
        Output("anomaly-features", "value"),
        Output("pca-features", "options"),
        Output("pca-features", "value"),
        Output("btn-download", "style"),
    ],
    Input("collect-button", "n_clicks"),
    [
        State("currency-pair-dropdown", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("advanced-options", "value"),
        State("fred-key", "value"),
        State("polygon-key", "value")
    ],
    prevent_initial_call=True
)
def collect_data(n_clicks, selected_pair, start_date, end_date, advanced_options, fred_key, polygon_key):
    global df_store
    
    if n_clicks is None:
        return None, {"display": "none"}, {}, {}, None, None, [], [], [], [], [], [], {"display": "none"}
    
    # Set API keys if provided
    if fred_key:
        os.environ["FRED_API_KEY"] = fred_key
    if polygon_key:
        os.environ["POLYGON_API_KEY"] = polygon_key
    
    # Update Config with the selected options
    Config.PRIMARY_PAIR = selected_pair
    Config.START_DATE = datetime.strptime(start_date, "%Y-%m-%d")
    Config.END_DATE = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Process advanced options
    if "all_currencies" not in advanced_options:
        base_currency = selected_pair[:3]
        quote_currency = selected_pair[3:]
        Config.MAJOR_CURRENCIES = [base_currency, quote_currency]
    
    if "commodities" not in advanced_options:
        Config.COMMODITIES = []
    
    if "indices" not in advanced_options:
        Config.STOCK_INDICES = {country: [] for country in Config.STOCK_INDICES}
    
    if "special" not in advanced_options:
        Config.SPECIAL_INDICATORS = []
    
    # Run data collection
    df_store = main()
    
    # Create price chart
    fig = go.Figure()
    
    if selected_pair in df_store.columns:
        fig.add_trace(go.Scatter(
            x=df_store.index, 
            y=df_store[selected_pair],
            mode='lines',
            name=selected_pair
        ))
        
        # Add moving averages if present
        for period in [20, 50, 200]:
            ma_col = f"{period} MA"
            if ma_col in df_store.columns:
                fig.add_trace(go.Scatter(
                    x=df_store.index,
                    y=df_store[ma_col],
                    mode='lines',
                    name=f"{period}-day MA",
                    line=dict(width=1, dash='dot')
                ))
    else:
        fig.add_annotation(
            text="Price data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig.update_layout(
        title=f"{selected_pair} Exchange Rate",
        xaxis_title="Date",
        yaxis_title="Rate",
        legend_title="Legend",
        template="plotly_white"
    )
    
    # Create statistics display
    stats_table = dbc.Table.from_dataframe(
        pd.DataFrame({
            'Metric': ['Start Date', 'End Date', 'Number of Indicators', 'Data Completeness',
                      'Price Min', 'Price Max', 'Price Mean', 'Price Std Dev'],
            'Value': [
                df_store.index.min().strftime('%Y-%m-%d'),
                df_store.index.max().strftime('%Y-%m-%d'),
                len(df_store.columns),
                f"{df_store.notna().mean().mean() * 100:.2f}%",
                f"{df_store[selected_pair].min():.4f}" if selected_pair in df_store.columns else "N/A",
                f"{df_store[selected_pair].max():.4f}" if selected_pair in df_store.columns else "N/A",
                f"{df_store[selected_pair].mean():.4f}" if selected_pair in df_store.columns else "N/A",
                f"{df_store[selected_pair].std():.4f}" if selected_pair in df_store.columns else "N/A"
            ]
        }),
        striped=True,
        bordered=True,
        hover=True
    )
    
    # Create correlations display
    if selected_pair in df_store.columns:
        correlations = df_store.corr()[selected_pair].sort_values(ascending=False)
        
        # Filter out NaN and self-correlation
        correlations = correlations.dropna()
        correlations = correlations[correlations.index != selected_pair]
        
        # Correlation chart
        corr_chart = dcc.Graph(
            figure=px.bar(
                x=correlations.index[:10],
                y=correlations.values[:10],
                title=f"Top Correlations with {selected_pair}",
                labels={'x': 'Indicator', 'y': 'Correlation'},
                color=correlations.values[:10],
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
        )
        
        correlations_display = html.Div([
            corr_chart
        ])
    else:
        correlations_display = html.Div([
            html.P(f"Currency pair {selected_pair} not found in the data.")
        ])
    
    # Create feature selection options
    # Filter numeric columns for dropdown options
    numeric_cols = df_store.select_dtypes(include=[np.number]).columns.tolist()
    feature_options = [{'label': col, 'value': col} for col in numeric_cols]
    
    # Default selected features (top correlated features if available)
    if selected_pair in df_store.columns and len(correlations) >= 5:
        default_features = correlations.index[:5].tolist()
    else:
        default_features = numeric_cols[:min(5, len(numeric_cols))]
    
    return None, {"display": "block"}, {"display": "none"}, fig, stats_table, correlations_display, \
           feature_options, default_features, feature_options, default_features, feature_options, default_features, \
           {"display": "block"}
