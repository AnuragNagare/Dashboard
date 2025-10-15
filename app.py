import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import clickhouse_connect
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("D:\\Cyber Anamoly detection\\AI Research Data - IDS Rules\\DATASET - Training\\Telemetry models\\Web app\\.env")

# ClickHouse Configuration
CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST')
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER')
CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')
DATABASE_NAME = os.getenv('DATABASE_NAME')

# Table names
TABLES = {
    'Generative Model': 'generative_model_results',
    'Sequence Model': 'sequence_model_results',
    'Partition Model': 'partition_model_results',
    'Boundary Model': 'boundary_model_results',
    'Enhanced Ensemble': 'enhanced_ensemble_strategy_results'
}

# Connect to ClickHouse
def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        secure=True
    )

# Data fetching functions
def fetch_table_data(table_name, limit=1000):
    """Fetch data from specific table"""
    client = get_clickhouse_client()
    query = f"SELECT * FROM {DATABASE_NAME}.{table_name} ORDER BY timestamp DESC LIMIT {limit}"
    result = client.query(query)
    df = pd.DataFrame(result.result_set, columns=result.column_names)
    return df

def fetch_summary_stats():
    """Fetch summary statistics from all tables"""
    client = get_clickhouse_client()
    stats = {}
    
    for model_name, table_name in TABLES.items():
        try:
            # Total records
            total_query = f"SELECT count() FROM {DATABASE_NAME}.{table_name}"
            total = client.query(total_query).result_set[0][0]
            
            # Anomaly count
            if table_name == 'enhanced_ensemble_strategy_results':
                anomaly_query = f"SELECT count() FROM {DATABASE_NAME}.{table_name} WHERE final_severity != 'NORMAL'"
                anomalies = client.query(anomaly_query).result_set[0][0]
            else:
                anomaly_query = f"SELECT count() FROM {DATABASE_NAME}.{table_name} WHERE is_anomaly = 1"
                anomalies = client.query(anomaly_query).result_set[0][0]
            
            # Average score
            if table_name == 'enhanced_ensemble_strategy_results':
                avg_query = f"SELECT avg(base_score) FROM {DATABASE_NAME}.{table_name}"
            else:
                avg_query = f"SELECT avg(anomaly_score) FROM {DATABASE_NAME}.{table_name}"
            avg_score = client.query(avg_query).result_set[0][0]
            
            stats[model_name] = {
                'total': total,
                'anomalies': anomalies,
                'detection_rate': (anomalies / total * 100) if total > 0 else 0,
                'avg_score': float(avg_score) if avg_score else 0
            }
        except Exception as e:
            stats[model_name] = {
                'total': 0,
                'anomalies': 0,
                'detection_rate': 0,
                'avg_score': 0,
                'error': str(e)
            }
    
    return stats

def fetch_ensemble_severity_breakdown():
    """Fetch severity breakdown from ensemble results"""
    client = get_clickhouse_client()
    query = f"""
    SELECT final_severity, count() as count 
    FROM {DATABASE_NAME}.enhanced_ensemble_strategy_results 
    GROUP BY final_severity
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_set, columns=['severity', 'count'])
    return df

def fetch_timeline_data(hours=24):
    """Fetch anomaly detection timeline"""
    client = get_clickhouse_client()
    timeline_data = {}
    
    for model_name, table_name in TABLES.items():
        try:
            if table_name == 'enhanced_ensemble_strategy_results':
                query = f"""
                SELECT toStartOfHour(timestamp) as hour, count() as anomalies
                FROM {DATABASE_NAME}.{table_name}
                WHERE final_severity != 'NORMAL'
                AND timestamp >= now() - INTERVAL {hours} HOUR
                GROUP BY hour
                ORDER BY hour
                """
            else:
                query = f"""
                SELECT toStartOfHour(timestamp) as hour, count() as anomalies
                FROM {DATABASE_NAME}.{table_name}
                WHERE is_anomaly = 1
                AND timestamp >= now() - INTERVAL {hours} HOUR
                GROUP BY hour
                ORDER BY hour
                """
            
            result = client.query(query)
            df = pd.DataFrame(result.result_set, columns=['hour', 'anomalies'])
            timeline_data[model_name] = df
        except Exception as e:
            timeline_data[model_name] = pd.DataFrame(columns=['hour', 'anomalies'])
    
    return timeline_data

def fetch_top_anomalies(table_name, limit=10):
    """Fetch top anomalies by score"""
    client = get_clickhouse_client()
    
    if table_name == 'enhanced_ensemble_strategy_results':
        query = f"""
        SELECT 
            id, timestamp, sensor_id, temperature, humidity,
            final_severity, confidence_score, base_score,
            full_analysis_report
        FROM {DATABASE_NAME}.{table_name}
        WHERE final_severity != 'NORMAL'
        ORDER BY base_score DESC
        LIMIT {limit}
        """
    else:
        query = f"""
        SELECT 
            id, timestamp, sensor_id, temperature, humidity,
            anomaly_score, is_anomaly, individual_explanation,
            full_analysis_report
        FROM {DATABASE_NAME}.{table_name}
        WHERE is_anomaly = 1
        ORDER BY anomaly_score DESC
        LIMIT {limit}
        """
    
    result = client.query(query)
    df = pd.DataFrame(result.result_set, columns=result.column_names)
    return df

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🔍 Multi-Model Anomaly Detection Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Real-time monitoring of 5 ML models with ClickHouse data storage",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '10px'}),
    
    # Refresh button and timestamp
    html.Div([
        html.Button('🔄 Refresh Data', id='refresh-button', n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                          'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer',
                          'fontSize': '16px', 'marginRight': '20px'}),
        html.Span(id='last-updated', style={'color': '#7f8c8d', 'fontSize': '14px'})
    ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
    
    # Summary cards
    html.Div(id='summary-cards', style={'marginBottom': '30px'}),
    
    # Tabs for different views
    dcc.Tabs(id='main-tabs', value='overview', children=[
        dcc.Tab(label='📊 Overview', value='overview', style={'fontSize': '16px'}),
        dcc.Tab(label='📈 Model Comparison', value='comparison', style={'fontSize': '16px'}),
        dcc.Tab(label='⏰ Timeline', value='timeline', style={'fontSize': '16px'}),
        dcc.Tab(label='🚨 Top Anomalies', value='anomalies', style={'fontSize': '16px'}),
        dcc.Tab(label='🎯 Detailed Analysis', value='details', style={'fontSize': '16px'}),
    ], style={'marginBottom': '20px'}),
    
    # Tab content
    html.Div(id='tab-content')
    
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa'})

# Callbacks
@app.callback(
    [Output('summary-cards', 'children'),
     Output('last-updated', 'children')],
    [Input('refresh-button', 'n_clicks')]
)
def update_summary_cards(n_clicks):
    """Update summary statistics cards"""
    stats = fetch_summary_stats()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cards = []
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, (model_name, data) in enumerate(stats.items()):
        card = html.Div([
            html.H3(model_name, style={'color': 'white', 'marginBottom': '10px', 'fontSize': '18px'}),
            html.Div([
                html.Div([
                    html.P('Total Records', style={'margin': '0', 'fontSize': '12px', 'color': '#ecf0f1'}),
                    html.H2(f"{data['total']:,}", style={'margin': '5px 0', 'color': 'white'})
                ], style={'flex': '1'}),
                html.Div([
                    html.P('Anomalies', style={'margin': '0', 'fontSize': '12px', 'color': '#ecf0f1'}),
                    html.H2(f"{data['anomalies']:,}", style={'margin': '5px 0', 'color': 'white'})
                ], style={'flex': '1'}),
                html.Div([
                    html.P('Detection Rate', style={'margin': '0', 'fontSize': '12px', 'color': '#ecf0f1'}),
                    html.H2(f"{data['detection_rate']:.1f}%", style={'margin': '5px 0', 'color': 'white'})
                ], style={'flex': '1'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'})
        ], style={
            'backgroundColor': colors[i],
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'flex': '1',
            'margin': '0 10px',
            'minWidth': '200px'
        })
        cards.append(card)
    
    cards_container = html.Div(cards, style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '10px'
    })
    
    return cards_container, f"Last updated: {current_time}"

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def render_tab_content(tab, n_clicks):
    """Render content based on selected tab"""
    
    if tab == 'overview':
        return render_overview()
    elif tab == 'comparison':
        return render_comparison()
    elif tab == 'timeline':
        return render_timeline()
    elif tab == 'anomalies':
        return render_top_anomalies()
    elif tab == 'details':
        return render_detailed_analysis()

def render_overview():
    """Render overview dashboard"""
    stats = fetch_summary_stats()
    severity_df = fetch_ensemble_severity_breakdown()
    
    # Algorithm Information Section
    algorithm_info = html.Div([
        html.H2("🤖 Algorithm Information", style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div([
            # AAE Anomaly Detector Card
            html.Div([
                html.H3("🧠 AAE Anomaly Detector", style={'color': 'white', 'marginBottom': '10px'}),
                html.P("Generative Model", style={'color': '#ecf0f1', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Strong("Type: "), "Adversarial Autoencoder",
                        html.Br(), html.Strong("Features: "), "4 (temp, humidity, sensor_id, data_type)",
                        html.Br(), html.Strong("Method: "), "Reconstruction-based detection",
                        html.Br(), html.Strong("Threshold: "), "0.7 (adaptive)",
                        html.Br(), html.Strong("Reliability: "), "95%",
                        html.Br(), html.Strong("Focus: "), "Security & Generative patterns"
                    ], style={'fontSize': '12px', 'lineHeight': '1.4'})
                ])
            ], style={
                'backgroundColor': '#3498db',
                'padding': '15px',
                'borderRadius': '10px',
                'color': 'white',
                'flex': '1',
                'margin': '5px',
                'minHeight': '200px'
            }),
            
            # LSTM Autoencoder Card
            html.Div([
                html.H3("🔄 LSTM Autoencoder", style={'color': 'white', 'marginBottom': '10px'}),
                html.P("Sequence Model", style={'color': '#ecf0f1', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Strong("Type: "), "Long Short-Term Memory",
                        html.Br(), html.Strong("Features: "), "2D (batch_size, features)",
                        html.Br(), html.Strong("Method: "), "Temporal pattern detection",
                        html.Br(), html.Strong("Threshold: "), "0.75 (adaptive)",
                        html.Br(), html.Strong("Reliability: "), "90%",
                        html.Br(), html.Strong("Focus: "), "Security & Operational"
                    ], style={'fontSize': '12px', 'lineHeight': '1.4'})
                ])
            ], style={
                'backgroundColor': '#2ecc71',
                'padding': '15px',
                'borderRadius': '10px',
                'color': 'white',
                'flex': '1',
                'margin': '5px',
                'minHeight': '200px'
            }),
            
            # Isolation Forest Card
            html.Div([
                html.H3("🌲 Isolation Forest", style={'color': 'white', 'marginBottom': '10px'}),
                html.P("Partition Model", style={'color': '#ecf0f1', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Strong("Type: "), "Tree-based Ensemble",
                        html.Br(), html.Strong("Features: "), "9 (numerical with padding)",
                        html.Br(), html.Strong("Method: "), "Outlier isolation",
                        html.Br(), html.Strong("Threshold: "), "0.6 (adaptive)",
                        html.Br(), html.Strong("Reliability: "), "88%",
                        html.Br(), html.Strong("Focus: "), "Operational patterns"
                    ], style={'fontSize': '12px', 'lineHeight': '1.4'})
                ])
            ], style={
                'backgroundColor': '#e74c3c',
                'padding': '15px',
                'borderRadius': '10px',
                'color': 'white',
                'flex': '1',
                'margin': '5px',
                'minHeight': '200px'
            }),
            
            # One-Class SVM Card
            html.Div([
                html.H3("🎯 One-Class SVM", style={'color': 'white', 'marginBottom': '10px'}),
                html.P("Boundary Model", style={'color': '#ecf0f1', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Strong("Type: "), "Support Vector Machine",
                        html.Br(), html.Strong("Features: "), "9 (numerical with padding)",
                        html.Br(), html.Strong("Method: "), "Boundary-based detection",
                        html.Br(), html.Strong("Threshold: "), "0.65 (adaptive)",
                        html.Br(), html.Strong("Reliability: "), "92%",
                        html.Br(), html.Strong("Focus: "), "Security patterns"
                    ], style={'fontSize': '12px', 'lineHeight': '1.4'})
                ])
            ], style={
                'backgroundColor': '#f39c12',
                'padding': '15px',
                'borderRadius': '10px',
                'color': 'white',
                'flex': '1',
                'margin': '5px',
                'minHeight': '200px'
            }),
            
            # Enhanced Decision Engine Card
            html.Div([
                html.H3("⚡ Enhanced Decision Engine", style={'color': 'white', 'marginBottom': '10px'}),
                html.P("Consensus Algorithm", style={'color': '#ecf0f1', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Strong("Type: "), "Multi-Model Consensus",
                        html.Br(), html.Strong("Features: "), "Weighted voting system",
                        html.Br(), html.Strong("Method: "), "Adaptive thresholds + Attack patterns",
                        html.Br(), html.Strong("Threshold: "), "Dynamic (85th percentile)",
                        html.Br(), html.Strong("Reliability: "), "Multi-factor confidence",
                        html.Br(), html.Strong("Focus: "), "Security & Operational fusion"
                    ], style={'fontSize': '12px', 'lineHeight': '1.4'})
                ])
            ], style={
                'backgroundColor': '#9b59b6',
                'padding': '15px',
                'borderRadius': '10px',
                'color': 'white',
                'flex': '1',
                'margin': '5px',
                'minHeight': '200px'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '30px'}),
        
        # Decision Engine Features
        html.Div([
            html.H3("🎯 Enhanced Decision Engine Features", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H4("🔍 Attack Pattern Recognition", style={'color': '#e74c3c', 'marginBottom': '8px'}),
                    html.Ul([
                        html.Li("Injection Attacks: Generative + Boundary models"),
                        html.Li("DoS Attacks: Sequence + Partition models"),
                        html.Li("Intrusion Detection: Boundary + Generative models"),
                        html.Li("Data Poisoning: Generative + Sequence models")
                    ], style={'fontSize': '14px', 'lineHeight': '1.6'})
                ], style={'flex': '1', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H4("📊 Multi-Tier Severity Classification", style={'color': '#f39c12', 'marginBottom': '8px'}),
                    html.Ul([
                        html.Li("CRITICAL_ATTACK: Immediate isolation required"),
                        html.Li("HIGH_SECURITY: Urgent investigation needed"),
                        html.Li("MEDIUM_OPERATIONAL: Close monitoring"),
                        html.Li("LOW_ANOMALY: Logging only"),
                        html.Li("NORMAL: No action required")
                    ], style={'fontSize': '14px', 'lineHeight': '1.6'})
                ], style={'flex': '1', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H4("🌡️ Domain-Specific Rules", style={'color': '#2ecc71', 'marginBottom': '8px'}),
                    html.Ul([
                        html.Li("Temperature: >60°C (critical), >45°C (warning)"),
                        html.Li("Humidity: >95% or <5% (extreme conditions)"),
                        html.Li("Impossible Conditions: High temp + high humidity"),
                        html.Li("Temporal Analysis: 50-record sliding window")
                    ], style={'fontSize': '14px', 'lineHeight': '1.6'})
                ], style={'flex': '1', 'margin': '10px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'})
        ], style={'marginBottom': '30px'})
    ])
    
    # Model comparison bar chart
    models = list(stats.keys())
    anomalies = [stats[m]['anomalies'] for m in models]
    totals = [stats[m]['total'] for m in models]
    
    fig1 = go.Figure(data=[
        go.Bar(name='Total Records', x=models, y=totals, marker_color='#3498db'),
        go.Bar(name='Anomalies', x=models, y=anomalies, marker_color='#e74c3c')
    ])
    fig1.update_layout(
        title='Model Detection Overview',
        barmode='group',
        xaxis_title='Model',
        yaxis_title='Count',
        height=400,
        template='plotly_white'
    )
    
    # Severity distribution pie chart
    if not severity_df.empty:
        fig2 = px.pie(
            severity_df,
            values='count',
            names='severity',
            title='Ensemble Severity Distribution',
            color='severity',
            color_discrete_map={
                'CRITICAL_ATTACK': '#c0392b',
                'CRITICAL_ANOMALY': '#e74c3c',
                'HIGH_SECURITY': '#e67e22',
                'HIGH_ANOMALY': '#f39c12',
                'MEDIUM_OPERATIONAL': '#f1c40f',
                'MEDIUM_SECURITY': '#f39c12',
                'LOW_ANOMALY': '#3498db',
                'NORMAL': '#2ecc71'
            },
            height=400
        )
    else:
        fig2 = go.Figure()
        fig2.add_annotation(text="No ensemble data available", showarrow=False)
    
    # Detection rate comparison
    detection_rates = [stats[m]['detection_rate'] for m in models]
    fig3 = go.Figure(data=[
        go.Bar(x=models, y=detection_rates, marker_color='#9b59b6', text=detection_rates,
               texttemplate='%{text:.1f}%', textposition='outside')
    ])
    fig3.update_layout(
        title='Detection Rate Comparison',
        xaxis_title='Model',
        yaxis_title='Detection Rate (%)',
        height=400,
        template='plotly_white'
    )
    
    return html.Div([
        algorithm_info,
        html.H2("📊 Performance Analytics", style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div([
            html.Div([dcc.Graph(figure=fig1)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig2)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
        dcc.Graph(figure=fig3)
    ])

def render_comparison():
    """Render model comparison view with clear, easy-to-understand charts"""
    stats = fetch_summary_stats()
    models = list(stats.keys())
    
    # Color scheme for consistency
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    model_colors = dict(zip(models, colors))
    
    # 1. Detection Rate Comparison (Horizontal Bar Chart)
    detection_rates = [stats[m]['detection_rate'] for m in models]
    fig1 = go.Figure(data=[
        go.Bar(
            y=models,
            x=detection_rates,
            orientation='h',
            marker_color=[model_colors[m] for m in models],
            text=[f"{rate:.1f}%" for rate in detection_rates],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Detection Rate: %{x:.2f}%<extra></extra>'
        )
    ])
    fig1.update_layout(
        title='Detection Rate Comparison',
        xaxis_title='Detection Rate (%)',
        yaxis_title='Model',
        height=400,
        width=600,  # Fixed width to prevent stretching
        template='plotly_white',
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)  # Add margins
    )
    
    # 2. Total Records vs Anomalies (Scatter Plot)
    fig2 = go.Figure()
    for i, model in enumerate(models):
        fig2.add_trace(go.Scatter(
            x=[stats[model]['total']],
            y=[stats[model]['anomalies']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=model_colors[model],
                opacity=0.8
            ),
            text=[model],
            textposition='top center',
            name=model,
            hovertemplate=f'<b>{model}</b><br>Total Records: {stats[model]["total"]:,}<br>Anomalies: {stats[model]["anomalies"]:,}<extra></extra>'
        ))
    
    fig2.update_layout(
        title='Total Records vs Anomalies Detected',
        xaxis_title='Total Records',
        yaxis_title='Anomalies Detected',
        height=400,
        width=600,  # Fixed width to prevent stretching
        template='plotly_white',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)  # Add margins
    )
    
    # 3. Average Score Comparison (Bar Chart)
    avg_scores = [stats[m]['avg_score'] for m in models]
    fig3 = go.Figure(data=[
        go.Bar(
            x=models,
            y=avg_scores,
            marker_color=[model_colors[m] for m in models],
            text=[f"{score:.3f}" for score in avg_scores],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Average Score: %{y:.4f}<extra></extra>'
        )
    ])
    fig3.update_layout(
        title='Average Anomaly Score by Model',
        xaxis_title='Model',
        yaxis_title='Average Score',
        height=400,
        width=800,  # Fixed width to prevent stretching
        template='plotly_white',
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)  # Add margins
    )
    
    # 4. Model Performance Summary (Gauge Charts)
    gauge_figures = []
    for i, model in enumerate(models):
        detection_rate = stats[model]['detection_rate']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=detection_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{model}<br>Detection Rate"},
            delta={'reference': 50},  # Reference line at 50%
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': model_colors[model]},
                'steps': [
                    {'range': [0, 25], 'color': '#ffebee'},
                    {'range': [25, 50], 'color': '#fff3e0'},
                    {'range': [50, 75], 'color': '#e8f5e8'},
                    {'range': [75, 100], 'color': '#e3f2fd'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        gauge_figures.append(fig_gauge)
    
    # Performance metrics table with better formatting
    table_data = []
    for model in models:
        table_data.append({
            'Model': model,
            'Total Records': f"{stats[model]['total']:,}",
            'Anomalies': f"{stats[model]['anomalies']:,}",
            'Detection Rate': f"{stats[model]['detection_rate']:.2f}%",
            'Avg Score': f"{stats[model]['avg_score']:.4f}",
            'Status': '🟢 High Performance' if stats[model]['detection_rate'] > 70 else 
                     '🟡 Medium Performance' if stats[model]['detection_rate'] > 40 else '🔴 Low Performance'
        })
    
    table = dash_table.DataTable(
        data=table_data,
        columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '14px'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '16px'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
            {'if': {'filter_query': '{Status} contains "High Performance"'}, 'backgroundColor': '#d4edda'},
            {'if': {'filter_query': '{Status} contains "Medium Performance"'}, 'backgroundColor': '#fff3cd'},
            {'if': {'filter_query': '{Status} contains "Low Performance"'}, 'backgroundColor': '#f8d7da'}
        ]
    )
    
    return html.Div([
        # Main comparison charts
        html.Div([
            html.Div([
                dcc.Graph(figure=fig1, style={'width': '100%', 'height': '400px'}),
                html.P("📊 Higher detection rates indicate better anomaly detection performance", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic', 'marginTop': '10px'})
            ], style={'flex': '1', 'margin': '10px', 'minWidth': '300px'}),
            html.Div([
                dcc.Graph(figure=fig2, style={'width': '100%', 'height': '400px'}),
                html.P("🎯 Shows relationship between data volume and anomaly detection", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic', 'marginTop': '10px'})
            ], style={'flex': '1', 'margin': '10px', 'minWidth': '300px'})
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'flexWrap': 'wrap'}),
        
        # Average score chart
        html.Div([
            dcc.Graph(figure=fig3, style={'width': '100%', 'height': '400px'}),
            html.P("📈 Average scores indicate model confidence in anomaly detection", 
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic', 'marginTop': '10px'})
        ], style={'marginBottom': '30px', 'display': 'flex', 'justifyContent': 'center'}),
        
        # Gauge charts for individual model performance
        html.H3('🎯 Individual Model Performance Gauges', style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div([
            html.Div([dcc.Graph(figure=gauge, style={'width': '100%', 'height': '250px'})], style={'flex': '1', 'margin': '5px', 'minWidth': '200px'}) 
            for gauge in gauge_figures
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '30px', 'justifyContent': 'center'}),
        
        # Performance metrics table
        html.H3('📋 Detailed Performance Metrics', style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
        table,
        
        # Summary insights
        html.Div([
            html.H4('💡 Key Insights', style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Ul([
                html.Li(f"Best Detection Rate: {max(detection_rates):.1f}% ({models[detection_rates.index(max(detection_rates))]})"),
                html.Li(f"Most Data Processed: {max([stats[m]['total'] for m in models]):,} records"),
                html.Li(f"Highest Average Score: {max(avg_scores):.4f}"),
                html.Li("Models with >70% detection rate are considered high performance"),
                html.Li("Average scores closer to 1.0 indicate higher confidence in detections")
            ], style={'fontSize': '16px', 'lineHeight': '1.6'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})
    ])

def render_timeline():
    """Render timeline analysis"""
    timeline_data = fetch_timeline_data(hours=24)
    
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, (model_name, df) in enumerate(timeline_data.items()):
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df['hour'],
                y=df['anomalies'],
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[i], width=2),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title='Anomaly Detection Timeline (Last 24 Hours)',
        xaxis_title='Time',
        yaxis_title='Anomaly Count',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.P('Note: Timeline shows hourly aggregation of detected anomalies across all models.',
               style={'color': '#7f8c8d', 'fontStyle': 'italic', 'marginTop': '10px'})
    ])

def render_top_anomalies():
    """Render top anomalies view"""
    
    # Model selector
    selector = html.Div([
        html.Label('Select Model:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='anomaly-model-selector',
            options=[{'label': name, 'value': table} for name, table in TABLES.items()],
            value=TABLES['Enhanced Ensemble'],
            style={'width': '300px'}
        )
    ], style={'marginBottom': '20px'})
    
    return html.Div([
        selector,
        html.Div(id='top-anomalies-content')
    ])

@app.callback(
    Output('top-anomalies-content', 'children'),
    [Input('anomaly-model-selector', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def update_top_anomalies(table_name, n_clicks):
    """Update top anomalies display"""
    df = fetch_top_anomalies(table_name, limit=20)
    
    if df.empty:
        return html.Div("No anomalies found in this model.", 
                       style={'textAlign': 'center', 'padding': '50px', 'color': '#7f8c8d'})
    
    # Create scatter plot
    if table_name == 'enhanced_ensemble_strategy_results':
        fig = px.scatter(
            df,
            x='temperature',
            y='humidity',
            size='base_score',
            color='final_severity',
            hover_data=['sensor_id', 'confidence_score'],
            title='Anomaly Distribution (Temperature vs Humidity)',
            height=400
        )
    else:
        fig = px.scatter(
            df,
            x='temperature',
            y='humidity',
            size='anomaly_score',
            color='anomaly_score',
            hover_data=['sensor_id'],
            title='Anomaly Distribution (Temperature vs Humidity)',
            height=400,
            color_continuous_scale='Reds'
        )
    
    # Create data table
    display_df = df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select columns to display
    if table_name == 'enhanced_ensemble_strategy_results':
        cols = ['id', 'timestamp', 'sensor_id', 'temperature', 'humidity', 
                'final_severity', 'confidence_score', 'base_score']
    else:
        cols = ['id', 'timestamp', 'sensor_id', 'temperature', 'humidity', 
                'anomaly_score', 'is_anomaly']
    
    table = dash_table.DataTable(
        data=display_df[cols].to_dict('records'),
        columns=[{'name': col, 'id': col} for col in cols],
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '14px'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
            {'if': {'column_id': 'final_severity', 'filter_query': '{final_severity} contains "CRITICAL"'},
             'backgroundColor': '#e74c3c', 'color': 'white'},
            {'if': {'column_id': 'final_severity', 'filter_query': '{final_severity} contains "HIGH"'},
             'backgroundColor': '#f39c12', 'color': 'white'}
        ],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.H3('Top Anomalies Details', style={'marginTop': '30px', 'color': '#2c3e50'}),
        table
    ])

def render_detailed_analysis():
    """Render detailed analysis view"""
    
    selector = html.Div([
        html.Label('Select Model for Detailed Analysis:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='detail-model-selector',
            options=[{'label': name, 'value': table} for name, table in TABLES.items()],
            value=TABLES['Enhanced Ensemble'],
            style={'width': '300px', 'marginBottom': '20px'}
        )
    ])
    
    return html.Div([
        selector,
        html.Div(id='detailed-analysis-content')
    ])

@app.callback(
    Output('detailed-analysis-content', 'children'),
    [Input('detail-model-selector', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def update_detailed_analysis(table_name, n_clicks):
    """Update detailed analysis display"""
    df = fetch_top_anomalies(table_name, limit=5)
    
    if df.empty:
        return html.Div("No data available for detailed analysis.", 
                       style={'textAlign': 'center', 'padding': '50px', 'color': '#7f8c8d'})
    
    cards = []
    
    for idx, row in df.iterrows():
        # Get full analysis report
        full_report = row.get('full_analysis_report', 'No analysis report available')
        individual_explanation = row.get('individual_explanation', 'No explanation available')
        
        card = html.Div([
            html.H4(f"Anomaly ID: {row['id']}", style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.Div([
                html.Strong('Timestamp: '), str(row['timestamp']), html.Br(),
                html.Strong('Sensor ID: '), str(row['sensor_id']), html.Br(),
                html.Strong('Temperature: '), f"{row['temperature']:.2f}°C", html.Br(),
                html.Strong('Humidity: '), f"{row['humidity']:.2f}%", html.Br(),
            ], style={'marginBottom': '15px', 'color': '#34495e'}),
            
            html.Div([
                html.H5('Individual Explanation:', style={'color': '#16a085', 'marginBottom': '5px'}),
                html.P(individual_explanation, style={'backgroundColor': '#ecf0f1', 'padding': '10px', 
                                                     'borderRadius': '5px', 'fontSize': '14px'})
            ], style={'marginBottom': '15px'}),
            
            html.Details([
                html.Summary('View Full Analysis Report', style={'cursor': 'pointer', 'color': '#3498db', 
                                                                 'fontWeight': 'bold'}),
                html.Pre(full_report, style={'backgroundColor': '#2c3e50', 'color': '#ecf0f1', 
                                            'padding': '15px', 'borderRadius': '5px', 'fontSize': '12px',
                                            'overflowX': 'auto', 'marginTop': '10px'})
            ])
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'border': '2px solid #e74c3c' if row.get('is_anomaly', 0) == 1 else '1px solid #ddd'
        })
        
        cards.append(card)
    
    return html.Div(cards)

# Run the app
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Anomaly Detection Dashboard")
    print("=" * 60)
    print(f"📊 Connected to: {CLICKHOUSE_HOST}")
    print(f"🗄️  Database: {DATABASE_NAME}")
    print(f"📋 Monitoring {len(TABLES)} tables:")
    for name, table in TABLES.items():
        print(f"   • {name}: {table}")
    print("=" * 60)
    print("🌐 Dashboard running at: http://localhost:8050")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=8050)
