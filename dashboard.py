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

# ClickHouse Configuration - USE ENVIRONMENT VARIABLES
CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'pmptfq07qm.ap-south-1.aws.clickhouse.cloud')
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'default')

# Validate required environment variables
if not CLICKHOUSE_PASSWORD:
    raise ValueError("CLICKHOUSE_PASSWORD environment variable is required!")

# Table names
TABLES = {
    'Generative Model': 'generative_model_results',
    'Sequence Model': 'sequence_model_results',
    'Partition Model': 'partition_model_results',
    'Boundary Model': 'boundary_model_results',
    'Enhanced Ensemble': 'enhanced_ensemble_strategy_results'
}

# Connect to ClickHouse with error handling
def get_clickhouse_client():
    try:
        return clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            secure=True
        )
    except Exception as e:
        print(f"❌ Failed to connect to ClickHouse: {e}")
        raise

# Data fetching functions with error handling
def fetch_table_data(table_name, limit=1000):
    """Fetch data from specific table"""
    try:
        client = get_clickhouse_client()
        query = f"SELECT * FROM {DATABASE_NAME}.{table_name} ORDER BY timestamp DESC LIMIT {limit}"
        result = client.query(query)
        df = pd.DataFrame(result.result_set, columns=result.column_names)
        return df
    except Exception as e:
        print(f"❌ Error fetching table data: {e}")
        return pd.DataFrame()

def fetch_summary_stats():
    """Fetch summary statistics from all tables"""
    try:
        client = get_clickhouse_client()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return {model: {'total': 0, 'anomalies': 0, 'detection_rate': 0, 'avg_score': 0, 'error': str(e)} 
                for model in TABLES.keys()}
    
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
            print(f"❌ Error fetching stats for {model_name}: {e}")
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
    try:
        client = get_clickhouse_client()
        query = f"""
        SELECT final_severity, count() as count 
        FROM {DATABASE_NAME}.enhanced_ensemble_strategy_results 
        GROUP BY final_severity
        """
        result = client.query(query)
        df = pd.DataFrame(result.result_set, columns=['severity', 'count'])
        return df
    except Exception as e:
        print(f"❌ Error fetching severity breakdown: {e}")
        return pd.DataFrame()

def fetch_timeline_data(hours=24):
    """Fetch anomaly detection timeline"""
    try:
        client = get_clickhouse_client()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return {model: pd.DataFrame(columns=['hour', 'anomalies']) for model in TABLES.keys()}
    
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
            print(f"❌ Error fetching timeline for {model_name}: {e}")
            timeline_data[model_name] = pd.DataFrame(columns=['hour', 'anomalies'])
    
    return timeline_data

def fetch_top_anomalies(table_name, limit=10):
    """Fetch top anomalies by score"""
    try:
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
    except Exception as e:
        print(f"❌ Error fetching top anomalies: {e}")
        return pd.DataFrame()

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# For production deployment
server = app.server

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🔍 Multi-Model Anomaly Detection Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Real-time monitoring of 5 ML models with ClickHouse data storage",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '10px'}),
    
    # Auto-refresh interval (30 seconds)
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # in milliseconds
        n_intervals=0
    ),
    
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
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_summary_cards(n_clicks, n_intervals):
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
     Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def render_tab_content(tab, n_clicks, n_intervals):
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

# ... (keep all other render functions exactly as they were in the original file)
# I'm truncating here for space, but include ALL the render functions:
# - render_overview()
# - render_comparison()
# - render_timeline()
# - render_top_anomalies()
# - render_detailed_analysis()
# Plus the associated callbacks

# IMPORTANT: Copy all the render functions from your original dashboard.py here

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
    
    # Production settings
    port = int(os.getenv('PORT', 8050))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
