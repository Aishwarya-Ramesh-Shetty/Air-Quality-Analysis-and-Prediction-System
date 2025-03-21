from flask import Flask, render_template, request, jsonify
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Initialize Flask application
server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/dash/')

# Updated constants - simplified color scheme for 4 categories
AIR_QUALITY_COLORS = {
    'Good': '#10B981',      # Green
    'Moderate': '#FBBF24',  # Yellow
    'Poor': '#F97316',      # Orange
    'Hazardous': '#EF4444'  # Red
}

# Map numeric labels back to categories
AIR_QUALITY_LABELS = {
    0: 'Good',
    1: 'Moderate',
    2: 'Poor',
    3: 'Hazardous'
}

# Helper function to get units
def get_unit(column):
    units = {
        'Temperature': '°C',
        'Humidity': '%',
        'PM2.5': 'μg/m³',
        'PM10': 'μg/m³',
        'NO2': 'ppb',
        'SO2': 'ppb',
        'CO': 'ppm',
        'Proximity_to_Industrial_Areas': 'km',
        'Population_Density': 'people/km²'
    }
    return units.get(column, '')

# Function to load data with standardized categories
def load_data(file_path=None):
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # Clean data if needed
            df.dropna(inplace=True)
            
            # Check if we need to standardize air quality categories
            target_column = 'Air Quality Levels' if 'Air Quality Levels' in df.columns else 'Air Quality'
            
            # Standardize to 4 categories if needed
            if target_column in df.columns and df[target_column].dtype == 'object':
                # Map original categories to our 4 categories
                category_mapping = {
                    'Good': 'Good',
                    'Moderate': 'Moderate',
                    'Unhealthy for Sensitive Groups': 'Poor',
                    'Unhealthy': 'Poor',
                    'Very Unhealthy': 'Hazardous',
                    'Hazardous': 'Hazardous'
                }
                
                # Apply mapping or keep original if not in mapping
                df[target_column] = df[target_column].apply(
                    lambda x: category_mapping.get(x, x) if isinstance(x, str) else x
                )
            
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return generate_sample_data()
    else:
        return generate_sample_data()

# Generate sample data if no file is provided
def generate_sample_data():
    np.random.seed(42)
    n_samples = 500
    
    date_range = pd.date_range(start='2023-01-01', periods=n_samples)
    
    # Using our 4 categories
    air_quality_categories = ['Good', 'Moderate', 'Poor', 'Hazardous']
    air_quality_weights = [0.4, 0.3, 0.2, 0.1]  # Distribution weights
    
    data = {
        'Date': date_range,
        'Temperature': np.random.normal(25, 5, n_samples),
        'Humidity': np.random.normal(60, 15, n_samples),
        'PM2.5': np.random.gamma(2, 5, n_samples),
        'PM10': np.random.gamma(3, 8, n_samples),
        'NO2': np.random.gamma(1, 10, n_samples),
        'SO2': np.random.gamma(0.5, 5, n_samples),
        'CO': np.random.gamma(0.8, 0.5, n_samples),
        'Proximity_to_Industrial_Areas': np.random.gamma(2, 1, n_samples),
        'Population_Density': np.random.gamma(10, 300, n_samples),
        'Air Quality': np.random.choice(air_quality_categories, n_samples, p=air_quality_weights)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure all numerical values are positive
    numerical_columns = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 
                        'Proximity_to_Industrial_Areas', 'Population_Density']
    for col in numerical_columns:
        df[col] = df[col].clip(lower=0)
    
    return df

# Try to load real data from the specified path
try:
    file_path = "C:\Users\ASUS\Downloads\TechBlitz DataScience Dataset (1).csv"

    df = load_data(file_path)
    print("Loaded actual dataset from specified path")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Generating sample data instead")
    df = load_data()

# Check if the target column is called 'Air Quality' or 'Air Quality Levels'
target_column = 'Air Quality Levels' if 'Air Quality Levels' in df.columns else 'Air Quality'

# Train XGBoost model with simplified categories
def train_model(df, target_column):
    # Create a copy of the dataframe to avoid modifying the original
    model_df = df.copy()
    
    # Encode the target variable if it's categorical
    if model_df[target_column].dtype == 'object':
        labelencoder = LabelEncoder()
        model_df[target_column] = labelencoder.fit_transform(model_df[target_column])
        # Save the encoder for later use
        joblib.dump(labelencoder, 'label_encoder.pkl')
    
    # Features and target
    X = model_df.drop([target_column, 'Date'] if 'Date' in model_df.columns else [target_column], axis=1)
    
    # Remove non-numeric columns for model training
    X = X.select_dtypes(include=['number'])
    y = model_df[target_column]
    
    # Feature names for later use
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a basic XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Generate evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save the model
    joblib.dump(model, 'xgboost_model.pkl')
    
    return model, X_test, y_test, y_pred, conf_matrix, report, feature_names

# Train the model and get evaluation metrics
model, X_test, y_test, y_pred, conf_matrix, report, feature_names = train_model(df, target_column)

# Calculate statistics
def calculate_stats(df):
    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    stats = {}
    
    for column in numerical_columns:
        values = df[column].dropna()
        stats[column] = {
            'min': values.min(),
            'max': values.max(),
            'avg': values.mean(),
            'unit': get_unit(column)
        }
    
    return stats

# Calculate air quality distribution
def calculate_air_quality_distribution(df, target_column):
    distribution = df[target_column].value_counts().reset_index()
    distribution.columns = ['name', 'value']
    
    # Add color to each quality level
    if distribution['name'].dtype != 'object':
        # If the target is already encoded, map it back to labels
        distribution['name'] = distribution['name'].map(AIR_QUALITY_LABELS)
    
    # Map colors
    distribution['color'] = distribution['name'].map(AIR_QUALITY_COLORS)
    
    return distribution.to_dict('records')

# Calculate pollutant averages by air quality
def calculate_pollutant_averages(df, target_column):
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    # Group by air quality and calculate mean for each pollutant
    # Check if we need to handle encoded values
    if df[target_column].dtype != 'object':
        # Create a temporary column with decoded values
        temp_df = df.copy()
        temp_df['Air_Quality_Label'] = temp_df[target_column].map(AIR_QUALITY_LABELS)
        averages = temp_df.groupby('Air_Quality_Label')[available_pollutants].mean().reset_index()
        averages.rename(columns={'Air_Quality_Label': 'Air Quality'}, inplace=True)
    else:
        averages = df.groupby(target_column)[available_pollutants].mean().reset_index()
        averages.rename(columns={target_column: 'Air Quality'}, inplace=True)
    
    return averages.to_dict('records')

# Generate feature importance plot
def get_feature_importance_plot():
    # Get feature importances from the model
    importance = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Create a Plotly figure
    fig = px.bar(
        features_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Feature Importance in Air Quality Classification',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Importance Score',
        yaxis_title='Feature'
    )
    
    return fig

# Generate confusion matrix plot
def get_confusion_matrix_plot():
    # Get unique classes
    classes = sorted(list(set(y_test) | set(y_pred)))
    
    # Create labels for the plot
    if classes[0] in [0, 1, 2, 3]:
        labels = [AIR_QUALITY_LABELS.get(c, f"Class {c}") for c in classes]
    else:
        labels = [str(c) for c in classes]
    
    # Create the heatmap
    fig = px.imshow(
        conf_matrix,
        x=labels,
        y=labels,
        color_continuous_scale='Viridis',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix for Air Quality Classification"
    )
    
    # Add text annotations to the cells
    annotations = []
    for i, row in enumerate(conf_matrix):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(value),
                    showarrow=False,
                    font=dict(color="white" if value > conf_matrix.max() / 2 else "black")
                )
            )
            
    fig.update_layout(annotations=annotations)
    
    return fig

# Generate classification report table
def get_classification_report_table():
    # Convert the classification report to a dataframe
    report_df = pd.DataFrame(report).transpose()
    
    # Filter out the avg/total row
    report_df = report_df.drop(['accuracy'], errors='ignore')
    
    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                report_df.index,
                report_df['precision'].round(3),
                report_df['recall'].round(3),
                report_df['f1-score'].round(3),
                report_df['support']
            ],
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig.update_layout(title="Classification Report")
    
    return fig

# Compute all the stats
stats = calculate_stats(df)
air_quality_distribution = calculate_air_quality_distribution(df, target_column)
pollutant_averages = calculate_pollutant_averages(df, target_column)

# Create Stats Cards HTML
def create_stats_cards():
    cards = []
    for stat_name, stat_info in stats.items():
        if stat_name in ['Date', target_column] or isinstance(stat_info, str):
            continue
            
        cards.append(
            html.Div([
                html.H3(stat_name, className="text-lg font-semibold"),
                html.Div([
                    html.Div([
                        html.Span("Min: ", className="font-medium"),
                        html.Span(f"{stat_info['min']:.2f} {stat_info['unit']}")
                    ]),
                    html.Div([
                        html.Span("Max: ", className="font-medium"),
                        html.Span(f"{stat_info['max']:.2f} {stat_info['unit']}")
                    ]),
                    html.Div([
                        html.Span("Avg: ", className="font-medium"),
                        html.Span(f"{stat_info['avg']:.2f} {stat_info['unit']}")
                    ])
                ])
            ], className="bg-white p-4 rounded-lg shadow")
        )
    return cards

# Prediction function
def make_prediction(input_values):
    # Load the model, encoder, and feature names
    loaded_model = joblib.load('xgboost_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Create a DataFrame with the input values
    input_df = pd.DataFrame([input_values], columns=feature_names)
    
    # Make prediction
    prediction_numeric = loaded_model.predict(input_df)[0]
    
    # Convert to category
    try:
        label_encoder = joblib.load('label_encoder.pkl')
        prediction_category = label_encoder.inverse_transform([prediction_numeric])[0]
    except:
        prediction_category = AIR_QUALITY_LABELS.get(prediction_numeric, f"Class {prediction_numeric}")
    
    # Get probabilities
    probabilities = loaded_model.predict_proba(input_df)[0]
    
    return prediction_numeric, prediction_category, probabilities

# Define layout for the dashboard
app.layout = html.Div([
    # Navigation tabs
    dcc.Tabs([
        # Dashboard Tab
        dcc.Tab(label="Air Quality Analysis", children=[
            html.Div([
                html.H1("Air Quality Analysis Dashboard", className="text-2xl font-bold mb-6"),
                
                # Air Quality Distribution Section
                html.Div([
                    html.H2("Air Quality Distribution", className="text-xl font-semibold mb-4"),
                    dcc.Graph(
                        id='air-quality-distribution',
                        figure=px.pie(
                            air_quality_distribution,
                            names='name',
                            values='value',
                            color='name',
                            color_discrete_map=AIR_QUALITY_COLORS,
                            title='Air Quality Classification Distribution'
                        )
                    )
                ], className="mb-6"),
                
                # Pollutant Averages by Air Quality
                html.Div([
                    html.H2("Pollutant Levels by Air Quality Category", className="text-xl font-semibold mb-4"),
                    dcc.Graph(
                        id='pollutant-averages',
                        figure=px.bar(
                            pollutant_averages,
                            x='Air Quality',
                            y=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO'],
                            title='Average Pollutant Levels by Air Quality Category',
                            barmode='group',
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                    )
                ], className="mb-6"),
                
                # Air Quality Category Legend
                html.Div([
                    html.H2("Air Quality Categories Legend", className="text-xl font-semibold mb-4"),
                    html.Div([
                        html.Div([
                            html.Div(className="w-6 h-6 rounded-full mr-2", style={"backgroundColor": AIR_QUALITY_COLORS["Good"]}),
                            html.Div([
                                html.H3("Good", className="font-medium"),
                                html.P("Clean air with low pollution levels", className="text-sm text-gray-600")
                            ])
                        ], className="flex items-center mb-3"),
                        html.Div([
                            html.Div(className="w-6 h-6 rounded-full mr-2", style={"backgroundColor": AIR_QUALITY_COLORS["Moderate"]}),
                            html.Div([
                                html.H3("Moderate", className="font-medium"),
                                html.P("Acceptable air quality with some pollutants present", className="text-sm text-gray-600")
                            ])
                        ], className="flex items-center mb-3"),
                        html.Div([
                            html.Div(className="w-6 h-6 rounded-full mr-2", style={"backgroundColor": AIR_QUALITY_COLORS["Poor"]}),
                            html.Div([
                                html.H3("Poor", className="font-medium"),
                                html.P("Noticeable pollution that may cause health issues for sensitive groups", className="text-sm text-gray-600")
                            ])
                        ], className="flex items-center mb-3"),
                        html.Div([
                            html.Div(className="w-6 h-6 rounded-full mr-2", style={"backgroundColor": AIR_QUALITY_COLORS["Hazardous"]}),
                            html.Div([
                                html.H3("Hazardous", className="font-medium"),
                                html.P("Highly polluted air posing serious health risks to the population", className="text-sm text-gray-600")
                            ])
                        ], className="flex items-center")
                    ], className="bg-white p-4 rounded-lg shadow")
                ])
            ], className="container mx-auto px-4 py-8")
        ]),
        
        # Model Performance Tab
        dcc.Tab(label="Classifier Performance", children=[
            html.Div([
                html.H1("Air Quality Classifier Performance", className="text-2xl font-bold mb-6"),
                
                # Feature Importance
                html.Div([
                    html.H2("Feature Importance", className="text-xl font-semibold mb-4"),
                    dcc.Graph(
                        id='feature-importance',
                        figure=get_feature_importance_plot()
                    )
                ], className="mb-6"),
                
                # Confusion Matrix
                html.Div([
                    html.H2("Confusion Matrix", className="text-xl font-semibold mb-4"),
                    dcc.Graph(
                        id='confusion-matrix',
                        figure=get_confusion_matrix_plot()
                    )
                ], className="mb-6"),
                
                # Classification Report
                html.Div([
                    html.H2("Classification Report", className="text-xl font-semibold mb-4"),
                    dcc.Graph(
                        id='classification-report',
                        figure=get_classification_report_table()
                    )
                ])
            ], className="container mx-auto px-4 py-8")
        ]),
        
        # Prediction Tab
        dcc.Tab(label="Air Quality Prediction", children=[
            html.Div([
                html.H1("Air Quality Prediction Tool", className="text-2xl font-bold mb-6"),
                
                # Input Form
                html.Div([
                    html.H2("Enter Environmental Parameters", className="text-xl font-semibold mb-4"),
                    
                    # Create input fields for key features (simplified from original)
                    html.Div([
                        html.Div([
                            html.Label("PM2.5 (μg/m³): ", className="block mb-2"),
                            dcc.Input(
                                id="input-PM2.5",
                                type="number",
                                placeholder="Enter PM2.5 level",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pr-2"),
                        html.Div([
                            html.Label("PM10 (μg/m³): ", className="block mb-2"),
                            dcc.Input(
                                id="input-PM10",
                                type="number",
                                placeholder="Enter PM10 level",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pl-2"),
                        html.Div([
                            html.Label("NO2 (ppb): ", className="block mb-2"),
                            dcc.Input(
                                id="input-NO2",
                                type="number",
                                placeholder="Enter NO2 level",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pr-2"),
                        html.Div([
                            html.Label("SO2 (ppb): ", className="block mb-2"),
                            dcc.Input(
                                id="input-SO2",
                                type="number",
                                placeholder="Enter SO2 level",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pl-2"),
                        html.Div([
                            html.Label("CO (ppm): ", className="block mb-2"),
                            dcc.Input(
                                id="input-CO",
                                type="number",
                                placeholder="Enter CO level",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pr-2"),
                        html.Div([
                            html.Label("Temperature (°C): ", className="block mb-2"),
                            dcc.Input(
                                id="input-Temperature",
                                type="number",
                                placeholder="Enter temperature",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pl-2"),
                        html.Div([
                            html.Label("Humidity (%): ", className="block mb-2"),
                            dcc.Input(
                                id="input-Humidity",
                                type="number",
                                placeholder="Enter humidity",
                                className="w-full p-2 border rounded"
                            )
                        ], className="mb-4 md:w-1/2 pr-2"),
                    ], className="flex flex-wrap"),
                    
                    # Add hidden inputs for any other required features
                    html.Div(id="hidden-inputs", style={"display": "none"}),
                    
                    # Prediction button
                    html.Button("Predict Air Quality", 
                                id="predict-button", 
                                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700")
                ], className="bg-white p-6 rounded-lg shadow-md mb-6"),
                
                # Prediction Results
                html.Div([
                    html.H2("Prediction Results", className="text-xl font-semibold mb-4"),
                    html.Div(id="prediction-output", className="p-4 bg-gray-100 rounded-lg")
                ], className="bg-white p-6 rounded-lg shadow-md")
            ], className="container mx-auto px-4 py-8")
        ])
    ])
])

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input("input-PM2.5", "value"),
     Input("input-PM10", "value"),
     Input("input-NO2", "value"),
     Input("input-SO2", "value"),
     Input("input-CO", "value"),
     Input("input-Temperature", "value"),
     Input("input-Humidity", "value")]
)
def update_prediction(n_clicks, pm25, pm10, no2, so2, co, temperature, humidity):
    # Check if button was clicked
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'predict-button':
        return "Enter values and click 'Predict Air Quality' to get a prediction."
    
    # Check if all inputs are provided
    if any(value is None for value in [pm25, pm10, no2, so2, co, temperature, humidity]):
        return "Please fill in all fields before making a prediction."
    
    try:
        # Get all feature names from the model
        feature_names = joblib.load('feature_names.pkl')
        
        # Create input dictionary with provided values
        input_values = {
            "PM2.5": float(pm25),
            "PM10": float(pm10),
            "NO2": float(no2),
            "SO2": float(so2),
            "CO": float(co),
            "Temperature": float(temperature),
            "Humidity": float(humidity)
        }
        
        # Fill in any missing features with reasonable defaults
        for feature in feature_names:
            if feature not in input_values:
                if feature == "Proximity_to_Industrial_Areas":
                    input_values[feature] = 5.0  # Default value
                elif feature == "Population_Density":
                    input_values[feature] = 1000.0  # Default value
                else:
                    input_values[feature] = 0.0  # Default for any other missing features
        
        # Make prediction
        prediction_numeric, prediction_category, probabilities = make_prediction(input_values)
        
        # Get category color
        category_color = AIR_QUALITY_COLORS.get(prediction_category, "#3B82F6")
        
        # Format probabilities
        probs_formatted = []
        for i, prob in enumerate(probabilities):
            category = AIR_QUALITY_LABELS.get(i, f"Class {i}")
            probs_formatted.append(html.Div([
                html.Span(f"{category}: ", className="font-medium"),
                html.Span(f"{prob:.2%}")
            ]))
        
        return html.Div([
            html.Div([
                html.H3("Predicted Air Quality:", className="text-lg font-medium mb-2"),
                html.Div(prediction_category, 
                       className="text-xl font-bold p-3 rounded-lg text-center",
                       style={"backgroundColor": category_color, "color": "white"})
            ], className="mb-4"),
            html.Div([
                html.H3("Category Description:", className="text-lg font-medium mb-2"),
                html.Div(get_category_description(prediction_category), className="p-3 bg-gray-50 rounded-lg")
            ], className="mb-4"),
            html.Div([
                html.H3("Prediction Confidence:", className="text-lg font-medium mb-2"),
                html.Div(probs_formatted, className="space-y-1 p-3 bg-gray-50 rounded-lg")
            ])
        ])
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Helper function to get category descriptions
def get_category_description(category):
    descriptions = {
        'Good': "Clean air with low pollution levels. Safe for all activities.",
        'Moderate': "Acceptable air quality with some pollutants present. Most people can safely engage in outdoor activities.",
        'Poor': "Noticeable pollution that may cause health issues for sensitive groups. People with respiratory conditions should limit outdoor exposure.",
        'Hazardous': "Highly polluted air posing serious health risks to the population. Everyone should avoid outdoor activities."
    }
    return descriptions.get(category, "Unknown air quality category")

# Flask route for the main page
@server.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Air Quality Analysis & Prediction System</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f3f4f6;
            }
            .card {
                background-color: white;
                border-radius: 0.5rem;
                padding: 1.5rem;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            }
            .feature-item {
                display: flex;
                align-items: center;
                margin-bottom: 0.75rem;
            }
            .feature-icon {
                display: inline-flex;
                align-items: center;
                """