from dash import Dash, html, dcc, Input, Output
import pickle
import numpy as np

# Load models and vectorizer
with open('random_forest_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('naive_bayes_model.pkl', 'rb') as nb_file:
    nb_model = pickle.load(nb_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Preprocessing function
def preprocess_input_text(text):
    return text.lower()

# Initializing Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div(
    style={
        'fontFamily': 'Arial, sans-serif',
        'background': 'linear-gradient(135deg, #74ebd5, #acb6e5)',
        'minHeight': '100vh',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'padding': '20px',
        'boxSizing': 'border-box',
    },
    children=[
        html.Div(
            style={
                'backgroundColor': 'white',
                'padding': '30px',
                'width': '100%',
                'maxWidth': '600px',
                'boxShadow': '0 8px 15px rgba(0, 0, 0, 0.2)',
                'borderRadius': '12px',
                'textAlign': 'center',
                'boxSizing': 'border-box',
            },
            children=[
                html.H1(
                    "Sentiment Analysis on Social Media Posts",
                    style={
                        'color': '#333',
                        'fontSize': '24px',
                        'marginBottom': '20px',
                    },
                ),
                dcc.Textarea(
                    id='post-input',
                    placeholder='Enter a social media post...',
                    style={
                        'width': '100%',
                        'height': '100px',
                        'border': '1px solid #ddd',
                        'borderRadius': '8px',
                        'padding': '10px',
                        'fontSize': '16px',
                        'boxShadow': 'inset 0 2px 4px rgba(0, 0, 0, 0.1)',
                        'resize': 'none',
                        'outline': 'none',
                    },
                ),
                html.Button(
                    'Analyze Sentiment',
                    id='analyze-button',
                    n_clicks=0,
                    style={
                        'marginTop': '20px',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 20px',
                        'fontSize': '16px',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'transition': 'background-color 0.3s ease',
                    },
                ),
                html.Div(
                    id='nb-output',
                    style={
                        'marginTop': '20px',
                        'fontSize': '18px',
                        'color': '#555',
                        'fontWeight': 'bold',
                    },
                ),
                html.Div(
                    id='rf-output',
                    style={
                        'marginTop': '10px',
                        'fontSize': '18px',
                        'color': '#555',
                        'fontWeight': 'bold',
                    },
                ),
            ],
        )
    ],
)

# Callback to analyze sentiment
@app.callback(
    [Output('nb-output', 'children'),
     Output('rf-output', 'children')],
    [Input('post-input', 'value'),
     Input('analyze-button', 'n_clicks')]
)
def analyze_sentiment(post_input, n_clicks):
    if n_clicks > 0 and post_input:
        try:
            processed_text = preprocess_input_text(post_input)
            input_vector = tfidf.transform([processed_text]).toarray()

            nb_prediction = nb_model.predict(input_vector)[0]
            rf_prediction = rf_model.predict(input_vector)[0]

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            nb_result = f"Naive Bayes Sentiment: {sentiment_map[nb_prediction]}"
            rf_result = f"Random Forest Sentiment: {sentiment_map[rf_prediction]}"

            return nb_result, rf_result

        except Exception as e:
            return f"Error: {str(e)}", ""
    return "", ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

