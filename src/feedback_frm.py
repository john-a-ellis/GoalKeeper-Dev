# import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from src.feedback_db import FeedbackStorage

feedback_form = dbc.Container([
    html.Img(src='assets/NearNorthCleanRight.png', height=100, className='align-start'),
    html.H2("Goalkeeper Feedback Form", className="text-center mb-4 mt-3"),
    html.P("Help us improve your AI coaching experience with MEL", className="text-center mb-4"),
    
    dbc.Form([
        # User Information
        dbc.Row([
            dbc.Col([
                dbc.Label("Name (optional)"),
                dbc.Input(type="text", id="name", placeholder="Enter your name")
            ], width=6),
            dbc.Col([
                dbc.Label("Email (optional)"),
                dbc.Input(type="email", id="email", placeholder="Enter your email")
            ], width=6),
        ], className="mb-3"),

        # Usage Frequency
        dbc.Row([
            dbc.Col([
                dbc.Label("How often do you use Goalkeeper?"),
                dbc.Select(
                    id="usage-frequency",
                    options=[
                        {"label": "Daily", "value": "daily"},
                        {"label": "2-3 times a week", "value": "weekly"},
                        {"label": "Once a week", "value": "once_weekly"},
                        {"label": "Monthly", "value": "monthly"},
                        {"label": "Less than monthly", "value": "less_than_monthly"}
                    ]
                )
            ], width=12)
        ], className="mb-3"),

        # MEL Rating
        dbc.Row([
            dbc.Col([
                dbc.Label("How would you rate your experience with MEL?"),
                dcc.Slider(
                    id="mel-rating",
                    min=1,
                    max=5,
                    step=1,
                    marks={i: str(i) for i in range(1, 6)},
                    value=3
                )
            ], width=12)
        ], className="mb-4"),

        # Feature Ratings
        dbc.Row([
            dbc.Col([
                html.H5("Please rate the following features:", className="mb-3"),
                dbc.Label("Goal Setting Assistance"),
                dcc.Slider(id="goal-setting-rating", min=1, max=5, step=1, marks={i: str(i) for i in range(1, 6)}, value=3, className="mb-3"),
                
                dbc.Label("Progress Tracking"),
                dcc.Slider(id="progress-tracking-rating", min=1, max=5, step=1, marks={i: str(i) for i in range(1, 6)}, value=3, className="mb-3"),
                
                dbc.Label("Coaching Feedback"),
                dcc.Slider(id="coaching-feedback-rating", min=1, max=5, step=1, marks={i: str(i) for i in range(1, 6)}, value=3, className="mb-3")
            ], width=12)
        ], className="mb-4"),
        # Feedback Message
        dbc.Row([
            dbc.Col([
                html.Div(id="feedback-message")
            ], width=12)
        ]),
        # Improvement Areas
        dbc.Row([
            dbc.Col([
                dbc.Label("Which areas of the app need the most improvement? (Select all that apply)"),
                dbc.Checklist(
                    id="improvement-areas",
                    options=[
                        {"label": "User Interface", "value": "ui"},
                        {"label": "MEL's Responses", "value": "mel_responses"},
                        {"label": "Goal Setting Features", "value": "goal_setting"},
                        {"label": "Progress Tracking", "value": "progress_tracking"},
                        {"label": "Notifications", "value": "notifications"},
                        {"label": "Mobile Experience", "value": "mobile"}
                    ],
                    className="mb-3"
                )
            ], width=12)
        ], className="mb-3"),

        # Open-ended Feedback
        dbc.Row([
            dbc.Col([
                dbc.Label("What do you like most about Goalkeeper and MEL?"),
                dbc.Textarea(id="likes", rows=3, className="mb-3")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("What suggestions do you have for improvement?"),
                dbc.Textarea(id="suggestions", rows=3, className="mb-3")
            ], width=12)
        ]),

        # Submit Button
        dbc.Row([
            dbc.Col([
                dbc.Button("Submit Feedback", color="primary", id="submit-feedback", n_clicks=0, className="w-100")
            ], width={"size": 6, "offset": 3})
        ], className="mt-4 mb-4"),

        
    ])
], fluid=True)

layout = feedback_form

# Callback for submit button
@callback(
    Output("feedback-message", "children"),
    # Output("feedback-modal", "is_open", allow_duplicate=True),
    Input("submit-feedback", "n_clicks"),
    State("name", "value"),
    State("email", "value"),
    State("usage-frequency", "value"),
    State("mel-rating", "value"),
    State("goal-setting-rating", "value"),
    State("progress-tracking-rating", "value"),
    State("coaching-feedback-rating", "value"),
    State("improvement-areas", "value"),
    State("likes", "value"),
    State("suggestions", "value"),
    prevent_initial_call=True
)
def submit_feedback(n_clicks, name, email, usage_frequency, mel_rating, goal_setting_rating, 
                   progress_tracking_rating, coaching_feedback_rating, improvement_areas, likes, suggestions):
    if n_clicks > 0:
        storage=FeedbackStorage()
        storage.initialize_database()
        try:
            feedback_data = {
                'name':name,
                'email':email,
                'usage_frequency':usage_frequency,
                'mel_rating':mel_rating,
                'goal_setting_rating': goal_setting_rating,
                'progress_tracking_rating':progress_tracking_rating,
                'coaching_feedback_rating':coaching_feedback_rating,
                'improvement_areas':improvement_areas or [],
                'likes':likes,
                'suggestions':suggestions,
            }
            # Store feedback
            feedback_id = storage.store_feedback(feedback_data)

            return dbc.Alert(
                ["Thank you for your feedback! We appreciate your help in improving Goalkeeper.", 
                 dbc.Button('Close', id='close-success-button', color='success', n_clicks=0)
                ],
                color="success",
                className="mt-3"
            )
        except Exception as e:
            return dbc.Alert("Oops! an error occurred submitting your feedback. Please try again.", 
                             color='danger')

@callback(
    Output('feedback-modal', 'is_open', allow_duplicate=True),
    Input('close-success-button', 'n_clicks'),
    prevent_initial_call = True
)

def close_feedback_modal(clicks):
    if clicks == 0:
        return True
    else:
        return False

