import dash
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from requests_oauthlib import OAuth2Session
from requests import request
from dotenv import load_dotenv, find_dotenv
import os

is_deployed = os.getenv('DEPLOYED')
is_deployed = True

# Load .env variables if not deployed.
if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login=[]
else:
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    redirect_uri = 'https://goalkeeper.nearnorthanalytics.com'
    # OAuth2 Settings
    authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
    token_url = 'https://accounts.google.com/o/oauth2/token'
    scope = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]

# Update get_login to be a list
    get_login = [
        dbc.Button("Login", id="login-button", color="success", size="sm"),
        dbc.Tooltip("Login with your Google Account", target="login-button"),
        dcc.Location(id='url', refresh=True)
    ]

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SKETCHY,
                                                                dbc.icons.BOOTSTRAP,
                                                                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css",
                                                                "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"],
                                                                suppress_callback_exceptions=True,
                                                                prevent_initial_callbacks=True)

server = app.server

color_mode_switch = [
                dbc.Label(className="fa-solid fa-moon", html_for="theme-switch"),
                dbc.Switch(id="theme-switch", value=True, className="d-inline-block ms-1", persistence=True),
                dbc.Label(className="fa-regular fa-sun", html_for="theme-switch"),
                # html.Div(className="vr")
            ]
            # width=3, className="d-flex align-content-center"



title = 'Welcome to the Goalkeeper'

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(color_mode_switch + get_login, className="d-flex justify-content-start"), width=1, className="d-flex float-start justify-content-md-start"),
        dbc.Col(
            html.Div([
                html.H2(title, className="text-center")
            ], className="d-flex justify-content-center align-items-start h-100"), 
            width=9.0
        ),
        dbc.Col([
            html.Div([
                dbc.Button(
                    size="sm",
                    id="entity-graph-button",
                    n_clicks=0,
                    class_name="ml-auto fa-solid fa-share-nodes",
                    color='success'
                ),
                dbc.Tooltip(
                    "Entity Memory Graph",
                    target="entity-graph-button",
                    id="entity-button-tooltip"
                ),
                dbc.Button(
                    size="sm",
                    id="memory-button",
                    n_clicks=0,
                    class_name="ml-auto fa-solid fa-brain",
                    color="danger"
                ),
                dbc.Tooltip(
                    "Memory",
                    target="memory-button",
                    id="memory-button-tooltip"
                ),
                dbc.Button(
                    size="sm",
                    id="settings-button",
                    n_clicks=0,
                    class_name="ml-auto fa-sharp fa-solid fa-gear",
                    color='warning'
                ),
                dbc.Tooltip(
                    "Settings",
                    target="settings-button",
                    id="settings-button-tooltip"
                ),
                dbc.Button(
                    size="sm",
                    id="about-button",
                    n_clicks=0,
                    class_name="bi bi-question-circle-fill"
                ),
                dbc.Tooltip(
                    "About",
                    target="about-button",
                    id="about-button-tooltip"
                ),
            ])
        ], className="d-grid gap-2 d-md-flex justify-content-md-end"),
    ], className=""),
    html.Div(id='login-content'),
    # dash.page_container
], fluid=True, className='dashboard-container border_rounded')

@app.callback(
        Output('url', 'href'), 
        [Input('login-button', 'n_clicks')],
        # prevent_initial_callback=True
        )
def login_with_google(n_clicks):
    if n_clicks:
        if is_deployed:
            google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
            authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")
            return authorization_url
    return None

@app.callback(
        Output('login-content', 'children'), 
        [Input('url', 'search')],
        # prevent_initial_callback=True
        )

def display_user_info(query_string):
    if is_deployed:
        if query_string:
            google = OAuth2Session(client_id, redirect_uri=redirect_uri)
            token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            return html.Div([
                html.H4(f"Welcome, {user_info['name']}"),
                html.P(f"Email: {user_info['email']}"),
                dash.page_container
            ])
        return "Please login with Google."
    else:
        return ""

if __name__ == '__main__':
    app.run_server(debug=True, port=os.getenv('DASH_PORT'))