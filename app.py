import dash
from dash import dcc, html, callback, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from requests_oauthlib import OAuth2Session
from requests import request
from dotenv import load_dotenv, find_dotenv
import os

is_deployed = os.getenv('DEPLOYED', 'True').lower() == 'true'
# is_deployed = True

# Load .env variables if not deployed
if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login = []
else:
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    redirect_uri = 'https://localhost:3050'
    # OAuth2 Settings
    authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
    token_url = 'https://accounts.google.com/o/oauth2/token'
    scope = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]

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
]

title = 'Welcome to the Goalkeeper'

def create_header(is_authenticated=False):
    return dbc.Row([
        dbc.Col(html.Div(color_mode_switch + ([] if is_authenticated else get_login), 
                className="d-flex justify-content-start"), width=1, 
                className="d-flex float-start justify-content-md-start"),
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
    ], className="")

# Store for authentication state
app.layout = dbc.Container([
    dcc.Store(id='auth-store', storage_type='session'),
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content'),
], fluid=True, className='dashboard-container border_rounded')

@app.callback(
    Output('url', 'href'),
    [Input('login-button', 'n_clicks')],
)
def login_with_google(n_clicks):
    if n_clicks and is_deployed:
            google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
            authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline", prompt="select_account")
            return authorization_url
    return no_update

@app.callback(
    [Output('page-content', 'children'),
     Output('auth-store', 'data')],
    [Input('url', 'pathname'),
     Input('url', 'search')],
    [State('auth-store', 'data')],
    prevent_initial_call=False

)
def update_page_content(pathname, query_string, auth_data):
    if not is_deployed:
        # For local development, show everything without authentication
        return [html.Div([
            create_header(True),
            dash.page_container
        ]), {'authenticated': True}]
    
    # Check if already authenticated
    if auth_data and auth_data.get('authenticated'):
        return [html.Div([
            create_header(True),
            dash.page_container
        ]), auth_data]

    # Handle new authentication
    if query_string:
        try:
            google = OAuth2Session(client_id, redirect_uri=redirect_uri)
            full_url = f"{redirect_uri}{pathname}{query_string}"
            token = google.fetch_token(token_url, client_secret=client_secret, 
                                     authorization_response=full_url)
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            
            return [html.Div([
                create_header(True),
                html.Div([
                    html.H4(f"Welcome, {user_info['name']}", className="text-center"),
                    html.P(f"Email: {user_info['email']}", className="text-center"),
                ]),
                dash.page_container
            ]), {'authenticated': True, 'user_info': user_info}]
        except Exception as e:
            print(f"Authentication error: {str(e)}")  # Add debugging
            return [html.Div([
                create_header(False),
                html.Div("Authentication failed. Please try again.", 
                         className="text-center text-danger")
            ]), {'authenticated': False}]
    
    # Show login page
    return [html.Div([
        create_header(False),
        html.Div("Please login with Google to access the application.", 
                 className="d-flex justify-content-center align-items-start h-100")
    ]), {'authenticated': False}]

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.getenv('DASH_PORT')))