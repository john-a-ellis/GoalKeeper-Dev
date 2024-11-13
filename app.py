import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask import request
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv, find_dotenv
import os
from urllib.parse import urlparse, parse_qs
from oauthlib.oauth2.rfc6749.errors import OAuth2Error
import logging
import sys
    
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'  # Ensure secure transport
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'   # Relax scope checking



def get_redirect_uri():
    """Dynamically determine the redirect URI based on request origin"""
    return 'https://goalkeeper-dev.onrender.com'


is_deployed = os.getenv('DEPLOYED', 'False').lower() == 'true'
# is_deployed = True

# Load .env variables if not deployed
if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login = []
else:
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    redirect_uri = get_redirect_uri()
    # redirect_uri = 'https://goalkeeper-dev.onrender.com'
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
                prevent_initial_callbacks=True
                )

server = app.server

def construct_callback_url(redirect_uri, pathname, query_string):
    base = redirect_uri.rstrip('/')
    path = pathname.lstrip('/') if pathname else ''
    return f"{base}/{path}{query_string}"

color_mode_switch = [
    dbc.Label(className="fa-solid fa-moon", html_for="theme-switch"),
    dbc.Switch(id="theme-switch", value=True, className="d-inline-block ms-1", persistence=True),
    dbc.Label(className="fa-regular fa-sun", html_for="theme-switch"),
]

title = 'Welcome to the Goalkeeper'

def create_header(is_authenticated=False, user_info="default"):
    # Display user info based on authentication state and deployment status
    user_display = html.Div(
        [
            html.I(className="fas fa-user me-2"),
            html.Span(
                "testing" if not is_deployed else (
                    user_info if is_authenticated else "not logged in"
                ),
                className="text-muted", id='login-span'
            ),
            dbc.Tooltip("Logout", target='login-span')
        ],
        id='login-name', className="d-flex align-items-center me-3"
    )

    return dbc.Row([
        dbc.Col(html.Div(color_mode_switch +  ([] if is_authenticated else get_login), 
                className="d-flex justify-content-start"), width=3, 
                className="d-flex float-start justify-content-md-start"),
        dbc.Col(
            html.Div([
                html.H2(title, className="text-center")
            ], className="d-flex justify-content-center align-items-start h-100"), 
            width=6
        ),
        dbc.Col([
            html.Div([
                # User Display Component
                user_display,
                dbc.Button(
                    size="md",
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
                    size="md",
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
                    size="md",
                    id="settings-button",
                    n_clicks=0,
                    class_name="ml-auto fa-sharp fa-solid fa-gear",
                    color="warning"
                ),
                dbc.Tooltip(
                    "Settings",
                    target="settings-button",
                    id="settings-button-tooltip"
                ),
                dbc.Button(
                    size="md",
                    id="about-button",
                    n_clicks=0,
                    class_name="ml-auto fa-solid fa-circle-info",
                    # color="info"  
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
    dcc.Loading(id="loading-response", type="cube", children=html.Div(id="loading-response-div"), target_components={"auth-store": "*"}),
    dcc.Store(id='auth-store', storage_type='session'),
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content'),
], fluid=True, className='m-3 dashboard-container border_rounded min-vh-75', id='main-container', style={'height': '900px'})

#Theme Switch Callback
@app.callback(
        Output('main-container', 'style'),
        Input('theme-switch', 'value')
)

def set_background(switch_value):
    return { 
        'background-color': 'darkslategray' if switch_value == False else 'aliceblue',
        'border-radius': '10px',
        }

# login callback
@app.callback(
    Output('url', 'href'),
    Output('loading-response','children'),
    Output('login-button', 'disabled'),
    Input('login-button', 'n_clicks'),
    prevent_initial_call = True
)
def login_with_google(n_clicks):
    if n_clicks and is_deployed:
        try:
            # logger.debug("=== Starting OAuth Flow ===")     
            
            # Get dynamic redirect URI
            redirect_uri = get_redirect_uri()
            google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)

            authorization_url, state = google.authorization_url(
                authorization_base_url, 
                access_type="offline", 
                prompt="select_account"
            )

            return authorization_url, no_update, True
        except Exception as e:
            return no_update, no_update, False
    return no_update, no_update, False

# logout callback
@app.callback(
        Output('login-span', 'children'),
        Output('auth-store', 'clear_data'),
        Output('url', 'href', allow_duplicate=True),
        Input('login-span', 'n_clicks'),
        # Input('login-span', 'children'),
        prevent_initial_call = True
        
)
def logout(clicked):
    if clicked and is_deployed:
        {'authenticated': False}
        create_header()
        return "Not Logged in", True, get_redirect_uri()
    else:
        return no_update, False, no_update
    
# authentication callback
@app.callback(
    Output('page-content', 'children'),
    Output('auth-store', 'data'),
    Input('url', 'pathname'),
    Input('url', 'search'),
    State('auth-store', 'data'),
    prevent_initial_call = True
)
def update_page_content(pathname, query_string, auth_data):
    if not is_deployed:
        # For local development, show everything without authentication
        return [html.Div([
            create_header(True),  # Will display "testing"
            dash.page_container
        ]), {'authenticated': True}]
    
    # Check if already authenticated
    if auth_data and auth_data.get('authenticated'):
        user_email = auth_data.get('user_info', {}).get('email', 'User')
        return [html.Div([
            create_header(True, user_email),
            dash.page_container
        ]), auth_data]


    # Handle new authentication
    if query_string:
        
        # Construct and log the full callback URL
        try:
            # Get dynamic redirect URI
            redirect_uri = get_redirect_uri()
            
            
            google = OAuth2Session(
                client_id, 
                redirect_uri=redirect_uri
            )
            
            callback_url = f"{redirect_uri.rstrip('/')}{pathname or ''}{query_string}"
           
            try:
                token = google.fetch_token(
                    token_url, 
                    client_secret=client_secret,
                    authorization_response=callback_url,
                    include_client_id=True
                )
                
            except OAuth2Error as oauth_err:
                raise
                
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()

            user_email = user_info.get('email', 'User')

            return [html.Div([
                create_header(True, user_email),
                dash.page_container
            ]), {'authenticated': True, 'user_info': user_info}]
            
        except Exception as e:

            return [html.Div([
                create_header(False),  # Will display "not logged in"
                html.Div(f"Authentication failed: {str(e)}", className="text-danger")
            ]), {'authenticated': False}]
    
    return [html.Div([
        create_header(False),  # Will display "not logged in"
        # html.Div("Please login with Google to access the application.", className="text-start")
    ]), {'authenticated': False}]

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.getenv('DASH_PORT')))