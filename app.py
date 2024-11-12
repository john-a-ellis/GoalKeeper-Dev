import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv, find_dotenv
import os
import logging
import sys

# OAuth2 Configuration
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'  # Ensure secure transport
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'   # Relax scope checking

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('oauth_debug')

# Environment and OAuth2 settings
is_deployed = os.getenv('DEPLOYED', 'False').lower() == 'true'

if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login = []
else:
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    redirect_uri = 'https://goalkeeper-dev.onrender.com'
    authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
    token_url = 'https://accounts.google.com/o/oauth2/token'
    scope = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]
    
    get_login = [
        dbc.Button("Login", id="login-button", color="success", size="sm"),
        dbc.Tooltip("Login with your Google Account", target="login-button"),
        dcc.Location(id='url', refresh=True)
    ]

# Initialize Dash app
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.SKETCHY,
        dbc.icons.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css",
        "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    ],
    suppress_callback_exceptions=True
)

server = app.server

# UI Components
color_mode_switch = [
    dbc.Label(className="fa-solid fa-moon", html_for="theme-switch"),
    dbc.Switch(id="theme-switch", value=True, className="d-inline-block ms-1", persistence=True),
    dbc.Label(className="fa-regular fa-sun", html_for="theme-switch"),
]

def create_header(is_authenticated=False, user_info="default"):
    user_display = html.Div(
        [
            html.I(className="fas fa-user me-2"),
            html.Span(
                "testing" if not is_deployed else (
                    user_info if is_authenticated else "not logged in"
                ),
                className="text-muted",
                id='login-span'
            ),
            dbc.Tooltip("Logout", target='login-span')
        ],
        id='login-name',
        className="d-flex align-items-center me-3"
    )

    return dbc.Row([
        dbc.Col(
            html.Div(
                color_mode_switch + ([] if is_authenticated else get_login),
                className="d-flex justify-content-start"
            ),
            width=3,
            className="d-flex float-start justify-content-md-start"
        ),
        dbc.Col(
            html.Div(
                html.H2("Welcome to the Goalkeeper", className="text-center"),
                className="d-flex justify-content-center align-items-start h-100"
            ),
            width=6
        ),
        dbc.Col([
            html.Div([
                # User Display Component
                user_display,
                # Entity Graph Button
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
                # Memory Button
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
                # Settings Button
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
                # About Button
                dbc.Button(
                    size="md",
                    id="about-button",
                    n_clicks=0,
                    class_name="ml-auto fa-solid fa-circle-info"
                ),
                dbc.Tooltip(
                    "About",
                    target="about-button",
                    id="about-button-tooltip"
                ),
            ])
        ], className="d-grid gap-2 d-md-flex justify-content-md-end"),
    ])

# App Layout
app.layout = dbc.Container([
    dcc.Loading(
        id="loading-response",
        type="cube",
        children=html.Div(id="loading-response-div"),
        target_components={"auth-store": "*"}
    ),
    dcc.Store(id='auth-store', storage_type='session'),
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
], fluid=True, className='m-3 dashboard-container border_rounded min-vh-75', id='main-container', style={'height': '900px'})

# Callbacks
@app.callback(
    Output('url', 'href'),
    Output('loading-response', 'children'),
    Output('login-button', 'disabled'),
    Input('login-button', 'n_clicks'),
    prevent_initial_call=True
)
def login_with_google(n_clicks):
    if n_clicks and is_deployed:
        try:
            google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
            authorization_url, state = google.authorization_url(
                authorization_base_url,
                access_type="offline",
                prompt="select_account"
            )
            return authorization_url, no_update, True
        except Exception as e:
            logger.error(f"OAuth Flow Error: {str(e)}", exc_info=True)
            return no_update, no_update, False
    return no_update, no_update, False

@app.callback(
    Output('login-span', 'children'),
    Output('auth-store', 'clear_data'),
    Output('url', 'href', allow_duplicate=True),
    Input('login-span', 'n_clicks'),
    prevent_initial_call='initial_duplicate'
)
def logout(clicked):
    if clicked and is_deployed:
        return "Not Logged in", True, redirect_uri
    return no_update, False, no_update

@app.callback(
    Output('page-content', 'children'),
    Output('auth-store', 'data'),
    Input('url', 'pathname'),
    Input('url', 'search'),
    State('auth-store', 'data')
)
def update_page_content(pathname, query_string, auth_data):
    if not is_deployed:
        return [html.Div([
            create_header(True),
            dash.page_container
        ]), {'authenticated': True}]

    if auth_data and auth_data.get('authenticated'):
        user_email = auth_data.get('user_info', {}).get('email', 'User')
        return [html.Div([
            create_header(True, user_email),
            dash.page_container
        ]), auth_data]

    if query_string:
        try:
            google = OAuth2Session(client_id, redirect_uri=redirect_uri)
            callback_url = f"{redirect_uri}{pathname or ''}{query_string}"
            
            token = google.fetch_token(
                token_url,
                client_secret=client_secret,
                authorization_response=callback_url,
                include_client_id=True
            )
            
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            user_email = user_info.get('email', 'User')

            return [html.Div([
                create_header(True, user_email),
                dash.page_container
            ]), {'authenticated': True, 'user_info': user_info}]
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            return [html.Div([
                create_header(False),
                html.Div(f"Authentication failed: {str(e)}", className="text-danger")
            ]), {'authenticated': False}]
    
    return [html.Div([
        create_header(False)
    ]), {'authenticated': False}]

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.getenv('DASH_PORT')))