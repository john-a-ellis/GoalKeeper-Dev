import dash
from dash import dcc, html, callback, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask import request
from requests_oauthlib import OAuth2Session
# from requests import request
from dotenv import load_dotenv, find_dotenv
import os
from urllib.parse import urljoin, urlparse, parse_qs
from oauthlib.oauth2.rfc6749.errors import OAuth2Error
import logging
import sys

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'  # Ensure secure transport
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'   # Relax scope checking

# Enable OAuth debug logging
logging.getLogger('oauthlib').setLevel(logging.DEBUG)
logging.getLogger('requests_oauthlib').setLevel(logging.DEBUG)

def log_request_details(request):
    """Log detailed request information"""
    logger.debug("=== Request Details ===")
    logger.debug(f"Headers: {dict(request.headers)}")
    logger.debug(f"Host: {request.host}")
    logger.debug(f"URL: {request.url}")
    logger.debug(f"Base URL: {request.base_url}")
    logger.debug(f"Path: {request.path}")
    logger.debug(f"Query String: {request.query_string}")
    logger.debug("=====================")

def get_redirect_uri():
    """Dynamically determine the redirect URI based on request origin"""
    # Get origin from request headers
    origin = request.headers.get('Origin') or request.headers.get('Referer')
    logger.debug(f"Request origin: {origin}")
    
    if origin:
        parsed_origin = urlparse(origin)
        hostname = parsed_origin.hostname
        logger.debug(f"Parsed hostname: {hostname}")
        
        if hostname == 'goalkeeper.nearnorthanalytics.com':
            return 'https://goalkeeper-dev.onrender.com'
        elif hostname == 'goalkeeper-dev.onrender.com':
            return 'https://goalkeeper-dev.onrender.com'
    
    # Default fallback
    logger.debug("No origin found, using default redirect URI")
    return 'https://goalkeeper-dev.onrender.com'

is_deployed = os.getenv('DEPLOYED', 'False').lower() == 'true'
is_deployed = True

# Load .env variables if not deployed
if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login = []
else:
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    # redirect_uri = 'https://goalkeeper.nearnorthanalytics.com'
    redirect_uri = 'https://goalkeeper-dev.onrender.com'
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

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('oauth_debug')

def log_oauth_request(request_url, redirect_uri, scope):
    """Log details about OAuth request"""
    logger.debug("=== OAuth Request Details ===")
    logger.debug(f"Full Request URL: {request_url}")
    
    # Parse and log URL components
    parsed_url = urlparse(request_url)
    query_params = parse_qs(parsed_url.query)
    
    logger.debug(f"Scheme: {parsed_url.scheme}")
    logger.debug(f"Netloc: {parsed_url.netloc}")
    logger.debug(f"Path: {parsed_url.path}")
    logger.debug(f"Query Parameters: {query_params}")
    logger.debug(f"Configured Redirect URI: {redirect_uri}")
    logger.debug(f"Requested Scope: {scope}")
    logger.debug("===========================")

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

def create_header(is_authenticated=False, user_id="testing"):
    return dbc.Row([
        dbc.Col(html.Div(color_mode_switch +  ([" User: " + user_id] if is_authenticated else get_login), 
                className="d-flex justify-content-start"), width=3, 
                className="d-flex float-start justify-content-md-start"),
        dbc.Col(
            html.Div([
                html.H2(title, className="text-center")
            ], className="d-flex justify-content-center align-items-start h-100"), 
            width=7
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

# Update the login callback
@app.callback(
    Output('url', 'href'),
    [Input('login-button', 'n_clicks')],
)
def login_with_google(n_clicks):
    if n_clicks and is_deployed:
        try:
            logger.debug("=== Starting OAuth Flow ===")
            
            # Get dynamic redirect URI
            # redirect_uri = get_redirect_uri()
            logger.debug(f"Using dynamic redirect URI: {redirect_uri}")
            
            google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
            logger.debug(f"Session state: {google.state}")
            logger.debug(f"Session scope: {google.scope}")

            authorization_url, state = google.authorization_url(
                authorization_base_url, 
                access_type="offline", 
                prompt="select_account"
            )
            
            logger.debug(f"Generated state: {state}")
            logger.debug(f"Full authorization URL: {authorization_url}")

            return authorization_url
        except Exception as e:
            logger.error("OAuth Flow Error:", exc_info=True)
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return no_update
    return no_update

# Update the authentication callback
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
            create_header(True, auth_data.get('name')),
            dash.page_container
        ]), auth_data]

    # Handle new authentication
    if query_string:
        logger.debug("=== OAuth Callback Details ===")
        logger.debug(f"Pathname: {pathname}")
        logger.debug(f"Query String: {query_string}")
        
        # Construct and log the full callback URL
        try:
            # Get dynamic redirect URI
            redirect_uri = get_redirect_uri()
            logger.debug(f"Using dynamic redirect URI for token fetch: {redirect_uri}")
            
            google = OAuth2Session(
                client_id, 
                redirect_uri=redirect_uri
            )
            
            callback_url = f"{redirect_uri.rstrip('/')}{pathname or ''}{query_string}"
            logger.debug(f"Constructed callback URL: {callback_url}")
            
            try:
                token = google.fetch_token(
                    token_url, 
                    client_secret=client_secret,
                    authorization_response=callback_url,
                    include_client_id=True
                )
                logger.debug("Token successfully obtained")
                
            except OAuth2Error as oauth_err:
                logger.error(f"OAuth2Error during token fetch: {str(oauth_err)}")
                logger.error(f"Error description: {getattr(oauth_err, 'description', 'No description')}")
                raise
                
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            logger.debug("Successfully retrieved user info")

            return [html.Div([
                create_header(True, auth_data.get('name')),
                dash.page_container
            ]), {'authenticated': True, 'user_info': user_info}]
            
        except Exception as e:
            logger.error("Authentication error:", exc_info=True)
            logger.error(f"Full error details: {str(e)}")
            return [html.Div([
                create_header(False),
                html.Div(f"Authentication failed: {str(e)}", className="text-start text-danger")
            ]), {'authenticated': False}]
    
    return [html.Div([
        create_header(False),
        html.Div("Please login with Google to access the application.", className="text-start")
    ]), {'authenticated': False}]

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.getenv('DASH_PORT')))