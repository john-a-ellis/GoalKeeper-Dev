#app.py
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv, find_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from src.custom_modules import read_prompt
from urllib.parse import urlparse, parse_qs
from oauthlib.oauth2.rfc6749.errors import OAuth2Error
import logging
import sys
from src.custom_modules import get_user_id

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)    

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'  # Ensure secure transport
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'   # Relax scope checking

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")
# llm = init_chat_model(model="gpt-4o", model_provider="openai")
# llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-70b-versatile")
# llm = OpenAI()

def get_redirect_uri():
    """Dynamically determine the redirect URI based on request origin"""
    return 'https://goalkeeper.nearnorthanalytics.com'
    # return 'http://localhost:3050'

title = 'Welcome to the Goalkeeper'
is_deployed = os.getenv('DEPLOYED', 'False').lower() == 'true'
# is_deployed = True
is_authenticated = False

welcome_prompt = read_prompt('welcome_prompt')
welcome_message = llm.invoke(welcome_prompt).content

# Load .env variables if not deployed
if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    os.environ["LANGCHAIN_PROJECT"] = "goalkeeper"
    get_login = []
    get_logout = []
else:
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    redirect_uri = get_redirect_uri()
    # redirect_uri = 'https://goalkeeper-dev.onrender.com'
    # OAuth2 Settings
    authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
    token_url = 'https://accounts.google.com/o/oauth2/token'
    scope = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]

    get_login = html.Div([
        dbc.Button("Login", id="login-button", color="success", size="sm"),
        dcc.Location(id='url', refresh=True)
    ], className="align-middle")
    get_logout = [
        dbc.Button("Logout", id="logout-button", color="success", size="sm"),
    ]
app = dash.Dash(__name__, 
                use_pages=True, 
                title = 'The Goalkeeper', 
                update_title = title, 
                external_stylesheets=[dbc.themes.SKETCHY,
                                        dbc.icons.BOOTSTRAP,
                                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css",
                                        "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
                                    ],
                suppress_callback_exceptions=True,
                prevent_initial_callbacks=True
                )

server = app.server

def construct_callback_url(redirect_uri, pathname, query_string):
    base = redirect_uri.rstrip('/')
    path = pathname.lstrip('/') if pathname else ''
    return f"{base}/{path}{query_string}"

color_mode_switch = html.Div([
    dbc.Label(className="fa-solid fa-moon", html_for="theme-switch"),
    dbc.Switch(id="theme-switch", value=True, className="d-inline-block ms-1", persistence=True),
    dbc.Label(className="fa-regular fa-sun", html_for="theme-switch"),
], className="d-flex align-items-center me-3")


def create_header(is_authenticated=False, user_info="default"):
    # Display user info based on authentication state and deployment status
    user_display = html.Div(
        [
            html.I(className="fas fa-user me-2"),
            html.Span(
                "testing" if not is_deployed else (
                    user_info if is_authenticated else "Not Logged in"
                ),
                className="text-muted", id='login-span'
            ),
            dbc.Tooltip("Logout", target='login-span')
        ],
        id='login-name', className="d-flex align-items-center me-3"
    )

    return dbc.Row([
        dbc.Col(
            [html.A(html.Img(src='assets/NearNorthCleanRight.png', height=100),href='https://nearnorthanalytics.com', target='_blank'),
             color_mode_switch, 
            html.Div(
                children=(get_logout if is_authenticated else get_login), 
                className = "d-flex align-items-center me-3 float-start")],
                width=3, className="d-grid gap-2 d-md-flex justify-content-md-start"),
        dbc.Col(
            html.Div([
                html.H2(title, className="text-center", id='title'),
            ], className="d-flex justify-content-center align-items-start h-100"), 
            width=6
        ),
        dbc.Col([
            html.Div([
                # User Display Component
                user_display,
                dbc.Button(
                    size="md",
                    id="feedback-button",
                    n_clicks=0,
                    class_name="ml-auto fa-solid fa-comment-dots",
                    style={"background-color":"blue",
                           "border-color":"blue"}
                    # color="secondary"
                ),
                dbc.Tooltip(
                    "Leave us some feedback!",
                    target="feedback-button",
                    id="feedback-button-tooltip"
                ),
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
                    style={"background-color":"darkturquoise",
                           "border-color":"darkturquoise"} 
                ),
                dbc.Tooltip(
                    "About",
                    target="about-button",
                    id="about-button-tooltip"
                ),
            ])
        ], className="d-grid gap-2 d-md-flex justify-content-md-end"),

    ], className="align-items-start")
def create_content_row(deployed):
    if deployed:
        
        return dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Response", tab_id="tab-response", active_label_style={"color":"gray"} ),
                ], id='tabs', active_tab="tab-response"),
                html.Div(id='content', 
                        children=dbc.Card(dbc.CardBody([
                            # html.H4('Now in Open Beta', className="card-title"),
                            dcc.Markdown(welcome_message, id="card-summary")])), 
                        style={
                    'height': '475px', 
                    'overflowY': 'auto', 
                    'whiteSpace': 'pre-line'
                    }, 
                    className="text-primary"),
                html.Div(id='error-output'),
            ], width={"size": 12}),
        ], justify="end")
    else:
        return dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Response", tab_id="tab-response", active_label_style={"color":"gray"} ),
                ], id='tabs', active_tab="tab-response"),
                html.Div(id='content', 
                        children=dbc.Card(dbc.CardBody([html.H4(title, className="card-title"),dcc.Markdown("", id="card-summary")])), 
                        style={
                    'height': '475px', 
                    'overflowY': 'auto', 
                    'whiteSpace': 'pre-line'
                    }, 
                    className="text-primary"),
                html.Div(id='error-output'),
            ], width={"size": 12}),
        ], justify="end")

app.layout = dbc.Container([
    dcc.Loading(id="loading-response",
                type="cube", 
                children=html.Div(id="loading-response-div"), 
                target_components={"card-summary":"children"},
                # custom_spinner=html.P("Hmmm... Recalling our discussions")
                ),
    # Store for authentication state    
    dcc.Store(id='auth-store', storage_type='session'),
    dcc.Location(id='url', refresh=True),
    dcc.Store(id='reddit-ad-store', storage_type='memory'),

    html.Div(id='page-content'),
    
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(html.A(html.Img(src='assets/NearNorthCleanRight.png', height=100), href='https://nearnorthanalytics.com', target='_blank'))),
            dbc.ModalBody('This is a test', id='TOS-body'),
        ],
        id='TOS-modal',
        fullscreen=True,
        autofocus=True
    ),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(html.A(html.Img(src='assets/NearNorthCleanRight.png', height=100), href='https://nearnorthanalytics.com', target='_blank'))),
            dbc.ModalBody(children ='', id='PP-body'),
        ],
        id='PP-modal',
        fullscreen=True,
        autofocus=True
    ),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(html.A(html.Img(src='assets/NearNorthCleanRight.png', height=100), href='https://nearnorthanalytics.com', target='_blank'))),
            dbc.ModalBody(children ='', id='FAQ-body'),
        ],
        id='FAQ-modal',
        fullscreen=True,
        autofocus=True
    ),
    dbc.Row([
        dbc.Col([html.Div(html.Span(children = 'Terms of Service', id='TOS-span', n_clicks=0), className ='text-end align-text-bottom')]),
        dbc.Col([html.Div(html.Span(children = 'FAQ', id='FAQ-span', n_clicks=0), className ='text-center align-text-bottom')]),
        dbc.Col([html.Div(html.Span(children = 'Privacy Policy', id='PP-span', n_clicks=0), className ='text-start align-text-bottom')]),
    ], class_name="fixed-top"),
], fluid=True, 
className='m-0 dashboard-container border_rounded min-vh-75', 
id='main-container', 
# style={'height': '850px'}
)

#Terms of Serivice Callback
@app.callback(
    Output(component_id='TOS-modal', component_property='is_open'),
    Output(component_id='TOS-body', component_property='children'),  
    Input(component_id='TOS-span', component_property='n_clicks'),
    State(component_id='TOS-modal', component_property='is_open'),
    prevent_initial_call = False
)
def show_TOS(tos_clicks, tos_open):
    if tos_clicks > 0:
        with open('assets/terms-of-service.md', 'r') as file:
            tos = file.read()
            tos_open = True
        return (tos_open, dcc.Markdown(tos))
    return (no_update, no_update)

#FAQ Callback
@app.callback(
    Output(component_id='FAQ-modal', component_property='is_open'),
    Output(component_id='FAQ-body', component_property='children'),  
    Input(component_id='FAQ-span', component_property='n_clicks'),
    State(component_id='FAQ-modal', component_property='is_open'),
    prevent_initial_call = False
)
def show_FAQ(FAQ_clicks, FAQ_open):
    if FAQ_clicks > 0:
        with open('assets/faq.md', 'r') as file:
            FAQ = file.read()
            FAQ_open = True
        return (FAQ_open, dcc.Markdown(FAQ))
    return (no_update, no_update)


#Privacy Policy Callback
@app.callback(
    Output(component_id='PP-modal', component_property='is_open'),
    Output(component_id='PP-body', component_property='children'),  
    Input(component_id='PP-span', component_property='n_clicks'),
    prevent_initial_call = 'initial_duplicate'
)
def show_PP(pp_clicks):
    if pp_clicks > 0:
        with open('assets/privacy-policy.md', 'r') as file:
            pp = file.read()
        return (True, dcc.Markdown(pp))
    return (no_update, no_update)

#Theme Switch Callback
@app.callback(
        Output('main-container', 'style', allow_duplicate=True), 
        Input('theme-switch', 'value')
)
def set_background(switch_value):
    return { 
        'background-color': 'darkslategray' if switch_value == False else 'aliceblue',
        'border-radius': '10px',
        }

app.clientside_callback("""function (theme_color) {
                        if (theme_color) { 
                            document.body.style.backgroundColor = '#63efdf'
                         } 
                        else {
                            document.body.style.backgroundColor = 'slategray'
                            }
                         }
                         return '';
                         """, 
                        Output('theme-switch', 'input_class_name', allow_duplicate=True), 
                        Input('theme-switch', 'value')
                        )
# login callback
@app.callback(
    Output('url', 'href'),
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

            return authorization_url, True
        except Exception as e:
            return no_update, False
    return no_update, False

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
        create_content_row(is_deployed)
        return "Not Logged in", True, get_redirect_uri()
    else:
        return no_update, False, no_update
@app.callback(
        # Output('login-span', 'children'),
        Output('auth-store', 'clear_data', allow_duplicate=True),
        Output('url', 'href', allow_duplicate=True),
        Input('logout-button', 'n_clicks'),
        # Input('login-span', 'children'),
        prevent_initial_call = True
        
)
def logout_button(clicked):
    if clicked and is_deployed:
        {'authenticated': False}
        create_header(),
        create_content_row(is_deployed)
        return True, get_redirect_uri()
    else:
        return False, no_update    
    
# authentication callback
@app.callback(
    Output('page-content', 'children'),
    Output('auth-store', 'data'),
    Output('reddit-ad-store', 'data'),
    Input('url', 'pathname'),
    Input('url', 'search'),
    State('auth-store', 'data'),
    prevent_initial_call = True
)
def update_page_content(pathname, query_string, auth_data):
    # print(f"this is the query string: {query_string}")
    if not is_deployed:
        # For local development, show everything without authentication
        return [html.Div([
            create_header(True),  # Will display "testing"
            create_content_row(is_deployed),
            dash.page_container
        ]), {'authenticated': True}, no_update]
    
    # Check if already authenticated
    if auth_data and auth_data.get('authenticated'):
        user_email = get_user_id(auth_data)
        print(f"this is the auth_data: {auth_data}")
        return [html.Div([
            create_header(True, user_email),
            create_content_row(is_deployed),
            dash.page_container
        ]), auth_data, no_update]


    # Handle new authentication
    if query_string:
        query_params = dict(x.split('=') for x in query_string.lstrip('?').split('&'))
        ad_param = query_params.get('ad')

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
            print(f"this is the user_infor: {user_info}")
            user_email = user_info.get('email', 'User')

            return [html.Div([
                create_header(True, user_email),
                create_content_row(is_deployed),
                dash.page_container
            ]), {'authenticated': True, 'user_info': user_info}, {'ad':ad_param}]
            
        except Exception as e:

            return [html.Div([
                create_header(False),  # Will display "not logged in"
                create_content_row(is_deployed),
                # html.Div(f"Authentication failed: {str(e)}", className="text-danger")
            ]), {'authenticated': False}, {'ad':ad_param}]
    
    return [html.Div([
        create_header(False),  # Will display "not logged in"
        create_content_row(is_deployed),
    ]), {'authenticated': False}, {'ad':None}]

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.getenv('DASH_PORT')))