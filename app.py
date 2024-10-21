import dash
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from requests_oauthlib import OAuth2Session
from requests import request
from dotenv import load_dotenv, find_dotenv
import os

is_deployed = os.getenv('DEPLOYED')

#Load .env variables if not deployed.
if not is_deployed:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login =""
else:
    # Your Google OAuth2 credentials
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    redirect_uri = 'https://goalkeeper.nearnorthanalytics.com'
    #OAuth2 Settings
    authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
    token_url = 'https://accounts.google.com/o/oauth2/token'
    scope = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]
    get_login = dbc.Modal([
        dbc.Button("Login with Google", id="login-button", color="success"),
        dcc.Location(id='url', refresh=True),
        html.Div(id='login-content')
    ], is_open = True)

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SKETCHY,
                                                                dbc.icons.BOOTSTRAP,
                                                                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css",
                                                                "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"],
                                                                suppress_callback_exceptions=True,
                                                                prevent_initial_callbacks=True)

                                                                
server = app.server

app.layout = ([get_login,
               dash.page_container]    
            )


@callback(Output('url', 'href'), [Input('login-button', 'n_clicks')])
def login_with_google(n_clicks):
    if n_clicks:
        if is_deployed:
            google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
            authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")
            return authorization_url
        
    return None

@callback(Output('login-content', 'children'), [Input('url', 'search')])
def display_user_info(query_string):
    if is_deployed:
        if query_string:
            google = OAuth2Session(client_id, redirect_uri=redirect_uri)
            token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            return html.Div([
                html.H1(f"Welcome, {user_info['name']}"),
                html.P(f"Email: {user_info['email']}")
            ])
        return "Please login with Google."
    else:
        ""
    return ""


    

if __name__ == '__main__':
    app.run_server(debug=True, port=os.getenv('DASH_PORT'))