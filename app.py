from dash import Dash, html, dcc, callback, Output, Input, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import find_dotenv, load_dotenv, dotenv_values
from src.keeper import video_comments_to_dataframe, get_channel_videos, create_columnDefs, ds_to_html_table, df_to_html_table, get_channel_statistics
import os, re, time, chromadb, pickle, json, traceback #, logging
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

dotenv_path = find_dotenv(raise_error_if_not_found = True)
load_dotenv(dotenv_path)

#setting the environment
myYouTube_Token=os.getenv('YOUTUBE_API_KEY')
myLangChain_Token = os.getenv('LANGCHAIN_API_KEY')
# myPineCone_Token = os.getenv('PINECONE_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#set the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#set the vectorstore db
client = chromadb.PersistentClient(path = "data")
collection = client.get_or_create_collection(name="keeper_collection", metadata={"hnsw:space": "cosine"})

vector_store_from_client = Chroma(
    client=client,
    collection_name="keeper_collection",
    embedding_function=embedding_model,
)

chat = ChatGroq(temperature = 0, groq_api_key = os.getenv('GROQ_API_KEY'), model_name = "llama3-70b-8192")
chat_history = ChatMessageHistory() #create a chat history
system = "You are a helpful life coach and expert in neurology.  All of your recommendations are backed by science.  All responses are generated in Markdown "
chat_response =" "
chat_context = ""
system_prompt = system
#set up chat app
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("minty")

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc_css], prevent_initial_callbacks='initial_duplicate')
server = app.server
myTitle = 'Welcome to the Goalkeeper'
app.title = myTitle

app.layout = dbc.Container([
    dcc.Store(id='store-response', storage_type='memory'),
    dcc.Store(id='store-context', storage_type='memory'),
    dbc.Row([
        dbc.Col([
            html.H1(myTitle, className="text-center"),
        dcc.Textarea(id='user_prompt', placeholder='Enter your prompt here...', style={'width': '100%', 'height': 200}, className='border-rounded'),
            
        ],
        width={"size":6, "offset":3},
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Submit', id='submit_button', n_clicks=0),
            dcc.Loading(
                 id="loading-response",
            # overlay_style={"visibility":"visible", "filter":"blur(2px)"},
                type="circle",
                children=html.Div(id="loading-response-div")
            ),
            html.Hr(),
            dbc.Tabs(
                [
                    dbc.Tab(label="Response", tab_id="tab-response"),
                    dbc.Tab(label="Context", tab_id="tab-context"),
                    dbc.Tab(label="System Prompt", tab_id="tab-system"),
                ],
                id='tabs',
                active_tab="tab-response",
                ),
            dcc.Markdown(id='content', children='', style={'height': '500px', 'overflowY': 'scroll', 'whiteSpace': 'pre-line'}, className="text-primary"),
            html.Div(id='debug-output'),
            html.Div(id='error-output'),
            ]),
    ]),
],
fluid = True,
class_name='dashboard-container border_rounded',

# style={'display':'flex'}
)
@callback(
    Output('store-response', 'data'),
    Output('store-context', 'data'),
    Output('loading-response-div', 'children'),
    Output('debug-output', 'children'),  # Debug output
    Input('submit_button', 'n_clicks'),
    State('user_prompt', 'value'),
    prevent_initial_call=True
)
def update_stores(n_clicks, value):
    if n_clicks > 0:
        try:
            human = value
            result = collection.query(query_texts=[human])
            context = result['documents'][0] if result['documents'] else []
            context.insert(0, "you can use the following for additional context ")
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
            chain = prompt | chat
            response = chain.invoke({"system": system, "human": human, "context": context})
            chat_response = response.content
            chat_context = "\n".join(context[1:])  # Join all context items except the first
            
            # Serialize data to JSON
            response_json = json.dumps({"response": chat_response})
            context_json = json.dumps({"context": chat_context})
            
            debug_msg = f"Updated stores - Response: {chat_response[:100]}..., Context: {chat_context[:100]}..."
            return response_json, context_json, "Query processed successfully", debug_msg
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            return (
                json.dumps({"error": "Failed to process query"}),
                json.dumps({"error": "Failed to process query"}),
                "Error occurred while processing query",
                error_msg
            )
    raise PreventUpdate

# @callback(
#     Output(component_id='content', component_property='children'),
#     Output(component_id='loading-response-div', component_property='children'),
#     Input(component_id='submit_button', component_property='n_clicks'),
#     State(component_id='user_prompt', component_property='value'),
#     prevent_initial_call = True
# )

# def update_output(n_clicks, value):
    
#     if n_clicks > 0:
#         human = value
#         result = collection.query(query_texts=human)
#         context = result['documents'].insert(0," you can use the following for additional context ")
#         prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
#         chain = prompt | chat
#         response = chain.invoke({"system":system, "human":human, "context":context})
#         chat_response = response.content
#         chat_context = result['documents'][1]
#         return chat_response, no_update
@callback(
    Output("content", "children"),
    Output("error-output", "children"),
    Input("tabs", "active_tab"),
    Input('store-response', 'data'),
    Input('store-context', 'data'),
)
def switch_tab(active_tab, stored_response, stored_context):
    try:
        # logger.debug(f"switch_tab called with active_tab: {active_tab}")
        # logger.debug(f"stored_response: {stored_response}")
        # logger.debug(f"stored_context: {stored_context}")

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        # logger.debug(f"Triggered by: {triggered_id}")

        if stored_response is None or stored_context is None:
            # logger.warning("stored_response or stored_context is None")
            return "No data available.", ""

        try:
            stored_response = json.loads(stored_response)
            stored_context = json.loads(stored_context)
        except json.JSONDecodeError as e:
            # logger.error(f"JSON decode error: {e}")
            return "Error: Invalid data in storage", str(e)

        if 'error' in stored_response or 'error' in stored_context:
            error_msg = stored_response.get('error', '') or stored_context.get('error', '')
            # logger.error(f"Error in stored data: {error_msg}")
            return f"An error occurred: {error_msg}", ""

        stored_response = stored_response.get('response', '')
        stored_context = stored_context.get('context', '')

        if triggered_id == 'tabs':
            if active_tab == "tab-response":
                return stored_response or "No response yet.", ""
            elif active_tab == "tab-context":
                return stored_context or "No context available.", ""
            elif active_tab == "tab-system":
                return system_prompt, ""
            return "Please select a tab.", ""
        elif triggered_id in ['store-response', 'store-context']:
            if active_tab == "tab-response":
                return stored_response or "No response yet.", ""
            elif active_tab == "tab-context":
                return stored_context or "No context available.", ""
            elif active_tab == "tab-system":
                return system_prompt, ""

        return "Please submit a query.", ""

    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
        # logger.error(error_msg)
        return "An error occurred. Please check the error output for details.", error_msg


# @callback(
#         Output(component_id="content", component_property="children",  allow_duplicate=True), 
#         [Input(component_id="tabs", component_property="active_tab")],
#         prevent_initial_call=True)

# def switch_tab(at):
#     if at == "tab-response":
#         return(chat_response)
#     elif at == "tab-context":
#         return(chat_context)
#     elif at == "tab-system":
#         return(system_prompt)
if __name__ == '__main__':
    app.run(jupyter_mode='_none', debug=True, port=3050)