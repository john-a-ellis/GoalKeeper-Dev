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
import os, re, time, chromadb, pickle, json #, traceback, logging
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import SQLChatMessageHistory

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
chat_message_history = SQLChatMessageHistory("test_session_id", connection="sqlite:///data/sqlite_memory.db") #create a chat history repo

system = "You are a helpful life coach and expert in neuro-science.  All of your recommendations are based on the context provided with the question.  All responses are generated in Markdown and are delivered in the 'Voice' of Mel Robbins"
chat_response =" "
chat_context = ""
system_prompt = system

#set up chat app
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("sketchy")

app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY, dbc_css], prevent_initial_callbacks='initial_duplicate')
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
    # Output('debug-output', 'children'),  # Debug output
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
            # add interaction to chat history
            # chat_message_history.add_user_message(prompt)
            # chat_message_history.add_ai_message(response)

            return response_json, context_json, "Query processed successfully" #, debug_msg
        except Exception as e:
            # error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            return (
                json.dumps({"error": "Failed to process query"}),
                json.dumps({"error": "Failed to process query"}),
                "Error occurred while processing query",
                error_msg
            )
    raise PreventUpdate


@callback(
    Output("content", "children"),
    Output("error-output", "children"),
    Input("tabs", "active_tab"),
    Input('store-response', 'data'),
    Input('store-context', 'data'),
)
def switch_tab(active_tab, stored_response, stored_context):
    try:
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if stored_response is None or stored_context is None:

            return "No data available.", ""

        try:
            stored_response = json.loads(stored_response)
            stored_context = json.loads(stored_context)
        except json.JSONDecodeError as e:
            return "Error: Invalid data in storage", str(e)

        if 'error' in stored_response or 'error' in stored_context:
            error_msg = stored_response.get('error', '') or stored_context.get('error', '')
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
        return "An error occurred. Please check the error output for details.", error_msg



if __name__ == '__main__':
    app.run(jupyter_mode='_none', debug=True, port=3050)