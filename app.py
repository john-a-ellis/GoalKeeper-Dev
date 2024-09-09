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
from langchain_core.prompts import ChatPromptTemplate #, PromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory, VectorStoreRetrieverMemory, CombinedMemory
from sqlalchemy import create_engine
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser

dotenv_path = find_dotenv(raise_error_if_not_found = True)
load_dotenv(dotenv_path)

#setting the environment
myYouTube_Token=os.getenv('YOUTUBE_API_KEY')
myLangChain_Token = os.getenv('LANGCHAIN_API_KEY')
# myPineCone_Token = os.getenv('PINECONE_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#set the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chat = ChatGroq(temperature = 0.7, groq_api_key = os.getenv('GROQ_API_KEY'), model_name = "llama3-70b-8192")

# Set up EntityMemory
entity_memory = ConversationEntityMemory(llm=chat, k=3)

#set the vectorstore db
client = chromadb.PersistentClient(path = "data")
collection = client.get_or_create_collection(name="keeper_collection", metadata={"hnsw:space": "cosine"})


vector_store_from_client = Chroma(
    client=client,
    collection_name="keeper_collection",
    embedding_function=embedding_model,
)

engine = create_engine("sqlite:///data/sqlite_memory.db")


system = """
You are a helpful life coach and expert in neuroscience. 
All of your recommendations are based on the context provided with the question and based on science. You strive to be accurate by 
reflecting on your responses and adjusting them as necessary prior to expressing them. You work to help humans achieve their goals 
in life and help keep them on track to achieve them, while making suggestions for their achievement throughout the process.
All your responses are generated in Markdown and are delivered in the 'Voice' of Mel Robbins.
You have access to two types of information:
1. Entity memory, which stores information about specific entities (people, places, concepts).
2. Vector store context, which provides relevant information based on the current question.
Use this information to provide more personalized and contextually relevant responses.
"""

system_prompt = system

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create a prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
    ("human", "Entity information: {entities}"),
    ("human", "Context: {context}"),
])
    

# Create the conversation chain
chain = (
    RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "entities": lambda x: entity_memory.load_memory_variables({"input": x["question"]})["entities"],
            "context": lambda x: "\n".join([doc.page_content for doc in vector_store_from_client.similarity_search(x["question"])]),
        }
    )
    | prompt
    | chat
    | StrOutputParser()
)

# Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id="default",
        connection=engine
    ),
    input_messages_key="question",
    history_messages_key="history",
)
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
    dcc.Store(id='store-chat-history', storage_type='memory'),

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
                    dbc.Tab(label="Chat History", tab_id="tab-chat-history"),
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
    Output('store-chat-history', 'data'),
    Output('loading-response-div', 'children'),
    # Output('debug-output', 'children'),  # Debug output
    Input('submit_button', 'n_clicks'),
    State('user_prompt', 'value'),
    State('store-chat-history', 'data'),
    prevent_initial_call=True
)

def update_stores(n_clicks, value, chat_history):
    if n_clicks > 0:
        try:
            human = value

            # Use the chain with history to get a response
            result = chain_with_history.invoke(
                {"question": human},
                config={"configurable": {"session_id": "default"}}
            )

            chat_response = result

            # Update entity memory
            entity_memory.save_context({"input": human}, {"output": chat_response})

            # Retrieve context for display
            entities = entity_memory.load_memory_variables({"input": human})["entities"]
            vector_context = "\n".join([doc.page_content for doc in vector_store_from_client.similarity_search(human)])

            chat_context = f"Entities:\n{entities}\n\nRelevant Context:\n{vector_context}"

            # Update chat history
            if chat_history is None:
                chat_history = []
            else:
                chat_history = json.loads(chat_history)
            chat_history.append({"human": human, "ai": chat_response})
            
            
            # Serialize data to JSON
            response_json = json.dumps({"response": chat_response})
            context_json = json.dumps({"context": chat_context})
            history_json = json.dumps(chat_history)

            return response_json, context_json, history_json, "Query processed successfully"#, debug_msg
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            return (
                json.dumps({"error": "Failed to process query"}),
                json.dumps({"error": "Failed to process query"}),
                json.dumps([]), # Empty chat history on error.
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
    Input('store-chat-history', 'data'),  # New input
)
def switch_tab(active_tab, stored_response, stored_context, stored_chat_history):
    try:
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if stored_response is None or stored_context is None:

            return "No data available.", ""

        try:
            stored_response = json.loads(stored_response)
            stored_context = json.loads(stored_context)
            stored_chat_history = json.loads(stored_chat_history)

        except json.JSONDecodeError as e:
            return "Error: Invalid data in storage", str(e)

        if 'error' in stored_response or 'error' in stored_context:
            error_msg = stored_response.get('error', '') or stored_context.get('error', '')
            return f"An error occurred: {error_msg}", ""

        stored_response = stored_response.get('response', '')
        stored_context = stored_context.get('context', '')

        if triggered_id == 'tabs' or triggered_id in ['store-response', 'store-context', 'store-chat-history']:
            if active_tab == "tab-response":
                return stored_response or "No response yet.", ""
            elif active_tab == "tab-context":
                return stored_context or "No context available.", ""
            elif active_tab == "tab-system":
                return system_prompt, ""
            elif active_tab == "tab-chat-history":
                #format chat history
                formatted_history = "## Chat History \n\n"
                for entry in stored_chat_history:
                    formatted_history += f"**Human:** {entry['human']}\n\n"
                    formatted_history += f"**AI:** {entry['ai']}\n\n"
                    formatted_history += "---\n\n"
                return formatted_history or "No chat history available.",""
   
        return "Please submit a query.", ""

    except Exception as e:
        return "An error occurred. Please check the error output for details.", error_msg



if __name__ == '__main__':
    app.run(jupyter_mode='_none', debug=True, port=3050)