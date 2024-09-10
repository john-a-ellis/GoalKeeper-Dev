import os
from typing import Dict, Any
from dash import Dash, html, dcc, callback, Output, Input, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationEntityMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, Table, MetaData, inspect
from langchain_community.chat_message_histories import SQLChatMessageHistory
import chromadb
import json
import traceback

# Load environment variables
load_dotenv(find_dotenv(raise_error_if_not_found=True))

# Initialize models and databases
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat = ChatGroq(temperature=0.7, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama3-70b-8192")
entity_memory = ConversationEntityMemory(llm=chat, k=3)

client = chromadb.PersistentClient(path="data")
collection = client.get_or_create_collection(name="keeper_collection", metadata={"hnsw:space": "cosine"})
vector_store = Chroma(client=client, collection_name="keeper_collection", embedding_function=embedding_model)

engine = create_engine("sqlite:///data/sqlite_memory.db")
metadata = MetaData()
message_store = Table('message_store', metadata, autoload_with=engine)

# Load system prompt
with open('data/system.txt', 'r') as file:
    system_prompt = file.read()

# Create conversation chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
    ("human", "Entity information: {entities}"),
    ("human", "Context: {context}"),
])

chain = (
    RunnableParallel({
        "question": lambda x: x["question"],
        "entities": lambda x: entity_memory.load_memory_variables({"input": x["question"]})["entities"],
        "context": lambda x: "\n".join([doc.page_content for doc in vector_store.similarity_search(x["question"])])
    })
    | prompt
    | chat
    | StrOutputParser()
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=engine),
    input_messages_key="question",
    history_messages_key="history"
)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY, "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"])
app.title = 'Welcome to the Goalkeeper'

# App layout
app.layout = dbc.Container([
    dcc.Store(id='store-response', storage_type='memory'),
    dcc.Store(id='store-context', storage_type='memory'),
    dcc.Store(id='store-chat-history', storage_type='memory'),
    
    dbc.Row([
        dbc.Col([
            html.H1(app.title, className="text-center"),
            dcc.Textarea(id='user_prompt', placeholder='Enter your prompt here...', style={'width': '100%', 'height': 200}, className='border-rounded'),
        ], width={"size": 6, "offset": 3})
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button('Submit', id='submit_button', n_clicks=0),
            dcc.Loading(id="loading-response", type="cube", children=html.Div(id="loading-response-div")),
            html.Hr(),
            dbc.Tabs([
                dbc.Tab(label="Response", tab_id="tab-response"),
                dbc.Tab(label="Context", tab_id="tab-context"),
                dbc.Tab(label="System Prompt", tab_id="tab-system", children=[
                    dbc.Button('Edit System Prompt', id='edit-system-prompt-button', n_clicks=0),
                    dcc.Textarea(id='system-prompt-textarea', style={'width': '100%', 'height': 200}, className='border-rounded'),
                    dbc.Button('Save System Prompt', id='save-system-prompt-button', n_clicks=0),
                ]),
                dbc.Tab(label="Chat History", tab_id="tab-chat-history", children=[
                    dbc.Button('Lobotomize Me', id='lobotomize-button', n_clicks=0),
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Lobotomy Successful")),
                        dbc.ModalBody("Who are you and what am I doing here ;-)"),
                        dbc.ModalFooter(dbc.Button("Close", id='close-modal', className="ms-auto", n_clicks=0))
                    ], id='lobotomy-modal', is_open=False),
                    html.Div(id='sqlite-memory-content'),
                ]),
            ], id='tabs', active_tab="tab-response"),
            dcc.Markdown(id='content', children='', style={'height': '500px', 'overflowY': 'scroll', 'whiteSpace': 'pre-line'}, className="text-primary"),
            html.Div(id='error-output'),
        ])
    ])
], fluid=True, className='dashboard-container border_rounded')

# Callback functions
@app.callback(
    Output('lobotomy-modal', "is_open"),
    Output('sqlite-memory-content', 'children'),
    [Input('lobotomize-button', 'n_clicks'), Input("close-modal", "n_clicks")],
    [State("lobotomy-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n1, n2, is_open):
    if n1 > 0:
        with engine.connect() as connection:
            connection.execute(message_store.delete())
            connection.commit()
        entity_memory.clear()
        sqlite_content = fetch_sqlite_memory()
        return not is_open, sqlite_content
    return is_open, no_update

@app.callback(
    Output('system-prompt-textarea', 'value'),
    Input('edit-system-prompt-button', 'n_clicks'),
    prevent_initial_call=True
)
def edit_system_prompt(n_clicks):
    if n_clicks > 0:
        return system_prompt
    return no_update

@app.callback(
    Output('content', 'children', allow_duplicate=True),
    Input('save-system-prompt-button', 'n_clicks'),
    State('system-prompt-textarea', 'value'),
    prevent_initial_call=True
)
def save_system_prompt(n_clicks, new_prompt):
    if n_clicks > 0:
        global system_prompt
        system_prompt = new_prompt
        with open('data/system.txt', 'w') as file:
            file.write(system_prompt)
        return "System prompt updated successfully."
    return no_update

@app.callback(
    Output('store-response', 'data'),
    Output('store-context', 'data'),
    Output('store-chat-history', 'data'),
    Output('content', 'children', allow_duplicate=True),
    Output('loading-response-div', 'children'),
    Input('submit_button', 'n_clicks'),
    State('user_prompt', 'value'),
    State('store-chat-history', 'data'),
    prevent_initial_call=True
)
def update_stores(n_clicks, value, chat_history):
    if n_clicks > 0:
        try:
            result = chain_with_history.invoke(
                {"question": value},
                config={"configurable": {"session_id": "default"}}
            )
            
            entity_memory.save_context({"input": value}, {"output": result})
            
            entities = entity_memory.load_memory_variables({"input": value})["entities"]
            vector_context = "\n".join([doc.page_content for doc in vector_store.similarity_search(value, k=2)])
            chat_context = f"Entities:\n{entities}\n\nRelevant Context:\n{vector_context}"
            
            chat_history = json.loads(chat_history) if chat_history else []
            chat_history.append({"human": value, "ai": result})
            
            return (
                json.dumps({"response": result}),
                json.dumps({"context": chat_context}),
                json.dumps(chat_history),
                "Query processed successfully",
                no_update
            )
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            return (
                json.dumps({"error": "Failed to process query"}),
                json.dumps({"error": "Failed to process query"}),
                json.dumps([]),
                error_msg,
                no_update
            )
    return no_update

@app.callback(
    Output("content", "children", allow_duplicate=True),
    Output("error-output", "children"),
    Input("tabs", "active_tab"),
    Input('store-response', 'data'),
    Input('store-context', 'data'),
    Input('store-chat-history', 'data'),
    prevent_initial_call=True
)
def switch_tab(active_tab, stored_response, stored_context, stored_chat_history):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        stored_response = json.loads(stored_response or '{}')
        stored_context = json.loads(stored_context or '{}')
        stored_chat_history = json.loads(stored_chat_history or '[]')
    except json.JSONDecodeError as e:
        return "Error: Invalid data in storage", str(e)

    if 'error' in stored_response or 'error' in stored_context:
        error_msg = stored_response.get('error', '') or stored_context.get('error', '')
        return f"An error occurred: {error_msg}", ""

    if triggered_id in ['tabs', 'store-response', 'store-context', 'store-chat-history']:
        if active_tab == "tab-response":
            return stored_response.get('response', 'No response yet.'), ""
        elif active_tab == "tab-context":
            return stored_context.get('context', 'No context available.'), ""
        elif active_tab == "tab-system":
            return system_prompt, ""
        elif active_tab == "tab-chat-history":
            try:
                return fetch_sqlite_memory(), ""
            except Exception as e:
                return f"Error fetching SQLite memory: {str(e)}", str(e)

    return "Please submit a query.", ""

def fetch_sqlite_memory():
    with engine.connect() as connection:
        inspector = inspect(engine)
        columns = inspector.get_columns('message_store')
        column_names = [column['name'] for column in columns]
        
        result = connection.execute(message_store.select())
        rows = result.fetchall()
    
    if not rows:
        return "No chat history available."
    
    formatted_history = "## SQLite Chat History \n\n"
    for row in rows:
        for column in column_names:
            formatted_history += f"**{column.capitalize()}:** {getattr(row, column, 'N/A')}\n"
        formatted_history += "---\n\n"
    
    return formatted_history

if __name__ == '__main__':
    app.run(debug=True, port=3050)