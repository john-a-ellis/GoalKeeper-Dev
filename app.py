import os
from typing import Dict, Any, List
from dash import Dash, html, dcc, callback, Output, Input, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationEntityMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Text, JSON, DateTime, select, inspect
from langchain_community.chat_message_histories import SQLChatMessageHistory
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
import json
import traceback
from datetime import datetime

user_id='default'

# Load environment variables
load_dotenv(find_dotenv(raise_error_if_not_found=True))

# Initialize models and databases
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chat = ChatGroq(temperature=0.1, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama3-70b-8192")
entity_memory = ConversationEntityMemory(llm=chat, k=3)

#initialize vector stores
context_vector_store = Chroma(collection_name="keeper_collection", embedding_function=embedding_model, persist_directory="data")
memory_vector_store = Chroma(collection_name="memory_collection", embedding_function=embedding_model, persist_directory="data")
# initialize Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return list(result)

neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Define the output structure for LLM-based entity extraction
class EntityExtraction(BaseModel):
    goals: List[str] = Field(description="List of identified goals")
    values: List[str] = Field(description="List of identified values")
    plans: List[str] = Field(description="List of identified plans")
    actions: List[str] = Field(description="List of identified actions")
# Create a parser for LLM output
parser = PydanticOutputParser(pydantic_object=EntityExtraction)

# Craft the prompt for LLM-based entity extraction

prompt_template = """
Analyze the following text and identify any goals, values, plans, and actions mentioned. 
Provide your answer in the following format:

goals:
- [List of goals]
values:
- [List of values]
plans:
- [List of plans]
actions:
- [List of actions]

If there are no items for a category, leave the list empty.

Text to analyze: {{input_text}}
"""


# initialize SQLite stores
engine = create_engine("sqlite:///data/sqlite_memory.db")
metadata = MetaData()
message_store = Table('message_store', metadata, autoload_with=engine)
# personal_info = Table('personal_info', metadata, autoload_with=engine)
# Load system prompt
with open('data/system.txt', 'r') as file:
    system_prompt = file.read()

def extract_entities_llm(text: str) -> EntityExtraction:
    # Use double curly braces in the format method to escape them
    full_prompt = prompt_template.format(input_text=text)
    
    # Use invoke instead of predict
    output = chat.invoke(full_prompt).content
    
    # Log the raw output for debugging
    print("Raw output from language model:")
    print(output)
    print("--------------------")
    
    # Parse the output
    parsed_output = {
        "goals": [],
        "values": [],
        "plans": [],
        "actions": []
    }
    current_key = None
    for line in output.split('\n'):
        line = line.strip()
        if line.endswith(':'):
            current_key = line[:-1].lower()
        elif current_key and line.startswith('-'):
            if current_key in parsed_output:
                parsed_output[current_key].append(line[1:].strip())
    
    return EntityExtraction(**parsed_output)

def update_entity_info(user_id: str, entities: EntityExtraction):
    for entity_type in ['goals', 'values', 'plans', 'actions']:
        for entity in getattr(entities, entity_type):
            query = (
                "MERGE (u:User {id: $user_id}) "
                f"MERGE (e:{entity_type.capitalize()} {{text: $entity_text}}) "
                "MERGE (u)-[:HAS_ENTITY]->(e)"
            )
            neo4j_conn.run_query(query, {
                "user_id": user_id,
                "entity_text": entity
            })

def get_user_entities(user_id: str) -> Dict[str, List[str]]:
    query = (
        "MATCH (u:User {id: $user_id})-[:HAS_ENTITY]->(e) "
        "RETURN labels(e)[0] as type, e.text as text"
    )
    results = neo4j_conn.run_query(query, {"user_id": user_id})
    entities = {
        "goals": [],
        "values": [],
        "plans": [],
        "actions": []
    }
    
    for record in results:
        entity_type = record["type"].lower()
        if entity_type in entities:
            entities[entity_type].append(record["text"])
    return entities

def fetch_user_entities(user_id: str) -> Dict[str, List[str]]:
    query = (
        "MATCH (u:User {id: $user_id})-[:HAS_ENTITY]->(e) "
        "RETURN labels(e)[0] as type, e.text as text"
    )
    results = neo4j_conn.run_query(query, {"user_id": user_id})
    entities = {
        "goals": [],
        "values": [],
        "plans": [],
        "actions": []
    }
    
    for record in results:
        entity_type = record["type"].lower()
        if entity_type in entities:
            entities[entity_type].append(record["text"])
    return entities

def update_vector_memory(user_id: str, content: str):
    memory_vector_store.add_texts(
        texts=[content],
        metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat()}]
    )

def retrieve_vector_memory(user_id: str, query: str, k: int = 4):
    results = memory_vector_store.similarity_search(
        query=query,
        k=k,
        filter={"user_id": user_id}
    )
    return [doc.page_content for doc in results]
    
def get_cross_session_memory(question: str) -> str:
    with engine.connect() as connection:
        # Fetch recent messages from all sessions
        stmt = select(message_store).order_by(message_store.c.id.desc()).limit(10)
        result = connection.execute(stmt)
        messages = result.fetchall()
    
    # Format messages into a string
    memory_string = "Recent conversations:\n"
    for msg in messages:
        try:
            message_data = json.loads(msg.message)
            message_type = message_data.get('type')
            content = message_data.get('data', {}).get('content', '')
            
            memory_string += f"Message ID: {msg.id}\n"
            memory_string += f"Session ID: {msg.session_id}\n"
            memory_string += f"Type: {message_type.capitalize()}\n"
            memory_string += f"Content: {content}\n"
            
            # Extract additional information if available
            additional_kwargs = message_data.get('data', {}).get('additional_kwargs', {})
            if additional_kwargs:
                memory_string += "Additional Information:\n"
                for key, value in additional_kwargs.items():
                    memory_string += f"  - {key}: {value}\n"
            
            # Add timestamp if available (you might need to add this to your database schema)
            timestamp = msg.timestamp if hasattr(msg, 'timestamp') else datetime.now()
            memory_string += f"Timestamp: {timestamp}\n"
            
            memory_string += "\n" + "-"*50 + "\n\n"

        except json.JSONDecodeError:
            print(f"Error decoding JSON for message ID {msg.id}")
    
    return memory_string

def print_table_schema():
    with engine.connect() as connection:
        inspector = inspect(engine)
        columns = inspector.get_columns('message_store')
        print("Current 'message_store' table schema:")
        for column in columns:
            print(f"Column: {column['name']}, Type: {column['type']}")

def get_memory_context(user_id: str, question: str):
    # personal_info = get_personal_info(user_id)
    vector_memory = retrieve_vector_memory(user_id, question)
    user_entities = get_user_entities(user_id)
    return f"""
Relevant Past Interactions:
{' '.join(vector_memory)}


User Entities:
{json.dumps(user_entities, indent=2)}
"""
# Personal Info: {json.dumps(personal_info, indent=2)}

# Create conversation chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
    ("human", "Memory Context: {memory_context}"),
    ("human", "Entity information: {entities}"),
    ("human", "Context: {context}")
    # ("human", "Cross-session memory: {cross_session_memory}")
])


chain = (
    RunnableParallel({
        "question": lambda x: x["question"],
        "memory_context": lambda x: get_memory_context(x["user_id"], x["question"]),
        "entities": lambda x: entity_memory.load_memory_variables({"input": x["question"]})["entities"],
        "context": lambda x: "\n".join([doc.page_content for doc in context_vector_store.similarity_search(x["question"])]),
        "llm_entities": lambda x: extract_entities_llm(x["question"]) if x["question"] else EntityExtraction(goals=[], values=[], plans=[], actions=[])
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
def get_structured_chat_history() -> str:
    with engine.connect() as connection:
        stmt = select(message_store).order_by(message_store.c.id.desc()).limit(20)  # Fetching last 20 messages
        result = connection.execute(stmt)
        messages = result.fetchall()
    
    history_components = []
    for msg in messages[::-1]:  # Reverse the order to show oldest first
        try:
            message_data = json.loads(msg.message)
            message_type = message_data.get('type')
            content = message_data.get('data', {}).get('content', '')
            
            message_component = f"""
Message ID: {msg.id}
Session ID: {msg.session_id}
Type: {message_type.capitalize()}
Content: {content}

---

"""            
            history_components.append(message_component)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for message ID {msg.id}")
    
    return "".join(history_components) if history_components else "No chat history available"

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY, "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"])
# Call this function when your app starts
print_table_schema()
app.title = 'Welcome to the Goalkeeper'

# App layout
app.layout = dbc.Container([
    dcc.Store(id='store-response', storage_type='memory'),
    dcc.Store(id='store-context', storage_type='memory'),
    dcc.Store(id='store-chat-history', storage_type='memory'),
    dcc.Store(id='store-entity-memory', storage_type='memory'),
    # dbc.Button('Test Components', id='test-components-button', n_clicks=0),
    
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
                dbc.Tab(label="Entities", tab_id="tab-entities"),
                dbc.Tab(label="System Prompt", tab_id="tab-system", children=''),
                dbc.Tab(label="Chat History", tab_id="tab-chat-history", children=''),
            ], id='tabs', active_tab="tab-response"),
            html.Div(id='content', children='', style={'height': '500px', 'overflowY': 'scroll', 'whiteSpace': 'pre-line'}, className="text-primary"),
            html.Div(id='error-output'),
        ])
    ])
], fluid=True, className='dashboard-container border_rounded')

# Callback functions
@app.callback(
    Output('lobotomy-modal', "is_open"),
    Output('loading-response-div', 'children'),
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
        memory_vector_store.delete_collection("memory_collection")
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
    Output('store-response', 'data', allow_duplicate=True),
    Output('store-context', 'data', allow_duplicate=True),
    Output('store-chat-history', 'data', allow_duplicate=True),
    Output('store-entity-memory', 'data', allow_duplicate=True),
    Output('content', 'children', allow_duplicate=True),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input('submit_button', 'n_clicks'),
    State('user_prompt', 'value'),
    State('store-chat-history', 'data'),
    prevent_initial_call=True
)
def update_stores(n_clicks, value, chat_history, user_id="default"):
   
    if n_clicks > 0:
        try:
            # Retrieve context from transcript vector store
            vector_context = "\n".join([doc.page_content for doc in context_vector_store.similarity_search(value, k=4)])
            # Extract entities using LLM
            llm_entities = extract_entities_llm(value)
            # Update entity info in Neo4j
            update_entity_info(user_id, llm_entities)
            # Get all user entities from Neo4j
            user_entities = fetch_user_entities(user_id)  # Use the new function name here
            
            result = chain_with_history.invoke(
                {"question": value, 
                 "user_id": user_id,
                 "llm_entities": llm_entities                 
                 },
                config={"configurable": {"session_id": user_id}}
            )
            # Update vector memory with the new interaction
            update_vector_memory(user_id, f"Human: {value}\nAI: {result}")
            entity_memory.save_context({"input": value}, {"output": result})
            
            # Get memory context (includes personal info and relevant past interactions)
            memory_context = get_memory_context(user_id, value)

            # Combine all context information
            full_context = f"""
                            Vector Store Context:
                            {vector_context}

                            Memory Context:
                            {memory_context}

                            Entities:
                            {entity_memory.load_memory_variables({"input": value})["entities"]}
                            """
         
            
            chat_history = json.loads(chat_history) if chat_history else []
            chat_history.append({"human": value, "ai": result})
            
            return (
                json.dumps({"response": result}),
                json.dumps({"context": full_context}),
                json.dumps(chat_history),
                json.dumps({"entities": entity_memory.load_memory_variables({"input": value})["entities"]}),
                "Query processed successfully",
                no_update
            )
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # This will print to your console or logs
            return (
                json.dumps({"error": f"Failed to process query: {str(e)}"}),
                json.dumps({"error": f"Failed to process query: {str(e)}"}),
                json.dumps([]),
                no_update,
                error_msg,
                no_update
            )
    return no_update, no_update, no_update, no_update, no_update, no_update

@app.callback(
    Output("content", "children", allow_duplicate=True),
    Output("error-output", "children", allow_duplicate=True),
    Input("tabs", "active_tab"),
    Input('store-response', 'data'),
    Input('store-context', 'data'),
    Input('store-chat-history', 'data'),
    Input('store-entity-memory', 'data'),
    prevent_initial_call=True
)
def switch_tab(active_tab, stored_response, stored_context, stored_chat_history, stored_entities):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        stored_response = json.loads(stored_response or '{}')
        stored_context = json.loads(stored_context or '{}')
        stored_chat_history = json.loads(stored_chat_history or '[]')
        stored_entities = json.loads(stored_entities or '{}')
    except json.JSONDecodeError as e:
        return "Error: Invalid data in storage", str(e)

    if 'error' in stored_response or 'error' in stored_context:
        error_msg = stored_response.get('error', '') or stored_context.get('error', '')
        return f"An error occurred: {error_msg}", ""

    if triggered_id in ['tabs', 'store-response', 'store-context', 'store-chat-history','store-entity-memory']:
        if active_tab == "tab-response":
            return dcc.Markdown(str(stored_response.get('response', 'No response yet.'))), ""
        elif active_tab == "tab-context":
            return str(stored_context.get('context', 'No context available.')), ""
        elif active_tab == "tab-system":
            this=[
                    dbc.Button('Edit System Prompt', id='edit-system-prompt-button', n_clicks=0),
                    dcc.Textarea(id='system-prompt-textarea', style={'width': '100%', 'height': 200}, className='border-rounded'),
                    dbc.Button('Save System Prompt', id='save-system-prompt-button', n_clicks=0),
                    html.Br()
                ]
            this.append(system_prompt)
            return this, ""
        elif active_tab == "tab-chat-history":
            this = [
                    dbc.Button('Lobotomize Me', id='lobotomize-button', n_clicks=0),
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Lobotomy Successful")),
                        dbc.ModalBody("Who are you and what am I doing here ;-)"),
                        dbc.ModalFooter(dbc.Button("Close", id='close-modal', className="ms-auto", n_clicks=0))
                    ], id='lobotomy-modal', is_open=False),
                    html.Div(id='sqlite-memory-content'),
                ]
            this.append(dcc.Markdown(str(get_structured_chat_history())))
            return this, ""
        elif active_tab=="tab-entities":
            entities = stored_context.get('entities', {})
            entity_display = []
            for entity_type, entity_list in entities.items():
                entity_display.append(html.H3(entity_type.capitalize()))
                entity_display.append(html.Ul([html.Li(entity) for entity in entity_list]))
            return html.Div(entity_display), ""
        
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