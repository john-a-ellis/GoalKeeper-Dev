import os, json, traceback, uuid, strip_markdown
from typing import Dict, List
from dash import Dash, html, dcc, Output, Input, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.memory import ConversationEntityMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from pydantic import BaseModel, Field
import networkx as nx
import plotly.graph_objects as go
import traceback
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("sketchy")

user_id='default'
today = datetime.now()
# Load environment variables
load_dotenv(find_dotenv(raise_error_if_not_found=True))
# Setup Langchain Tracing
# LANGCHAIN_TRACING_V2 = True
os.environ["LANGCHAIN_PROJECT"] = "goalkeeper"

# Initialize models and databases
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chat = ChatGroq(temperature=0.7, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-70b-Versatile")
entity_memory = ConversationEntityMemory(llm=chat, k=3)

# initialize Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

#initialize vector stores

context_vector_store = Neo4jVector.from_existing_index(
    embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="vector",
)

memory_vector_store = Neo4jVector.from_existing_index(
    embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="message_vector",
)
#initialize Graph Database
graph_database = Neo4jGraph(url=NEO4J_URI,
                            username=NEO4J_USERNAME,
                            password=NEO4J_PASSWORD)

#intialize Graph Database Transformer
allowed_nodes = ["Value", 
                "Goal", 
                "Plan", 
                "Action", 
                "Obstacle", 
                "Solution", 
                "Person", 
                "Place", 
                "Skill", 
                "Organization", 
                "Age", 
                "BirthDate"]

allowed_relationships = ['IN_PLAN',
                        'HAS_VALUE', 
                        'HAS_GOAL', 
                        'HAS_PLAN', 
                        'LIVES_IN', 
                        'WORKS_AT', 
                        'WORKED_AT', 
                        'HAS_SKILL', 
                        'HAS_OBSTACLE', 
                        'HAS_SOLUTION']

# Craft the prompt for LLM-based entity extraction
graph_transformer_prompt_template = PromptTemplate(input_variables=["allowed_nodes", "allowed_relationships"],
                                                   template ="""
You are a data analyst expert in the use of Neo4j for storage and retrieval.
Analyze the following text and identify the following as nodes: {{allowed_nodes}} mentioned 
and store them in the Neo4j database.  When creating nodes for Goals they must be based on value nodes, 
plan nodes must relate to at least one goal node, action nodes must belong to a plan node, obstacle nodes can 
only be related to goal nodes or plan nodes, solution nodes most relate to obstacle nodes.  
These are the allowed identifiers for these relationships: {{allowed_relationships}}.
""")
graph_transformer = LLMGraphTransformer(llm = chat,
                                        prompt = graph_transformer_prompt_template,
                                        allowed_nodes = allowed_nodes,
                                        allowed_relationships = allowed_relationships,
                                        strict_mode = False,
                                        relationship_properties = True,
                                        node_properties = True
                                      )

class Neo4jConnection:
    def __init__(self, url, user, password):
        self._driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return list(result)

neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)


class ShortTermMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int = 5):
        return self.messages[-limit:]

    def clear(self):
        self.messages = []

# Initialize short-term memory
short_term_memory = ShortTermMemory()


# Load system prompt
with open('data/system.txt', 'r') as file:
    system_prompt = file.read()

#collect data for Entity Relationship Plot
def get_graph_data(url, user, password):
    driver = GraphDatabase.driver(url, auth=(user, password))
    with driver.session() as session:
        result = session.run("""
MATCH (n:!Chunk)-[r]->(m) 
        RETURN id(n) AS source, id(m) AS target, 
               labels(n) AS source_labels, labels(m) AS target_labels,
               type(r) AS relationship_type, n.id as id  
        """)
        return [record for record in result]


def update_graph_memory( user_id: str, content: str, type:str):
    this = strip_markdown.strip_markdown(content)
    document = Document(page_content=this, metadata={"source":type,
                                                    "user":user_id,
                                                    "id": None})
    graph_document = graph_transformer.process_response(document)
    print(graph_document)
    graph_database.add_graph_documents(
            [graph_document],
            baseEntityLabel=False,
            include_source=True
)

# Function to create the network entity graph
def create_network_graph(url, username, password):
    graph_data = get_graph_data(url, username, password)
    G = nx.Graph()
    for record in graph_data:
        G.add_edge(record['source'], 
                record['target'], 
                attr={"source_label":record['source_labels'][0],
                        "target_label":record['target_labels'][0],
                        "id":record['id'],
                        "relationship_type":record['relationship_type']
                    }
        )   
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    edge_text = []
    edge_label_x = []
    edge_label_y = []
    edge_labels = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        relationship_type = edge[2]['attr']['relationship_type']
        edge_info = f"Relationship Type: {relationship_type}"
        edge_text.append(edge_info)
        
        # Calculate midpoint for edge label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_label_x.append(mid_x)
        edge_label_y.append(mid_y)
        edge_labels.append(relationship_type)

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Collect node information for hover text
        node_info = f"Node ID: {node}<br>"
        edges = G.edges(node, data=True)
        if edges:
            edge = next(iter(edges))  # Get the first edge
            if node == edge[0]:  # This node is the source
                node_info += f"Label: {edge[2]['attr']['source_label']}<br>"
            else:  # This node is the target
                node_info += f"Label: {edge[2]['attr']['target_label']}<br>"
            node_info += f"Edge ID: {edge[2]['attr']['id']}<br>"
        else:
            node_info += "No connected edges"
        node_text.append(node_info)

    # Step 3: Create the network plot
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Turbo',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Add edge labels
    edge_labels_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode='text',
        text=edge_labels,
        textposition='middle center',
        textfont=dict(size=10, color='black'),
        hoverinfo='none'
    )

    # Color nodes based on number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace, edge_labels_trace],
                    layout=go.Layout(
                        title='Relationships Identified from Conversations',
                        titlefont_size=16,
                        plot_bgcolor="aliceblue",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                    )

    return fig

def update_vector_memory(user_id: str, content: str, type: str):

    def generate_unique_id():
        while True:
            message_id = str(uuid.uuid4())
            # Check if the ID already exists
            check_query = "MATCH (m:Message {id: $id}) RETURN count(m) AS count"
            result = neo4j_conn.run_query(check_query, {"id": message_id})
            if result[0]['count'] == 0:
                return message_id
    message_id = generate_unique_id()

    memory_vector_store.add_texts(
        texts=[content],
        metadatas=[{"user_id": user_id, "type":type, "timestamp": datetime.now().isoformat()}],
        ids = [message_id]
    )

def retrieve_vector_memory(user_id: str, query: str, k: int = 4):
    results = memory_vector_store.similarity_search(
        query=query,
        k=k,
        filter={"user_id": user_id}
    )
    return [doc.page_content for doc in results]

def get_memory_context(user_id: str, question: str):
    long_term_memory = retrieve_vector_memory(user_id, question)
    recent_messages = short_term_memory.get_recent_messages()
    # user_entities = get_user_entities(user_id)
    
    return f"""
Long-term Memory (from previous conversations):
{' '.join(long_term_memory)}

Short-term Memory (current conversation):
{json.dumps(recent_messages, indent=2)}
"""
# User Entities:
# {json.dumps(user_entities, indent=2)}


# Create conversation chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
    ("human", "Memory context: {memory_context}"),
    # ("human", "Entity information: {entities}"),
    ("human", "Additional context: {context}")
])


chain = (
    RunnableParallel({
        "question": lambda x: x["question"],
        "memory_context": lambda x: get_memory_context(x.get("user_id", "default"), x["question"]),
        # "entities": lambda x: entity_memory.load_memory_variables({"input": x["question"]})["entities"] if hasattr(entity_memory, 'load_memory_variables') else {},
        "context": lambda x: "\n".join([doc.page_content for doc in context_vector_store.max_marginal_relevance_search(x["question"])]),
        # "llm_entities": lambda x: extract_entities_llm(x["question"]) if x["question"] else EntityExtraction(goals=[], values=[], plans=[], actions=[])
    })
    | prompt
    | chat
    | StrOutputParser()
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: Neo4jChatMessageHistory(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history"
)

def get_structured_chat_history() -> str:
    query = """
    MATCH (m:Message)
    WITH m ORDER BY m.timestamp DESC LIMIT 20
    RETURN m.id, m.session_id, m.type, m.text, m.timestamp
    ORDER BY m.timestamp ASC
    """
    result = neo4j_conn.run_query(query)
    
    history_components = []
    for record in result:
        message_component = f"""
Message ID: {record['m.id']}
Session ID: {record['m.session_id']}
Type: {record['m.type']}
Content: {record['m.text']}
Timestamp: {record['m.timestamp']}

---

"""
        history_components.append(message_component)
    
    return "".join(history_components) if history_components else "No chat history available"

def summarize_sessions(sessions):
    summary_prompt = f"""
    Today is {today}. Your name is Gais an expert performance coach, and Neuroscientist. You use science backed recommendations to help users achieve their goals. 
    If you know the name of the human user greet them by name and if available, summarize the following chat sessions and recommend next steps to the human.
    If no chat sessions are available you are meeting the user for the first time so introduce yourself and ask the user how they would like you to address them.

    Sessions:
    {sessions}

    Summary:
    """
    summary = chat.invoke(summary_prompt).content
    return summary

def get_session_summary(limit, user_id = 'default'):
    query = f"""
    MATCH (m:Message)
    WHERE m.user_id = "{user_id}"
    WITH m
    ORDER BY m.timestamp DESC
    LIMIT {limit}
    RETURN m.user_id AS user_id, 
           m.id AS id,
           m.text AS text, 
           m.type AS type,
           m.timestamp AS timestamp
    """
    result = neo4j_conn.run_query(query)
    
    sessions = []
    current_user = None
    session_content = ""
    
    for record in result:
        user_id = record['user_id']
        if user_id != current_user:
            if current_user is not None:
                sessions.append(session_content)
            current_user = user_id
            session_content = f"User ID: {user_id}\n"
        
        message_id = record.get('id', 'No ID')
        text = record.get('text', 'No content')
        timestamp = record.get('timestamp', 'No timestamp')
        
        if text is not None:
             parts = text.split('AI:', 1)
        else:
            # Handle the case where text is None
            parts = []

        if len(parts) > 1:
            human_text = parts[0].replace('Human:', '').strip()
            ai_text = parts[1].strip()
            session_content += f"{timestamp} - Human: {human_text}\n"
            session_content += f"{timestamp} - AI: {ai_text}\n"
        else:
            session_content += f"{timestamp} - Content: {text}\n"
        
        session_content += f"Message ID: {message_id}\n"
        session_content += "---\n"
    
    if session_content:
        sessions.append(session_content)
    
    return "\n\n".join(sessions)



# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY, 
                                           "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"],
                                           suppress_callback_exceptions=True, )
# Call this function when your app starts

app.title = 'Welcome to the Goalkeeper'

# App layout
app.layout = dbc.Container([
    dcc.Store(id='store-response', storage_type='memory'),
    dcc.Store(id='store-context', storage_type='memory'),
    dcc.Store(id='store-chat-history', storage_type='memory'),
    dcc.Store(id='store-entity-memory', storage_type='memory'),
    dcc.Store(id='store-session-summary', storage_type='memory'),
    # dbc.Button('Test Components', id='test-components-button', n_clicks=0),
    
    dbc.Row([
        dbc.Col([
            html.H1(app.title, className="text-center"),
            dcc.Textarea(id='user-prompt', placeholder='Enter your prompt here...', style={'width': '100%', 'height': 200}, className='border-rounded'),
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
    Output('content', 'children', allow_duplicate=True),
    Output('store-session-summary', 'data', allow_duplicate=True),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input('store-response', 'data'),  # This is just a dummy input to trigger the callback on page load
    prevent_initial_call='initial_duplicate',
)
def update_session_summary(dummy):
    ctx = callback_context
    if not ctx.triggered:

        sessions = get_session_summary(5)
        stored_summary = summarize_sessions(sessions)
        print(f"this is the summary: {stored_summary}")
        
        return_this =dbc.Card(dbc.CardBody([
            html.H4("Previous Sessions Summary", className="card-title"),
            dcc.Markdown(stored_summary, className="card-text")
        ]), className="mb-3")
        json.dumps({'summary':return_this})
        return return_this, return_this, no_update
    else:
        stored_summary = "No Summary "
        json.dumps({'summary':stored_summary})
        return no_update, no_update, no_update

@app.callback(
    Output('lobotomy-modal', "is_open"),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('content', 'children', allow_duplicate=True),
    [Input('lobotomize-button', 'n_clicks'), Input("close-modal", "n_clicks")],
    [State("lobotomy-modal", "is_open")],
    prevent_initial_call=True
)

def toggle_modal(n1, n2, is_open):
    if n1 > 0:
        query = "MATCH (n:!Chunk) DETACH DELETE n" # Delete all Nodes that are not Chunks of Transcripts
        neo4j_conn.run_query(query)
        short_term_memory.clear()
        neo4j_content = get_structured_chat_history()
        return not is_open, neo4j_content, ""
    return is_open, no_update, no_update

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
    Output('user-prompt', 'value'),
    Input('submit_button', 'n_clicks'),
    State('user-prompt', 'value'),
    State('store-chat-history', 'data'),
    prevent_initial_call=True
)
def update_stores(n_clicks, value, chat_history, user_id="default"):
    if n_clicks > 0:
        try:
            # Retrieve context from transcript vector store
            vector_context = "\n".join([doc.page_content for doc in context_vector_store.similarity_search(value, k=4)])

            # Get memory context (includes personal info and relevant past interactions)
            memory_context = get_memory_context(user_id, value)
            entity_context = entity_memory.load_memory_variables({"input": value})["entities"]
            # Update short-term memory
            short_term_memory.add_message("human", value)

            result = chain.invoke(
                {"question": value, 
                 "user_id": user_id,
                 "memory": memory_context,
                 "context": vector_context
                #  "llm_entities": user_entities = ""                
                 },
                config={"configurable": {"session_id": user_id}}
            )
            # # Use history_chain just for storing the complete conversation
            # chain_with_history.invoke(
            #     {"question": value},
            #     config={"configurable": {"session_id": user_id}}
            # )

            # Update short-term memory with AI response
            short_term_memory.add_message("ai", result)

            # Update vector memory with the new interaction
            update_vector_memory(user_id, value, "Human")
            update_vector_memory(user_id, result, "AI")
            update_graph_memory(user_id, value, "Human")
            update_graph_memory(user_id, result, "AI")            

            # Combine all context information
            full_context = f"""
                            Memory Context:
                            {memory_context}

                            Additional Context:
                            {vector_context}

                            # Entities:
                            # {entity_context}
                            """
         
            chat_history = json.loads(chat_history) if chat_history else []
            chat_history.append({"human": value, "ai": result})
            
            return (
                json.dumps({"response": result}),
                json.dumps({"context": full_context}),
                json.dumps({'history':chat_history}),
                json.dumps({"entities": entity_context}),
                "Query processed successfully",
                no_update,
                ""
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
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input("tabs", "active_tab"),
    Input('store-response', 'data'),
    Input('store-context', 'data'),
    Input('store-chat-history', 'data'),
    Input('store-entity-memory', 'data'),
    Input('store-session-summary', 'data'),
    prevent_initial_call=True
)
def switch_tab(active_tab, stored_response, stored_context, stored_chat_history, stored_entities, stored_summary):

    logger.debug(f"stored_response: {stored_response}")
    logger.debug(f"stored_context: {stored_context}")
    logger.debug(f"stored_chat_history: {stored_chat_history}")
    logger.debug(f"stored_entities: {stored_entities}")
    logger.debug(f"stored_summary: {stored_summary}")

    def safe_json_loads(data, default):
        try:
            return json.loads(data or '{}')
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for data: {data}")
            logger.error(f"Error: {str(e)}")
            return default
        
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        stored_response = safe_json_loads(stored_response, {})
        stored_context = safe_json_loads(stored_context, {})
        stored_chat_history = safe_json_loads(stored_chat_history, [])
        stored_entities = safe_json_loads(stored_entities, {})
        stored_summary = safe_json_loads(stored_summary, {}) 
    except json.JSONDecodeError as e:
        return "Error: Invalid data in storage", str(e), no_update

    if 'error' in stored_response or 'error' in stored_context or 'error' in stored_summary:
        error_msg = stored_response.get('error', '') or stored_context.get('error', '') or stored_summary.get('error','')
        return no_update, f"An error occurred: {error_msg}", no_update

    if triggered_id in ['tabs', 'store-response', 'store-context', 'store-chat-history','store-entity-memory']:
        if active_tab == "tab-response":
            return dcc.Markdown(str(stored_response.get('response', stored_summary.get('summary', 'No response yet')))), no_update, no_update
        elif active_tab == "tab-context":
            return dcc.Markdown(str(stored_context.get('context', 'No context available.'))), no_update, no_update
        elif active_tab == "tab-system":
            this=[
                    dbc.Button('Edit System Prompt', id='edit-system-prompt-button', n_clicks=0),
                    dcc.Textarea(id='system-prompt-textarea', style={'width': '100%', 'height': 200}, className='border-rounded'),
                    dbc.Button('Save System Prompt', id='save-system-prompt-button', n_clicks=0),
                    html.Br()
                ]
            this.append(system_prompt)
            return this, no_update, no_update
        elif active_tab == "tab-chat-history":
            this = [
                    dbc.Button('Lobotomize Me', id='lobotomize-button', n_clicks=0),
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Lobotomy Successful")),
                        dbc.ModalBody("Who are you and what am I doing here ;-)"),
                        dbc.ModalFooter(dbc.Button("Close", id='close-modal', className="ms-auto", n_clicks=0))
                    ], id='lobotomy-modal', is_open=False),
                    html.Div(id='neo4j-memory-content'),
                ]
            this.append(dcc.Markdown(str(fetch_neo4j_memory())))
            return this, no_update, no_update
        elif active_tab=="tab-entities":
            this = [dcc.Graph(id='network-graph', figure=create_network_graph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD))]
            return this, no_update, no_update            
        
    return "Please submit a query.", no_update, no_update

def fetch_neo4j_memory():
    query = """
    MATCH (m:Message)
    WHERE m.text IS NOT NULL  // This ensures we're getting the vector message nodes
    RETURN m.id, m.text, m.type, m.timestamp
    ORDER BY m.timestamp DESC
    LIMIT 100
    """
    result = neo4j_conn.run_query(query)
    
    if not result:
        return "No chat history available."
    
    formatted_history = "## Neo4j Chat History \n\n"
    for record in result:
        formatted_history += f"**ID:** {record['m.id']}\n"
        formatted_history += f"**Type:** {(record.get('m.type') or 'Unknown').capitalize()}\n"
        formatted_history += f"**Text:** {record.get('m.text', 'No ')}\n"
        formatted_history += f"**Timestamp:** {record.get('m.timestamp', 'No timestamp')}\n"
        formatted_history += "---\n\n"
    
    return formatted_history

def vector_similarity_search(query_text, k=4):
    query = f"""
    CALL {{
      CALL db.index.vector.queryNodes('message_vector', $k, $query_embedding)
      YIELD node, score
      WHERE exists(node.embedding)  // This ensures we're getting vector message nodes
      RETURN node, score
    }}
    RETURN node.text as text, score
    ORDER BY score DESC
    """
    query_embedding = embedding_model.embed_query(query_text)
    results = neo4j_conn.run_query(query, {"k": k, "query_embedding": query_embedding})
    return results


if __name__ == '__main__':
    app.run(debug=True, port=3050)