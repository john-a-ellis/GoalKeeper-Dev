import os, json, traceback, uuid, strip_markdown
from typing import Dict, List
from dash import Dash, html, dcc, Output, Input, State, no_update, callback_context, clientside_callback, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_cytoscape as cyto
import dash
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.memory import ConversationEntityMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableWithMessageHistory, RunnableSequence
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
import traceback
from requests import request
from requests_oauthlib import OAuth2Session
from datetime import datetime
# import logging
from pprint import pprint as pprint

#Load .env variables if not deployed.
if not os.getenv('DEPLOYED'):
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    get_login =""
# else:
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




# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template(["sketchy", "sketchy_dark"])

user_id='default'
today = datetime.now()
# Load environment variables 


# Setup Langchain Tracing
# os.environ["LANGCHAIN_TRACING_V2"] = True
os.environ["LANGCHAIN_PROJECT"] = "goalkeeper"
hf_key = os.getenv("HUGGINGFACE_API_KEY")
# Initialize models and databases
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=hf_key)

llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-70b-Versatile")
tool_llm = ChatGroq(temperature=0.0, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama3-groq-70b-8192-tool-use-preview")
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
                "AI", 
                "Age", 
                "BirthDate"]

allowed_relationships = ['HAS_PLAN',
                        'VALUES', 
                        'INFLUENCES', 
                        'HAS_PLAN', 
                        'LIVES_IN', 
                        'WORKS_AT', 
                        'WORKED_AT', 
                        'HAS_SKILL', 
                        'HAS_OBSTACLE', 
                        'HAS_SOLUTION',
                        'FACES',
                        'MITIGATES',
                        'SPOUSE_OF']

# Craft the prompt for LLM-based entity extraction
graph_transformer_prompt_template = PromptTemplate(template ="""
You are a data analyst expert in the use of Neo4j for storage and retrieval.
Analyze the following document and identify any nodes and relationships contained in the text and store them in the Neo4j database. 
The document properties identify whether the message was generated by the AI (type='AI'), or the human user (type = 'Human').
Pay special attention to identifying the human users name and relate those messages back to that named 'Person' entity. Similarly assign the 
AI messages to the 'AI' entity as appropriate. Strive to normalize the database by minimizing the duplication of entities.
                                                   
Here are some examples of how to relate other identified entities in the database:

MERGE (value: Value {{name: "Integrity", description: "Being honest and having strong moral principles"}})
MERGE (goal: Goal {{name: "Fitness", description: "Achieve a healthy body weight", deadline: "2025-01-01"}})
MERGE (plan: Plan {{name: "Workout Plan", steps: "Walk 60 minutes per day", resources: "Good walking shoes"}})
MERGE (action: Action {{description: "Morning run", date: "2024-10-01", status: "completed"}})
MERGE (obstacle: Obstacle {{description: "Lack of time", impact: "High"}})
MERGE (solution: Solution {{description: "Better time management", effectiveness: "Medium"}})
MERGE (person: Person {{name: "Henry Tudor", goes_by: "Hank", app_user: True}})
MERGE (place: Place {{name: "University of Toronto", type: "school"}})
MERGE (place: Place {{name: "IBM", type: "employer"}})

MERGE (person)-[:VALUES]->(value)
MERGE (value)-[:INFLUENCES]->(goal)
MERGE (goal)-[:HAS_PLAN]->(plan)
MERGE (plan)-[:CONSISTS_OF]->(action)
MERGE (action)-[:FACES]->(obstacle)
MERGE (solution)-[:MITIGATES]->(obstacle)
MERGE (person)-[:LIVES_IN]->(place)
MERGE (person)-[:WORKS_AT]->(place)
MERGE (person)-[:SPOUSE_OF]->(person)

Now, analyze the following input and create appropriate nodes and relationships:

{input}

Provide ONLY the Cypher queries to create the nodes and relationships.  Do not provide any descriptive analysis.
""")
graph_transformer = LLMGraphTransformer(llm = tool_llm,
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
with open('/etc/secrets/system.txt', 'r') as file:
    system_prompt = file.read()

#collect data for Entity Relationship Plot
def get_graph_data(url, user, password):
    driver = GraphDatabase.driver(url, auth=(user, password))
    with driver.session() as session:
        result = session.run("""
MATCH p=(n:!Chunk)-[r]->(m) 
        RETURN id(n) AS source, id(m) AS target, 
               labels(n) AS source_labels, labels(m) AS target_labels,
               type(r) AS relationship_type, n.id as id, n.text as text 
        """)
        
        return [record for record in result]


    
def update_graph_memory( user_id: str, content: str, type:str):
    ### updates neo4j with nodes and edges from chat messages

    this = strip_markdown.strip_markdown(content)
    document = Document(page_content=this, metadata={"source":type,
                                                    "user":user_id,
                                                    "id": None})
    try:
        graph_document = graph_transformer.process_response(document=document)
    
        graph_database.add_graph_documents(
                [graph_document],
                baseEntityLabel=False,
                include_source=True
                )
        
    except:
        ""


def update_vector_memory(user_id: str, content: str, type: str):
    ### updates neo4j message_vector index with messages for RAG retrieval

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
    ### retrieves x messages from vector memory using similarity search

    results = memory_vector_store.similarity_search(
        query=query,
        k=k,
        filter={"user_id": user_id}
    )
    return [doc.page_content for doc in results]

def get_memory_context(user_id: str, question: str):
    ### constructs memories to provide context to the LLM during chat sessions
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
    ("system", "Additional context: {context}")
])


chain = (
    RunnableParallel({
        "question": lambda x: x["question"],
        "memory_context": lambda x: get_memory_context(x.get("user_id", "default"), x["question"]),
        "context": lambda x: "\n".join([doc.page_content for doc in context_vector_store.max_marginal_relevance_search(x["question"])]),
    })
    | prompt
    | llm
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
    
    return "".join(history_components) if history_components else ""

def summarize_sessions(sessions):

    summary_prompt = f"""
    Today is {today}.   
    If you know the name of the human user greet them by name and if available, summarize the following chat sessions and recommend next steps to the human.
    If no chat sessions are available you are meeting the user for the first time so introduce yourself as Mel (the mental edge leader) a performance coach 
     and expert in Neuroscience who is designed to help users achieve their goals and ask the user how they would like you to address them. You only have to
     introduce yourself if their are no chat sessions to summarize.

    Sessions:
    {sessions}

    Summary:
    """
    summary = llm.invoke(summary_prompt).content
    return summary

def safe_json_loads(data, default):
    if isinstance(data, (dict, list)):
        return data
    try:
        return json.loads(data or '{}')
    except json.JSONDecodeError as e:
        # logger.error(f"JSON decode error for data: {data}")
        # logger.error(f"Error: {str(e)}")
        return default
    except TypeError as e:
        # logger.error(f"Type error for data: {data}")
        # logger.error(f"Error: {str(e)}")
        return default
    
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

def lobotomize_me():
        query = "MATCH (n:!Chunk) DETACH DELETE n" # Delete all Nodes that are not Chunks of Transcripts
        neo4j_conn.run_query(query)
        short_term_memory.clear()

def display_memory():
            this = []
            this.append(dbc.Button('Lobotomize Me', id='lobotomize-button', n_clicks=0, color='danger'))
            this.append(dbc.Tooltip(children='Erase all Conversations with the LLM', target='lobotomize-button', id='lobotomize-button-tip'))
            this.append(dcc.Markdown(str(fetch_neo4j_memory())))           
            this.append(dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Lobotomy Successful")),
                        dbc.ModalBody("Who are you and what am I doing here ;-)"),
                        dbc.ModalFooter(dbc.Button("Close", id='close-modal', className="ms-auto", n_clicks=0))
                    ], id='lobotomy-modal', is_open=False)),
            this.append(html.Div(id='neo4j-memory-content')),
        
            return this
def display_about():
    this = [dbc.Alert("""The GoalKeeper is an AI-powered performance coach. It leverages a large language model
                       (LLM) that accesses a large collection of curated YouTube transcripts featuring 
                      discussions with world-renowned experts in goal achievement across various fields. 
                      Both the LLM and the transcript cache are grounded in neuroscience. 
                      As you interact with the coach, it builds a “memory” from your conversations, 
                      enabling it to provide more personalized and effective responses to your future queries. 
                      This way, the coach becomes more attuned to your specific needs and can better
                       assist you in achieving your goals. You can view this memory by 'clicking' on the 'Entity Memory
                      Graph' button""", color='info', id="about-alert")]

    return this


def gen_entity_graph():
    my_nodes, my_edges = create_cyto_graph_data(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    default_stylesheet, nodes, edges, all_elements = create_cyto_elements(my_nodes, my_edges)
    edges_to_select = list(set(edge['data']['label'] for edge in edges))

    this = [
        dcc.Store(id='all-elements', storage_type='memory', data=all_elements),
        dcc.Store(id='all-node-elements', storage_type='memory', data=nodes),
        
        html.Label("Select Relationships", "edge-label"),
        dcc.Dropdown(
            id='select-edges-dropdown',
            value=edges_to_select,
            clearable=False,
            multi=True,
            options=[
                {'label': name.capitalize(), 'value':name}
                for name in edges_to_select
            ]
        ),
        dbc.Modal(

            children=[
                dbc.ModalHeader(dbc.ModalTitle(id='node-detail-title')),
                dbc.ModalBody(id='node-detail-content'),
                dbc.ModalFooter(dbc.Button(
                                    'Close',
                                    id='close-this-modal',
                                    n_clicks=0,
                                ),id='node-detail-footer',
                            ),
            ],
            id='node-detail-modal',
            is_open=False,
        ),   
        cyto.Cytoscape(
            id='cytoscape-memory-plot',
            layout={'name': 'cose',
            'nodeRepulsion': 400000,
            'idealEdgeLength': 50,  
            'edgeElasticity': 100,
            'numIter': 1000,
            'gravity': 80,
            'initialTemp': 200
            },
            elements=edges+nodes,
            boxSelectionEnabled = True,
            stylesheet=default_stylesheet,
            style={'width': '100%', 'height': '750px', 'background-color': 'aliceblue'}
        ),

    ]
    return this

# Register this page
dash.register_page(__name__, title='The GoalKeeper' )

#erase the memory at launch if app is deployed to the web
if os.getenv("DEPLOYED"):
    lobotomize_me()


# Call this function when your app starts
color_mode_switch = dbc.Row(
    [
        dbc.Col(
            [dbc.Label(className="fa-solid fa-moon", html_for="theme-switch"),
            dbc.Switch(id="theme-switch", value=True, className="d-inline-block ms-1", persistence=True),
            dbc.Label(className="fa-regular fa-sun", html_for="theme-switch")
            ],
            width=3, className="d-flex align-content-center")
    ],
    # align="center",
    className="d-flex align-content-center"
)


title = 'Welcome to the Goalkeeper'

# App layout
layout = dbc.Container([
    dcc.Store(id='store-response', storage_type='memory'),
    dcc.Store(id='store-context', storage_type='memory'),
    dcc.Store(id='store-chat-history', storage_type='memory'),
    dcc.Store(id='store-entity-memory', storage_type='memory'),
    dcc.Store(id='store-session-summary', storage_type='memory'),
    
    get_login,
    
    dbc.Row([
        dbc.Col([
            color_mode_switch  

        ], width={"size":3}),

        dbc.Col([
            html.H2(title, className="text-center"),
            dcc.Textarea(id='user-prompt',
                        placeholder='Enter your prompt here...',
                        style={'width': '100%', 'height': 100}, 
                        className='border-rounded'),
            dbc.Button('Submit', id='submit-button', n_clicks=0),
        ], width={"size": 6}),
        
        dbc.Col([
            html.Div([
                dbc.Button(
                    size="sm",
                    # variant="filled",
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
                    # variant="filled",
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
                    # variant="filled",
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
                    # variant="filled",
                    id="about-button",
                    n_clicks=0,
                    # color='info',
                    class_name="bi bi-question-circle-fill"
                ), 
                dbc.Tooltip(
                    "About",
                    target="about-button",
                    id="about-button-tooltip"
                ), 
                
            ],className="d-grid gap-2 d-md-flex justify-content-md-end"),
            dbc.Offcanvas([
                html.P("This is the settings Offcanvas")],
                placement="end",
                id='settings-offcanvas',
                title ="Settings",
            ),
            dbc.Offcanvas([
                html.P("This is the memory Offcanvas")],
                placement="end",
                id='memory-offcanvas',
                title ="Memories",
            ),
            dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Entity Memory")),
                        dbc.ModalBody("This is Entity Graph of the GoalKeeper", id="entity-graph-modal-body")
                    ],
                id='entity-graph-modal',
                fullscreen = True,
            ),
            dbc.Offcanvas([
                html.P("This is the about Offcanvas")],
                placement="end",
                id='about-offcanvas',
                title ="About",
            ),
        ], width = {"size":3})
    ], justify="end"),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(id="loading-response", type="cube", children=html.Div(id="loading-response-div")),
            # html.Hr(),
            dbc.Tabs([
                dbc.Tab(label="Response", tab_id="tab-response"),
                dbc.Tab(label="Context", tab_id="tab-context"),
                # dbc.Tab(label="Entities", tab_id="tab-entities"),
                # dbc.Tab(label="System Prompt", tab_id="tab-system", children=''),
                # dbc.Tab(label="Chat History", tab_id="tab-chat-history", children=''),
            ], id='tabs', active_tab="tab-response"),
            html.Div(id='content', children='', style={'height': '600px', 'overflowY': 'scroll', 'whiteSpace': 'pre-line'}, className="text-primary"),
            html.Div(id='error-output'),
        ])
    ])
], fluid=True, className='dashboard-container border_rounded')

# Callback functions
clientside_callback(
    """
    function(switchOn) {
        document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark");
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-switch", "id"),
    Input("theme-switch", "value"),
)
# @callback(
#         Output('url', 'href'), 
#         [Input('login-button', 'n_clicks')])

# def login_with_google(n_clicks):
#     if n_clicks:
#         google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
#         authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")
#         return authorization_url
#     return None

# @callback(
#         Output('content', 'children'), 
#         [Input('url', 'search')])

# def display_user_info(query_string):
#     if query_string:
#         google = OAuth2Session(client_id, redirect_uri=redirect_uri)
#         token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)
#         user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
#         return html.Div([
#             html.H1(f"Welcome, {user_info['name']}"),
#             html.P(f"Email: {user_info['email']}")
#         ])
#     return "Please login with Google."

@callback(
    Output('settings-offcanvas', 'is_open'),
    Output('settings-offcanvas', 'children'),
    Input('settings-button', 'n_clicks'),
    [State('settings-offcanvas', 'is_open')],
)
def display_settings(clicks, open_status):
    if clicks >0:
        this = dbc.Alert(
    [
        # html.H4("Settings", className="system-prompt"),
    
        # html.Label('System Prompt', id='settings-prompt-label'),
        # dcc.Textarea(id='system-prompt-textarea', style={'width': '100%', 'height': 400}, className='border-rounded', value=system_prompt),
        # dbc.Button('Edit System Prompt', id='edit-system-prompt-button', n_clicks=0, ),

        html.Label('LLM Temperature'),
        dcc.Slider(0, 1, 0.1, value=0.7, id='settings-slider'),
        html.Hr(),   
        dbc.Button('Save', id='save-settings-button', n_clicks=0, color="success", className="me-1"),
        html.Br(), 
    ], id='settings-alert'
)

    return True, this
@callback(
        Output('settings-alert', 'children'),
        Input('system-prompt-textarea', 'value'),
        Input('save-settings-button', 'n_clicks'), 
)
def save_settings(prompt, clicked):
    if clicked >0:
        if not os.getenv('DEPLOYED'):
            with open('/etc/secrets/system.txt', 'w') as file:
                file.write(prompt)
        return "System settings updated successfully."
    else:
        return no_update


@callback(
    Output('about-offcanvas', 'is_open'),
    Output('about-offcanvas', 'children'),
    Input('about-button', 'n_clicks'),
    [State('about-offcanvas', 'is_open')],
)
def update_about(clicks, open_status):
    this = display_about()
    return True, this

@callback(
    Output('memory-offcanvas', 'children'),
    Output('memory-offcanvas', 'is_open'),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input('memory-button', 'n_clicks'),
    [State('memory-offcanvas', 'is_open')],
    prevent_initial_call='initial_duplicate'
)
def show_memory(n_clicks, opened):
    if n_clicks > 0:
        this = display_memory()
        return this, True, no_update
    return no_update, no_update, no_update

@callback(
    Output('entity-graph-modal-body', 'children'),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('entity-graph-modal', 'is_open'),
    Input('entity-graph-button', 'n_clicks'),
    [State('entity-graph-modal', 'is_open')],
    # Input('store-entity-memory', 'data'),
    prevent_initial_call='initial_duplicate'
)
def update_entity_graph(clicks, dummy):

    this = gen_entity_graph()

    return this, no_update, True
    # return no_update, no_update


@callback(
    Output('content', 'children', allow_duplicate=True),
    Output('store-session-summary', 'data'),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input('store-response', 'data'),  # This is just a dummy input to trigger the callback on page load
    prevent_initial_call='initial_duplicate',
)
def update_session_summary(dummy):
    ctx = callback_context
    if not ctx.triggered:

        # if this is the initial callback from launch generate a summary of past sessions
        summary = summarize_sessions(get_session_summary(10))
        stored_summary = json.dumps({'summary':summary})
       
        
        summary_card = dbc.Card(
            dbc.CardBody([
                html.H4("Summary", className="card-title"),
                dcc.Markdown(summary, className="card-text")
            ]), 
            className="mb-3"
        )
        
        return summary_card, stored_summary, no_update
    # If it's not the initial load, don't update anything
    raise PreventUpdate    

@callback(
    Output('lobotomy-modal', "is_open"),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('content', 'children', allow_duplicate=True),
    [Input('lobotomize-button', 'n_clicks'), Input("close-modal", "n_clicks")],
    [State("lobotomy-modal", "is_open")],
    prevent_initial_call=True
)

def toggle_modal(n1, n2, is_open):
    if n1 > 0:
        lobotomize_me()
        neo4j_content = get_structured_chat_history()
        return not is_open, neo4j_content, ""
    return is_open, no_update, no_update


@callback(
    Output('cytoscape-memory-plot', 'elements'),
    State('all-node-elements', 'data'),
    Input('select-edges-dropdown', 'value'),
    State('all-elements', 'data'),
    prevent_initial_call = True
)

def updateElements(nodes, edges, elements):
    new_edges = [d for d in elements if d['data']['label'] in edges]
    new_elements = nodes + new_edges
    return new_elements

@callback(
    Output('user-prompt', 'value', allow_duplicate=True),
    Output('submit-button', 'disabled'),
    Input('edit-system-prompt-button', 'n_clicks'),
    prevent_initial_call=True
)
def edit_system_prompt(n_clicks):
    if n_clicks > 0:
        return system_prompt, True
    return no_update

@callback(
    Output('content', 'children', allow_duplicate=True),
    Output('submit-button', 'disabled', allow_duplicate=True),
    Output('user-prompt', 'value', allow_duplicate=True),
    Input('save-settings-button', 'n_clicks'),
    State('user-prompt', 'value'),
    prevent_initial_call=True
)
def save_system_prompt(n_clicks, new_prompt):
    if n_clicks > 0:
        global system_prompt
        system_prompt = new_prompt
        with open('/etc/secrets/system.txt', 'w') as file:
            file.write(system_prompt)
        return "System prompt updated successfully.", False, ""
    return no_update, no_update, no_update

@callback(
    Output('store-response', 'data', allow_duplicate=True),
    Output('store-context', 'data', allow_duplicate=True),
    Output('store-chat-history', 'data', allow_duplicate=True),
    # Output('store-entity-memory', 'data', allow_duplicate=True),
    Output('content', 'children', allow_duplicate=True),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('user-prompt', 'value', allow_duplicate=True),
    Input('submit-button', 'n_clicks'),
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
            # Update short-term memory
            short_term_memory.add_message("human", value)
            result = ""
            
            result = chain.invoke(
                {"question": value, 
                 "user_id": user_id,
                 "memory": memory_context,
                 "context": vector_context                
                 },
                config={"configurable": {"session_id": user_id}}
            )
 
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

                            """
         
            chat_history = safe_json_loads(chat_history,[]) if chat_history else []
            # print(f'THIS IS CHAT HISTORY: {chat_history}')
            # chat_history.append({"human": value, "ai": result})
            
            return (
                json.dumps({"response": result}),
                json.dumps({"context": full_context}),
                json.dumps({'history':chat_history}),
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
                no_update,
                ""
            )
    return no_update, no_update, no_update, no_update, no_update, ""

@callback(
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
    
    # logger.debug(f"stored_response: {stored_response}")
    # logger.debug(f"stored_context: {stored_context}")
    # logger.debug(f"stored_chat_history: {stored_chat_history}")
    # logger.debug(f"stored_entities: {stored_entities}")
    # logger.debug(f"stored_summary: {stored_summary}")
        
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
            if not stored_response.get('response') and stored_summary.get('summary'):
                return dbc.Card(dbc.CardBody([html.H4("Previous Sessions Summary", className="card-title"),dcc.Markdown(stored_summary['summary'], className="card-summary")])), no_update, no_update
            
            elif stored_response.get('response'):
                return dbc.Card(dbc.CardBody([dcc.Markdown(stored_response['response'], className="card-response")])), no_update, no_update
            else:
                return "No response or summary available.", no_update, no_update
        elif active_tab == "tab-context":
            if os.getenv("DEPLOYED")==True:
                return dbc.Card("No context available at this time")
            else:
                return dbc.Card(
                                dbc.CardBody(dcc.Markdown(str(stored_context.get('context', 'No context available.')), className="card-context"))
                                ), no_update, no_update
        # elif active_tab == "tab-system":
        #     this=[
        #             dbc.Button('Edit System Prompt', id='edit-system-prompt-button', n_clicks=0),
        #             # dcc.Textarea(id='system-prompt-textarea', style={'width': '100%', 'height': 200}, className='border-rounded'),
        #             dbc.Button('Save System Prompt', id='save-system-prompt-button', n_clicks=0),
        #             html.Br()
        #         ]
        #     this.append(system_prompt)
        #     return dbc.Card(dbc.CardBody(this, className="card-context")), no_update, no_update
    return no_update, no_update, no_update

@callback(
    Output('node-detail-modal', 'is_open'),
    Output('node-detail-title', 'children'),
    Output('node-detail-content', 'children'),
    Input('cytoscape-memory-plot', 'tapNodeData'),
    Input('close-this-modal', 'n_clicks'),
    State('node-detail-modal', 'is_open')
)
def display_node_details(node_data, n_clicks, is_open):
    ctx = callback_context

    if not ctx.triggered:
        return is_open, no_update, no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'close-this-modal' and is_open:
        return False, no_update, no_update

    if triggered_id == 'cytoscape-memory-plot' and node_data:
        return True, f"Details of {node_data['label']}", f"Message Text: {node_data.get('text', 'No text available')}."

    return is_open, no_update, no_update

def fetch_neo4j_memory(limit=100):
    query = f"""
    MATCH (m:Message)
    WHERE m.text IS NOT NULL  // This ensures we're getting the vector message nodes
    RETURN m.id, m.text, m.type, m.timestamp
    ORDER BY m.timestamp DESC
    LIMIT {limit}
    """
    result = neo4j_conn.run_query(query)
    
    if not result:
        return "No chat history available."
    
    formatted_history = ""
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

def create_cyto_graph_data(url, username, password):
    this = get_graph_data(url, username, password)
    if this != []:
        graph_nodes = []
        graph_edges = []
        # for record in graph_data:
    for record in this:
        if record['source'] not in graph_nodes:
            graph_nodes.append((record['source'], record['source_labels'][0], record['text']))
        if record['target'] not in graph_nodes:
            graph_nodes.append((record['target'], record['target_labels'][0], record['text']))
        graph_edges.append(
            (
                record['source'],
                record['target'],
                record['relationship_type'],
            )
        )
                           
            
    return graph_nodes, graph_edges

def create_cyto_elements(graph_nodes, graph_edges):
    
##### cytoscape layout
    label_to_class = {
        'Ai': 'Ai',
        'Document': 'Document',
        'Goal': 'Goal',
        'Action': 'Action',
        'Plan': 'Plan',
        'Person': 'Person',
        'Obstacle': 'Obstacle',
        'Value': 'Value',
        'Solution': 'Solution'
    }
        
    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    nodes = [
        {
            'data': {'id': id, 'label': label, 'text':text},
            'classes': label_to_class.get(label, 'default_class')
        }
        
        for id, label, text in (graph_nodes)
    ]

    edges = [
        {
            'data': {'source': source, 'target':target, 'label': label }
        }
        for source, target, label in (graph_edges)
    ]
    all_elements = edges + nodes

    default_stylesheet = [
        #group selectors
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'border-width':1,
                'shape':'ellipse',
                'width': 25,
                'opacity' : 0.5,
                'text-opacity': 1,
                'text-halign':'center',
                'text-valign':'center',   
            
            },
        },
        {
            'selector': 'node:selected',
            'style': {
                'background-color': '#F0C27B',
                'border-width': 2,
                'border-color': '#D89C3D'
            },
       },
 
        {
            'selector': 'edge',
            'style': {
                'label': 'data(label)',
                'line-color': 'gray',
                'curve-style':'straight',
                'width':1,
                'text-rotation':'autorotate',
                'target-arrow-shape':'triangle-backcurve',
                'target-arrow-color':'grey',
            }
        },
        {
            'selector': '*',
            'style': {
                'font-size':10,
            }
        },
        #class selectors
        {
            'selector': '.Document',
            'style': {
                'background-color': 'blue',
            }
        },
        {
            'selector': '.Ai',
            'style': {
                'background-color': 'yellow',
            }
        },
        {
            'selector': '.Goal',
            'style': {
                'background-color': 'green',
            }
        },
        {
            'selector': '.Action',
            'style': {
                'background-color': 'red',
            }
        },
        {
            'selector': '.Plan',
            'style': {
                'background-color': 'purple',
            }
        },
        {
            'selector': '.Person',
            'style': {
                'background-color': 'orange',
            }
        },
        {
            'selector': '.Obstacle',
            'style': {
                'background-color': 'black',
            }
        },
        {
            'selector': '.Solution',
            'style': {
                'background-color': 'lightgreen',
            }
        },
        {
            'selector': '.Value',
            'style': {
                'background-color': 'darkgreen',
            }          
        },
    ]
    return default_stylesheet, nodes, edges, all_elements
    ##### End cytoscape layout
#redirect port for Render deployment
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('DASH_PORT'))
