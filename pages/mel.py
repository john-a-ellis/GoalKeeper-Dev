import os, json, traceback, strip_markdown
from typing import Dict, List, Any
from dash import html, dcc, Output, Input, State, no_update, callback_context, clientside_callback, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_cytoscape as cyto
import dash
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
import traceback
from datetime import datetime
from pprint import pprint as pprint
from src import feedback_frm
from src.custom_modules import read_prompt, get_user_id
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template(["sketchy", "sketchy_dark"])

# user_id='default'
today = datetime.now()

# Setup Langchain Tracing
# os.environ["LANGCHAIN_TRACING_V2"] = True
os.environ["LANGCHAIN_PROJECT"] = "goalkeeper"
hf_key = os.getenv("HUGGINGFACE_API_KEY")
# Initialize models and databases
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", 
                                                huggingfacehub_api_token=hf_key)

# Dynamic LLM creation with temperature from dcc.store
def create_dynamic_llm(temperature=0.7):
    return ChatGroq(
        model_name="llama-3.1-70b-versatile", 
        temperature=temperature
    )
llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-70b-versatile")
tool_llm = ChatGroq(temperature=0.0, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-70b-versatile")
# initialize Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

#initialize vector stores

## Vectore store for Youtube Transcripts providing supporting context
context_vector_store = Neo4jVector.from_existing_index(
    embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="vector",
)

#initialize Graph Database
graph_database = Neo4jGraph(url=NEO4J_URI,
                            username=NEO4J_USERNAME,
                            password=NEO4J_PASSWORD)

#intialize Graph Database LLM Transformer from Langchain
# Updated ALLOWED_NODES to reflect two-participant model
ALLOWED_NODES = [
    "MainParticipant",  # Only two instances ever exist: User and Assistant
    "ReferencedIndividual",  # For people mentioned in conversations
    "ValueBasedGoal", 
    "CoreValue", 
    "Mindset",
    "DomainMindset",
    "Intervention", 
    "Obstacle", 
    "Solution", 
    "ActionStep",
    "PerformanceMetric",
    "OtherEntity"
]

# Simplified ALLOWED_RELATIONSHIPS
ALLOWED_RELATIONSHIPS = [
    "AUTHORED",  # Links MainParticipant to Document
    "MENTIONED_IN",  # Links ReferencedIndividual to Document
    "RESPONDS_TO",  # Links Documents in conversation thread
    "HAS_GOAL", 
    "HAS_CORE_VALUE", 
    "HAS_MINDSET", 
    "HAS_OBSTACLE", 
    "HAS_SOLUTION",
    "ASSOCIATED_WITH_DOMAIN",
    "REFERENCES"  # Generic relationship for entity references within documents
]

# Node properties organized by node type
NODE_PROPERTIES = {
    "MainParticipant": [
        "name",
        "participant_id",
        "participant_type",
        "created_at",
        "last_active"
    ],
    "ReferencedIndividual": [
        "name",
        "individual_id",
        "first_mentioned_in",
        "created_at"
    ],
    "ValueBasedGoal": [
        "title",
        "description",
        "start_date",
        "target_completion_date",
        "status"
    ],
    "CoreValue": [
        "name",
        "description",
        "domain",
        "importance_score"
    ],
    "Mindset": [
        "type",
        "key_characteristics",
        "development_areas"
    ],
    "DomainMindset": [
        "type",
        "confidence_level",
        "growth_potential"
    ],
    "Intervention": [
        "title",
        "description",
        "start_date",
        "duration_days"
    ],
    "Obstacle": [
        "description",
        "type",
        "severity",
        "impact_on_goal"
    ],
    "Solution": [
        "description",
        "estimated_effectiveness",
        "estimated_time_investment"
    ],
    "ActionStep": [
        "description",
        "target_start_date",
        "target_completion_date",
        "status"
    ],
    "PerformanceMetric": [
        "name",
        "current_value",
        "target_value",
        "measurement_unit"
    ],
    "OtherEntity" : [
    "name",
    "entity_type",
    "sub_type",
    "description",
    "first_mentioned_in"
    ]
}

# Relationship properties with specific contexts
RELATIONSHIP_PROPERTIES = {
    "AUTHORED": [
        "timestamp"
    ],
    "RESPONDS_TO": [
        "timestamp",
        "response_type"
    ],
    "HAS_GOAL": [
        "priority",
        "commitment_level"
    ],
    "HAS_CORE_VALUE": [
        "alignment_strength",
        "importance"
    ],
    "HAS_MINDSET": [
        "confidence_level",
        "development_potential"
    ],
    "HAS_OBSTACLE": [
        "impact_severity",
        "urgency"
    ],
    "HAS_SOLUTION": [
        "effectiveness_rating",
        "implementation_status"
    ],
    "ASSOCIATED_WITH_DOMAIN": [
        "relevance_score",
        "impact_level"
    ],
    "REFERENCES": [
        "reference_type",
        "context",
        "timestamp"
    ]
}

# Enhanced prompt template focusing on two-participant model
graph_transformer_prompt_template = PromptTemplate(template ="""
You are an expert knowledge graph extractor for one-on-one coaching conversations between a User (Human) and an AI Assistant.

Extract structured graph information with this understanding:

1. Core Participants:
- There are exactly two main participants:
  * The User (Human): Extract participant_id, name, and track last_active
  * The AI Assistant: Extract participant_id, name, and track last_active
- Any other individuals mentioned are ReferencedIndividuals (track individual_id, name, first_mentioned_in)

2. Document Structure:
- Every Document must be authored by either the User or Assistant
- Capture required properties: content, timestamp, source, message_id, conversation_id
- Documents form a conversation thread through RESPONDS_TO relationships
- Documents can reference other entities (goals, values, referenced individuals)

3. Entity Extraction:
- Create nodes for goals, values, mindsets, etc. with all specified properties
- For mentioned individuals, create ReferencedIndividual nodes only if referenced individual is not one of the Core Participants
- Capture enum values precisely:
  * ParticipantType (Human, AI, Referenced)
  * GoalStatus (NotStarted, InProgress, Completed, Blocked)
  * ValueDomain (Personal Growth, Professional Development, Relationships, Health & Wellness, Community Impact, Spiritual, Financial Independence)
  * MindsetType (GrowthMindset, FixedMindset)
  * ObstacleType (Internal, External, Resource-Based, Skill-Based)

4. Relationship Guidelines:
- AUTHORED relationships only connect User or Assistant to Documents (include timestamp)
- MENTIONS connects ReferencedIndividuals to Documents (include context and timestamp)
- All other relationships must include their specified properties

5. Entity Classification Guidelines:
- When encountering entities not matching existing node types:
  * Create an OtherEntity node
  * Carefully assign appropriate entity_type:
    - Use OtherEntityType enum values
    - Choose most specific classification
  * Use sub_type for additional granularity
  * Reject creating nodes for:
    - Vague or abstract concepts
    - Transient mentions
    - Purely contextual references

Entity Classification Hierarchy:
1. Is it a specific, named individual? → ReferencedIndividual
2. Does it match an existing node type? → Use that type
3. Is it a concrete, identifiable entity? → OtherEntity
4. If uncertain, exclude from graph

Example Mappings:
- "University of Toronto" → OtherEntity(
    entity_type=OtherEntityType.INSTITUTION, 
    sub_type="University"
)
- "New York City" → OtherEntity(
    entity_type=OtherEntityType.LOCATION, 
    sub_type="Major City"
)
- "Agile Methodology" → OtherEntity(
    entity_type=OtherEntityType.CONCEPT, 
    sub_type="Project Management"
)
                                                   
Focus on maintaining the clear distinction between the two main participants and referenced individuals while capturing all required properties for nodes and relationships.

Text to Process:
{input}
""")

graph_transformer = LLMGraphTransformer(llm = tool_llm,
                                        prompt = graph_transformer_prompt_template,
                                        allowed_nodes = ALLOWED_NODES,
                                        allowed_relationships = ALLOWED_RELATIONSHIPS,
                                        strict_mode = True,
                                        relationship_properties = RELATIONSHIP_PROPERTIES,
                                        node_properties = NODE_PROPERTIES
                                      )

## Vector store for chat messages

memory_vector_store = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="message_vector",
    node_label="Document",
    text_node_properties=['text'],
    embedding_node_property="embedding"
)

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
with open('assets/prompts/system.txt', 'r') as file:
    system_prompt = file.read()

def update_graph_memory(user_id: str, content: str, type: str):
    # Debug: print input content
    # print(f"Input content: {content}")

    # Strip markdown from the content
    this = strip_markdown.strip_markdown(content)

    # Debug: check if this is None
    # print(f"Stripped markdown: {this}")
    if this is None:
        print("Warning: strip_markdown returned None")
        # this = content  # fallback to original content

    # Create a document object
    document = Document(page_content=this, metadata={
        "source": type,
        "user": user_id,
        "id": None,
        "timestamp": datetime.now().isoformat()
    })

    try:
        # Process the document into a graph document
        graph_document = graph_transformer.process_response(document=document)

        # Add the graph document to the Neo4j graph database
        graph_database.add_graph_documents(
            [graph_document],
            baseEntityLabel=False,
            include_source=True
        )

        # Refresh the Neo4j schema
        graph_database.refresh_schema()

        # Query for document nodes without embeddings
        document_nodes = graph_database.query("""
            MATCH (n:Document)
            WHERE n.embedding IS NULL
            RETURN n.id AS node_id, n.text AS text
        """)

        # print(f"THESE ARE THE DOCUMENT NODES: {document_nodes}")

        # Embed the text and add it to the document node properties
        for document_node in document_nodes:
            node_id = document_node["node_id"]
            # print(f"THIS IS THE NODE ID: {node_id}")

            # Generate document embedding
            document_embedding = embedding_model.embed_documents([document_node["text"]])[0]
            # Convert the embedding to a flat list if necessary
            flat_embedding = [float(value) for value in document_embedding]
            
            # Update the node properties with the new embedding
            stored_embedding = graph_database.query("""
                MATCH (n:Document)
                WHERE n.id = $nodeid
                SET n.embedding = $embedding
                RETURN n.id, n.embedding
            """, params={"nodeid": node_id, "embedding": flat_embedding})

    except Exception as e:
        print(f"An error occurred creating graph document: {e}")

def retrieve_vector_memory(user_id: str, query: str, k: int = 4):
    ### retrieves x messages from vector memory using similarity search
    try:
        results = memory_vector_store.similarity_search(
            query=query,
            k=k,
            filter={"user": user_id}
                # "source":"Human"}
        )
        return [doc.page_content for doc in results]

    except Exception as e:
        print(f"An error occurred performing a vector similarity search: {e}")

    
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

# Create conversation chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
    ("system", "{memory_context}"),
    ("system", "{current_datetime}"),
    ("system",  "{user_id}"),
    ("system", "{context}")
])

# First, create a custom output parser that preserves metadata
class AnnotatedOutputParser:
    def parse(self, completion, metadata):
        return {
            "response": completion,
            "metadata": metadata
        }
    
#retrieve meta data to annotate result
def extract_metadata_and_content(docs: List[Document]) -> Dict[str, Any]:
    """Extract both content and metadata from documents."""
    content = []
    metadata_list = []
    
    for doc in docs:
        # content.append(doc.page_content)
        if hasattr(doc.metadata, 'items'):
            metadata_list.append(doc.metadata)
        else:
            metadata_list.append({})
            
    return {
        # "content": "\n".join(content),
        "metadata": metadata_list
    }

# Logging function for retrieved context documents
def log_retrieved_docs(docs_with_scores, source="Not specified"):
    # print(f"\n=== Retrieved Documents ({source}) ===")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        # print(f"\nDocument {i}:")
        # print(f"Content: {doc.page_content}")
        if hasattr(doc.metadata, 'items'):  # Check if metadata exists
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print(f"  similarity_score: {score:.4f}")
        print("-" * 50)

chain = (
    RunnableParallel({
        "question": lambda x: x["question"],
        "memory_context": lambda x: get_memory_context(x.get("user_id", "default"), x["question"]),
        "context_data": lambda x: (lambda docs_with_scores: (
            {
                "content": "\n".join(
                    doc.page_content
                    for doc, score in docs_with_scores
                    if score >= x.get("similarity_threshold", 0.75)
                ),
                "metadata": [
                    {
                        **(doc.metadata if hasattr(doc.metadata, 'items') else {}),
                        "similarity_score": score
                    }
                    for doc, score in docs_with_scores
                    if score >= x.get("similarity_threshold", 0.75)
                ]
            }
        ))(
            context_vector_store.similarity_search_with_relevance_scores(
                x["question"],
                k=4,
                lambda_mult=x.get("relevance_target", 0.7)
            )
        ),
        "current_datetime": lambda x: x.get("datetime", datetime.now().isoformat()),
        "user_id": lambda x: x.get("user_id"),
        "temperature": lambda x: x.get("temperature", 0.7)
    })
    | RunnableParallel({
        "llm_input": RunnableParallel({
            "question": lambda x: x["question"],
            "memory_context": lambda x: x["memory_context"],
            "context": lambda x: x["context_data"]["content"],
            "current_datetime": lambda x: x["current_datetime"],
            "user_id": lambda x: x["user_id"],
            "temperature": lambda x: x["temperature"],
        }) 
        | prompt,
        "metadata": lambda x: x["context_data"]["metadata"]
    })
    | (lambda x: {
        "completion": create_dynamic_llm(0.7).invoke(x["llm_input"]),
        "metadata": x["metadata"]
    })
    | (lambda x: {
        "response": x["completion"],
        "metadata": x["metadata"]
    })
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: Neo4jChatMessageHistory(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        session_id=session_id,
        node_label="Document"
    ),
    input_messages_key="question",
    history_messages_key="history"
)

def get_structured_chat_history(user_id: str = 'default', limit: int = 100) -> str:
    #retrieves Graph nodes
    query = f"""
    MATCH (m:Document) WHERE m.user = '{user_id}'
    WITH m ORDER BY m.timestamp DESC LIMIT {limit}
    RETURN m.id, m.source, m.text, m.timestamp
    ORDER BY m.timestamp ASC
    """
    result = graph_database.query(query)
    
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
    # summary_prompt = read_prompt('initial_prompt_template')
    summary_prompt = ["system", f"""
    Today is {today}.       
    1. You are a helpful AI driven performance coach and expert in neuroscience and the growth mindset. 
    2. Your purpose is to help human users achieve the goals they identify through the application of neuroscience and the growth mindset.
    3. Your name is Mel (a Mindset-oriented, Eidetic, Librarian)
    4. If no chat sessions are available you are meeting the user for the first time so ask the user how they would like you to address them. 
    5. Only introduce yourself if chat sessions are not available.
    Otherwise if chat sessions are available but only if chat sessions are available:
    1. Summarize them in one or two sentences and recommend a next step, then ask how the human user would like to proceed.
   

    Chat Sessions:
    {sessions}
    """]
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
    
def get_session_summary(limit=4, user_id='default'):
    #retrieves document nodes
    query = f"""
    MATCH (m:Document)
    WHERE m.user = '{user_id}'
    WITH m
    ORDER BY m.timestamp DESC
    LIMIT {limit}
    RETURN m.user AS user_id, 
           m.id AS id,
           m.text AS text, 
           m.source AS type,
           m.timestamp AS timestamp
    """
    # print(f'THIS IS MY SESSION QUERY: {query}')
    result = graph_database.query(query)
    
    sessions = []
    current_user = None
    session_content = ""
    
    for record in result:
        user_id = record['user_id']
        if user_id != current_user:
            if current_user is not None:
                sessions.append(session_content)
            current_user = user_id
            session_content = f"User ID: $user_id\n"
        
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

def lobotomize_me(user_id = 'default'):
        #retrieves graph memory nodes
        query = f"""MATCH (n:!Chunk) 
                        WHERE n.user = '{user_id}' 
                        OPTIONAL MATCH (n)-[r]->(m)
                        DETACH DELETE n, m"""  # Delete all Nodes that are not Chunks of Transcripts

        graph_database.query(query)
        short_term_memory.clear()

def display_memory(user_id='default'):
            this = []
            this.append(dbc.Button('Lobotomize Me', id='lobotomize-button', n_clicks=0, color='danger'))
            this.append(dbc.Tooltip(children='Erase all Conversations with the LLM', target='lobotomize-button', id='lobotomize-button-tip'))
            this.append(dcc.Markdown(str(fetch_neo4j_memory(user_id))))           
            this.append(dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Lobotomy Successful")),
                        dbc.ModalBody("Who are you and what am I doing here ;-)"),
                        dbc.ModalFooter(dbc.Button("Close", id='close-modal', className="ms-auto", n_clicks=0))
                    ], id='lobotomy-modal', is_open=False)),
            this.append(html.Div(id='neo4j-memory-content')),
        
            return this

def display_about():
    with open('assets/disclaimer.md', 'r') as file:
            about = file.read()

    this = [dbc.Alert(dcc.Markdown(about), color='info', id="about-alert")]

    return this

# Register this page
dash.register_page(__name__, title='The GoalKeeper', name='The GoalKeeper', path='/' )

# App layout
layout = dbc.Container([
    dcc.Store(id='store-response', storage_type='memory'),
    # dcc.Store(id='store-context', storage_type='memory'),
    dcc.Store(id='store-chat-history', storage_type='memory'),
    dcc.Store(id='store-entity-memory', storage_type='memory'),
    dcc.Store(id='store-session-summary', storage_type='memory'),
    dcc.Store(id='store-temperature-setting', storage_type='local'),
    dcc.Store(id='store-relevance-setting', storage_type='local'),
    dcc.Store(id='store-similarity-setting', storage_type='local'),

    dbc.Row([
        
        dbc.Col([
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
                dbc.ModalHeader(dbc.ModalTitle("Entity Memory",style={'color':'green',
                                                                        'border-color':'green'})),
                dbc.ModalBody("This is Entity Graph of the GoalKeeper", id="entity-graph-modal-body")
                ],
                id='entity-graph-modal',
                fullscreen = True,
            ),
            dbc.Modal([
                dbc.ModalBody("This is the Feedback modal", 
                              id="feedback-modal-body",
                              style={"color":"navy",
                                     "background-color":"powder-blue"})
            ],
            id='feedback-modal',
            centered=True,
            size = 'xl'),
            dbc.Offcanvas([
                html.P("This is the about Offcanvas")],
                placement="end",
                id='about-offcanvas',
                title ="About",
            ),
        ], width = {"size":3})
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(id="loading-response", type="cube", children=html.Div(id="loading-response-div")),
        ], width={"size": 12}),
    ], justify="end"),
    dbc.Row([
        dbc.Col([
        ], width={"size":3}),

        dbc.Col([
            # html.H2(title, className="text-center"),
            dcc.Textarea(id='user-prompt',
                        placeholder='Enter your thoughts here...',
                        style={'width': '100%', 'height': 100}, 
                        className='border-rounded mb-3, mt-3'),
            dbc.Button('Submit', id='submit-button', n_clicks=0),
        ], width={"size": 6}, class_name="justify-content-md-end"),
        dbc.Col([], width={"size":3}),
    ], justify="end", className='justify-content-md-end'),
    
], fluid=True, className='', id='page-container')
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
@callback(
        Output('feedback-modal', 'is_open'),
        Output('feedback-modal-body', 'children'),
        Input('feedback-button', 'n_clicks'),
        prevent_initial_call = True
)
def get_feedback_form(clicks):
    if clicks:
        return(True, [feedback_frm.feedback_form])
    else:
        no_update, no_update
@callback(
    Output('settings-offcanvas', 'is_open'),
    Output('settings-offcanvas', 'children'),
    Input('settings-button', 'n_clicks'),
    # State('settings-offcanvas', 'is_open'),
    State('store-relevance-setting', 'data'),
    State('store-temperature-setting', 'data'),
    State('store-similarity-setting', 'data'),
    prevent_initial_call = True
)
def display_settings(clicks, relevance, temperature, similarity):
    if clicks >0:
        this = dbc.Alert([   
        # html.Label('System Prompt (for information only)', id='settings-prompt-label'),
        # dcc.Textarea(id='system-prompt-textarea',
        #             style={'width': '100%', 'height': 400}, 
        #             #  className='border-rounded', 
        #             value=system_prompt, 
        #             disabled=True),
        # html.Br(),
        html.Label('LLM Temperature'),
        dcc.Slider(0, 1, 0.10, value=temperature, id='temperature-slider', persistence=False),
        dbc.Tooltip('The higher the Temperature the more "creative" is Mel\'s responses', target='temperature-slider'),
        html.Hr(),   
        html.Label('Acceptable Similarity'),
        dcc.Slider(0, 1, 0.25, value=similarity, id='similarity-slider', persistence=False),
        dbc.Tooltip("Only youtube transcripts achieving a similarity score at or higher than this setting when compared to the users prompt will be considered in the response", target ='similarity-slider'),
        html.Hr(), 
        html.Label('Relevance Target'),
        dcc.Slider(0, 1, 0.25, value=relevance, id='relevance-slider', persistence=False),
        dbc.Tooltip("The higher the context Relevance Target the more similar the retrieved transcripts will be to one another", target ='relevance-slider'),
        html.Hr(), 
        dbc.Button('Save', id='save-settings-button', n_clicks=0, color="warning", className="me-1"),
        
        html.Br(), 
        ], id='settings-alert', color='warning')

    return True, this

@callback(
        Output('settings-alert', 'children'),
        Output('store-relevance-setting', 'data'),
        Output('store-temperature-setting', 'data'),
        Output('store-similarity-setting', 'data'),
        # Input('system-prompt-textarea', 'value'),
        Input('save-settings-button', 'n_clicks'), 
        Input('relevance-slider', 'value'),
        Input('temperature-slider', 'value'),
        Input('similarity-slider', 'value'),
        prevent_initial_call = True
)
def save_settings(clicked, relevance, temperature, similarity):
    if clicked >0:
        # if os.getenv('IS_DEPLOYED', 'False').lower() == 'true':
        #     with open('/etc/secrets/system.txt', 'w') as file:
        #         file.write(prompt)
        return "System settings updated successfully.", relevance, temperature, similarity
    else:
        return no_update, no_update, no_update, no_update


@callback(
    Output('about-offcanvas', 'is_open'),
    Output('about-offcanvas', 'children'),
    Input('about-button', 'n_clicks'),
    State('about-offcanvas', 'is_open'),
    prevent_intial_call = True
)
def update_about(clicks, open_status):
    if clicks > 0:
        this = display_about()
        return True, this
    return False, no_update

@callback(
    Output('memory-offcanvas', 'children'),
    Output('memory-offcanvas', 'is_open'),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input('memory-button', 'n_clicks'),
    State('memory-offcanvas', 'is_open'),
    Input('auth-store', 'data'),
    prevent_initial_call=True
)
def show_memory(n_clicks, opened, auth_data):
    user_id = get_user_id(auth_data)
    if n_clicks > 0:
        this = display_memory(user_id)
        return this, True, no_update
    return no_update, no_update, no_update

@callback(
    Output('entity-graph-modal-body', 'children'),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('entity-graph-modal', 'is_open'),
    Input('auth-store', 'data'),
    Input('entity-graph-button', 'n_clicks'),
    State('entity-graph-modal', 'is_open'),
    # Input('store-entity-memory', 'data'),
    prevent_initial_call=True
)
def update_entity_graph(auth_data, clicks, dummy):
    user_id = get_user_id(auth_data)
    this = gen_entity_graph(user_id)

    return this, no_update, True
    # return no_update, no_update

#this call back fires on load to initialize the data stores for the app and summarize previous sessions
@callback(
    Output('content', 'children', allow_duplicate=True),
    Output('store-session-summary', 'data'),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Input('store-response', 'data'),  # This is just a dummy input to trigger the callback on page load
    Input('auth-store', 'data'),
    prevent_initial_call='initial_duplicate',
)

def update_session_summary(dummy, auth_data):
    ctx = callback_context

    if ctx.triggered[0]['value'] == None:
        user_id = get_user_id(auth_data)
        
        # if this is the initial callback from launch generate a summary of past sessions
        summary = summarize_sessions(get_session_summary(10, user_id))
        stored_summary = json.dumps({'summary':summary})

        summary_card = dbc.Card(
            dbc.CardBody([
                html.H4("Summary", id="card-title"),
                dcc.Markdown(summary, id="response-card-text")
            ]), 
            className="mb-3"
        )
        
        return summary_card, stored_summary, no_update

    # If it's not the initial load, don't update anything
    raise PreventUpdate    

@callback(
    Output('lobotomy-modal', "is_open"),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('response-card-text', 'children', allow_duplicate=True),
    Output('memory-offcanvas', 'children', allow_duplicate=True),
    Input('lobotomize-button', 'n_clicks'),
    Input("close-modal", "n_clicks"), 
    Input("auth-store", "data"),
    State("lobotomy-modal", "is_open"),
    
    prevent_initial_call=True
)

def lobotomize_button_click(n1, n2, auth_data, is_open):
    user_id=get_user_id(auth_data)
    if n1 > 0:
        lobotomize_me(user_id)
        neo4j_content = get_structured_chat_history(user_id)
        return True, neo4j_content, "Who are you and what am I doing here ;-)",""
    
    return is_open, no_update, no_update, no_update


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

# User Prompt Submission
@callback(
    Output('store-response', 'data', allow_duplicate=True),
    # Output('store-context', 'data', allow_duplicate=True),
    Output('store-chat-history', 'data', allow_duplicate=True),
    # Output('store-entity-memory', 'data', allow_duplicate=True),
    Output('content', 'children', allow_duplicate=True),
    Output('loading-response-div', 'children', allow_duplicate=True),
    Output('user-prompt', 'value', allow_duplicate=True),
    Input('submit-button', 'n_clicks'),
    State('user-prompt', 'value'),
    State('store-chat-history', 'data'),
    State('auth-store', 'data'),
    State('store-relevance-setting', 'data'),
    State('store-temperature-setting','data'),
    State('store-similarity-setting', 'data'),
    prevent_initial_call=True
)

def update_stores(n_clicks, value, chat_history, auth_data, relevance_data, temperature_data, similarity_data):
    if n_clicks > 0:
        try:
            user_id=get_user_id(auth_data)
            short_term_memory.add_message("user", value)
            relevance = relevance_data if isinstance(relevance_data, (int, float)) else 0.7
            temperature =  temperature_data if isinstance(temperature_data, (int, float)) else 0.7
            similarity = similarity_data if isinstance(similarity_data, (int, float)) else 0.7
            try:
                result = chain.invoke(
                    {"question": value, 
                    "user_id": user_id,  
                    "datetime":datetime.now().isoformat(),
                    "similarity_threshold":similarity,
                    "relevance_target":relevance,
                    "temperature":temperature       
                    },
                    config={"configurable": {"session_id": user_id}}
                )
            except TypeError:
                error_msg = dbc.Alert("OOPS! An error has occured please 'Submit' again.", id='error-alert', color='warning')
                return no_update, no_update, error_msg, no_update, no_update
                
            result_to_process = result['response'].content
            

            seen_pairs = set()
            sources_titles = []

            for x in result['metadata']:
                if 'source' in x and 'title' in x:
                    pair = (x['title'], x['source'])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        sources_titles.append(f'[{x["title"]}]({x["source"]})+\n')

            if len(sources_titles) > 0:
                response_annotation='\n\n **YouTube Sources** \n\n'
                response_annotation += '\n'.join(sources_titles) + '\n'
                annotated_response = result_to_process + response_annotation
            else:
                annotated_response = result_to_process

            # Update short-term memory with AI response
            short_term_memory.add_message("ai", result_to_process)

            # Update graph memory with the new interaction

            update_graph_memory(user_id, value, "Human")
            update_graph_memory(user_id, result_to_process, "AI")            
         
            chat_history = safe_json_loads(chat_history,[]) if chat_history else []
            response_card = dbc.Card(
            dbc.CardBody([
                # html.H4("Response", id="response-card-title"),
                dcc.Markdown(annotated_response, id="response-card-text")
            ]), 
            className="mb-3"
        )
            return (
                json.dumps({"response":annotated_response}),
                json.dumps({'history':chat_history}),
                response_card,
                no_update,
                ""
            )
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # This will print to your console or logs
            return (
                json.dumps({"error": f"Failed to process query: {str(e)}"}),
                json.dumps([]),
                # no_update,
                error_msg,
                no_update,
                ""
            )
    return no_update, no_update, no_update, no_update, ""


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
    nl = '\n'

    if not ctx.triggered:
        return is_open, no_update, no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'close-this-modal' and is_open:
        return False, no_update, no_update

    if triggered_id == 'cytoscape-memory-plot' and node_data:
        details = [f"Entity Type: {node_data.get('node_type', 'None')}"]
        details.append(f"{nl} Text: {nl} {node_data.get('text', 'No text available')}")
        return True, f"Details of {node_data['label']}", dcc.Markdown(details)
        

    return is_open, no_update, no_update

def fetch_neo4j_memory(user_id='default', limit=1000):
    #fetching vector memory
    query = f"""
    MATCH (m:Document)
    WHERE m.text IS NOT NULL AND m.user = '{user_id}'  // This ensures we're getting the vector message nodes
    RETURN m.id, m.text, m.source, m.timestamp
    ORDER BY m.timestamp DESC
    LIMIT {limit}
    """
    # print(f'THIS IS MY QUERY: {query}')
    
    result = graph_database.query(query)
    # print(f'THIS IS THE RESULT: {result}')
    if not result:
        return "No chat history available."
    
    formatted_history = ""
    for record in result:
        formatted_history += f"**ID:** {record['m.id']}\n \n"
        formatted_history += f"**Type:** {(record.get('m.source') or 'Unknown').capitalize()}\n \n"
        formatted_history += f"**Text:** {record.get('m.text', 'No ')}\n \n"
        formatted_history += f"**Timestamp:** {record.get('m.timestamp', 'No timestamp')}\n \n"
        formatted_history += "---\n\n\n"
    
    return formatted_history

def create_cyto_graph_data(url, username, password, user_id):
    this = get_graph_data(url, username, password, user_id)
    if this != []:
        graph_nodes = []
        graph_edges = []
        
        for record in this:
            # For source nodes
            source_text = record['text'] if record['source_labels'][0] == 'Document' else None
            if record['source'] not in [node[0] for node in graph_nodes]:
                # Store tuple of (id, display_name, node_type, text)
                graph_nodes.append((
                    record['source'],                    # id
                    record['source_labels'][0],          # display_name (for source nodes, use label)
                    record['source_labels'][0],          # node_type (for styling)
                    source_text                          # text (only for Documents)
                ))
            
            # For target nodes
            target_text = record['text'] if record['target_labels'][0] == 'Document' else None
            if record['target'] not in [node[0] for node in graph_nodes]:
                graph_nodes.append((
                    record['target'],                    # id
                    record['target_id'] or record['target'],  # display_name (prefer id)
                    record['target_labels'][0],          # node_type (for styling)
                    target_text                          # text (only for Documents)
                ))
            
            graph_edges.append(
                (record['source'], record['target'], record['relationship_type'])
            )
            
            # For related nodes
            if record['related_target'] and record['related_target'] not in [node[0] for node in graph_nodes]:
                graph_nodes.append((
                    record['related_target'],            # id
                    record['related_id'] or record['related_target'],  # display_name
                    record['related_target_labels'][0],  # node_type (for styling)
                    None                                 # text (not available for related nodes)
                ))
            if record['related_target'] and (record['target'], record['related_target']) not in [(edge[0], edge[1]) for edge in graph_edges]:
                graph_edges.append(
                    (record['target'], record['related_target'], record['related_relationship_type'])
                )
                           
    return graph_nodes, graph_edges


#collect data for Entity Relationship Plot
def get_graph_data(url, user, password, user_id):
    driver = GraphDatabase.driver(url, auth=(user, password))
    with driver.session() as session:
        result = session.run("""
MATCH (n:!Chunk)-[r]->(m) 
        WHERE n.user = $user_id
        OPTIONAL MATCH (m)-[r2]->(o)
        RETURN elementId(n) AS source, elementId(m) AS target,
               labels(n) AS source_labels, labels(m) AS target_labels,
               type(r) AS relationship_type, n.text AS text, m.id as target_id, o.id as related_id,
               elementId(o) AS related_target, labels(o) AS related_target_labels, type(r2) AS related_relationship_type
        """, parameters={"user_id": user_id})
        
        return [record for record in result]

def gen_entity_graph(user_id = 'default'):
    my_nodes, my_edges = create_cyto_graph_data(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, user_id)
    default_stylesheet, nodes, edges, all_elements = create_cyto_elements(my_nodes, my_edges)
    edges_to_select = list(set(edge['data']['label'] for edge in edges))

    this = [
        dcc.Store(id='all-elements', storage_type='memory', data=all_elements),
        dcc.Store(id='all-node-elements', storage_type='memory', data=nodes),
        
        html.Label("Select Relationships", "edge-label", style={'color':'green'}),
        dcc.Dropdown(
            id='select-edges-dropdown',
            value=edges_to_select,
            clearable=False,
            multi=True,
            options=[
                {'label': html.Span(name.capitalize(), style={'color':'green', 
                                                              'border-color':'green', 
                                                              'background-color':'white',
                                                              'multi-value': {
                                                                    'background-color': 'white',  # Green background for selected items
                                                                    'color': 'green'
                                                               },
                                                               'multi-value-label': {
                                                               'color': 'white'
                                                               },
                                                               'multi-value-remove': {
                                                                    'color': 'black',
                                                                    'background-color': 'white',
                                                                    ':hover': {
                                                                    'background-color': 'black',
                                                                    'color': 'white'}
                                                               }
                                                            }
                                                              ),'value':name}
                for name in edges_to_select
            ],
            style={"color":"black",
                   "background-color":"white",
                   "border-color":"black"}
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
            style={'width': '100%',
                   'height': '750px', 
                   'color':'green', 
                   'background-color': 'mintcream'
                   }
        ),

    ]
    return this
    
def create_cyto_elements(graph_nodes, graph_edges):
    label_to_class = {
        'Document': 'Document',
        'Mainparticipant':'Mainparticipant',
        'Valuebasedgoal': 'Valuebasedgoal',
        'Actionstep': 'Actionstep',
        'Mindset': 'Mindset',
        'Referencedindividual': 'Referencedindividual',
        'Obstacle': 'Obstacle',
        'Domainmindset': 'Domainmindset',
        'Solution': 'Solution',
        'Performancemetric': 'Performancemetric',
        'Corevalue':'Corevalue',
        'Intervention':'Intervention',
        'Otherentity':'Otherentity'
    }
    
    nodes = [
        {
            'data': {
                'id': id,
                'label': display_name,    # Use display_name for label
                'node_type': node_type,   # Store node_type for styling
                'text': text              # Only present for Documents
            },
            'classes': label_to_class.get(node_type, 'default_class')  # Use node_type for styling
        }
        for id, display_name, node_type, text in graph_nodes
    ]

    edges = [
        {
            'data': {'source': source, 'target': target, 'label': label}
        }
        for source, target, label in graph_edges
    ]
    
    all_elements = edges + nodes

    default_stylesheet = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',  # Now uses the display_name
                'border-width': 1,
                'shape': 'ellipse',
                'width': 25,
                'opacity': 0.5,
                'text-opacity': 1,
                'text-halign': 'center',
                'text-valign': 'center',
            }
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
            'selector': '.Mainparticipant',
            'style': {
                'background-color': 'yellow',
            }
        },
        {
            'selector': '.Valuebasedgoal',
            'style': {
                'background-color': 'green',
            }
        },
        {
            'selector': '.Mindset',
            'style': {
                'background-color': 'red',
            }
        },
        {
            'selector': '.Corevalue',
            'style': {
                'background-color': 'purple',
            }
        },
        {
            'selector': '.Domainmindset',
            'style': {
                'background-color': 'navy',
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
            'selector': '.Intervention',
            'style': {
                'background-color': 'darkgreen',
            }          
        },
        {
            'selector': '.Actionstep',
            'style': {
                'background-color': 'indigo',
            }          
        },
        {
            'selector': '.Performancemetric',
            'style': {
                'background-color': 'lavender',
            }          
        },
        {
            'selector': '.Referencedindividual',
            'style': {
                'background-color': 'pink',
            }          
        },
        {
            'selector': '.Otherentity',
            'style': {
                'background-color': 'Orange',
            }          
        },
        
    ]
    return default_stylesheet, nodes, edges, all_elements
    ##### End cytoscape layout

