from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
# from pydantic import BaseModel
from datetime import datetime
from typing import List
import os, strip_markdown, asyncio, logging
from tenacity import retry, stop_after_attempt, wait_exponential
from src.reddit import RedditConversionTracker
from src.Neo4j_customs import SingleEntitySinglePropertyExactMatchResolver
import asyncio

# # Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the clients
#Set up RedditConversionTracking
reddit_client = RedditConversionTracker(ad_account_id=os.environ["REDDIT_PIXEL_ID"],
                                        access_token=os.environ["GOALKEEPER_CONVERSION"])
# Wrapper to make it sync
def sync_wrapper(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# Add retry decorator for embedding generation
@retry(
    stop=stop_after_attempt(3),  # Try 3 times
    wait=wait_exponential(multiplier=1, min=4, max=10)  # Exponential backoff
)
def generate_embedding_with_retry(embedding_model, text):
    try:
        # Langchain's embed_documents method
        return embedding_model.embed_documents([text])[0]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise



def read_prompt(prompt_name: str) -> List[SystemMessage]:
    """
    Reads a prompt file and returns it as a SystemMessage.
    
    Args:
        prompt_name (str): Name of the prompt file to read (without .txt extension)
        
    Returns:
        List[SystemMessage]: A list containing a single SystemMessage
    """
    try:
        with open(f'assets/prompts/{prompt_name}.txt', 'r') as file:
            prompt_text = file.read()
            # Remove "system," prefix if it exists in the text file
            if prompt_text.startswith("system,"):
                prompt_text = prompt_text[7:]
            return [SystemMessage(content=prompt_text)]
    except FileNotFoundError:
        print(f'The {prompt_name} does not exist')
        # Return empty system message or handle error as needed
        return [SystemMessage(content="")]
    except IOError:
        print(f'An error occurred trying to read the {prompt_name}')
        # Return empty system message or handle error as needed
        return [SystemMessage(content="")]
    
def get_user_id(auth_data):
    if os.getenv('DEPLOYED', 'False').lower() == 'true':
        user_id = auth_data.get('user_info', {}).get('email', 'User')
        pass
    else: 
        user_id='default'
        pass
    return user_id

def get_user_name(auth_data):
    if os.getenv('DEPLOYED', 'False').lower() =='true':
        user_name = auth_data.get('user_info', {}).get('email', 'User')
        pass
    else:
        user_name='default'
        pass
    return user_name

def get_elapsed_chat_time(graph_database, user_id): 
    now = datetime.now() 
    try: 
        user_id_str = f"'{user_id}'" if isinstance(user_id, str) else user_id 
        query = f""" 
        MATCH (d:Document) 
        WHERE d.user = {user_id_str} 
        RETURN MAX(d.timestamp) AS most_recent_timestamp """ 
        result = graph_database.query(query) 
        elapsed_chat_time = result[0]
        if elapsed_chat_time is None: 
            elapsed_chat_time = None 
        else: # Convert elapsed_chat_time to a datetime object if needed 
            elapsed_chat_time = datetime.fromisoformat(str(elapsed_chat_time.get('most_recent_timestamp'))) 
            elapsed_chat_time = now - elapsed_chat_time 
    except Exception as e: 
        print(f"An error occurred finding elapsed_chat_time: {e}") 
        elapsed_chat_time = None 
        
    return elapsed_chat_time
                         
def retrieve_vector_memory(memory_vector_store, user_id: str, query: str, k: int = 4):
    ### retrieves x messages from vector memory using similarity search
    try:
        results = memory_vector_store.similarity_search(
            query=query,
            k=k,
            filter={"user": user_id}
        )
        return [doc.page_content for doc in results]

    except Exception as e:
        print(f"An error occurred performing a Message vector similarity search: {e}")

async def async_update_graph_memory(graph_transformer, graph_database, graph_driver, embedding_model, user_id: str, content: str, type: str):
    try:
        # Strip markdown from the content
        this = strip_markdown.strip_markdown(content)
        
        if this is None:
            logger.warning("strip_markdown returned None, using original content")
            this = content

        # Create a document object from the current message
        document = Document(page_content=this, metadata={
            "source": type,
            "user": user_id,
            "id": None,
            "timestamp": datetime.now().isoformat()
        })
        print(f"this is the document: {document}")

        documents = [document]
        # Find document nodes previously created containing documents not processed and add them to a list for reprocessing
        try:
    
            documents_to_reprocess = find_missed_document_nodes(graph_database, user_id)
            documents += documents_to_reprocess

        except Exception as e:
            print(f"an exception occured processing: find_missed_document_nodes: {e}")

        # Process the document(s) into a graph document(s)
        for document in documents:
            graph_document = process_response_with_user(graph_transformer, document, user_id)

            # Add the graph document to the Neo4j graph database
            try:
                graph_database.add_graph_documents(
                   [graph_document],
                    baseEntityLabel=False,
                    include_source=True
               )
            except Exception as e:
                (f"An error occurred adding a graph document: {e}")
            
        try:
            #identify duplicates in Mainparticpant nodes and Resolve them
            resolve_node_type = 'Mainparticipant'
            resolve_property = 'participant_type'

            filter=f"WHERE entity.user='{user_id}'"

            resolver = SingleEntitySinglePropertyExactMatchResolver(driver=graph_driver,
                                                                    resolve_node_type=resolve_node_type,
                                                                    filter_query=filter,
                                                                    resolve_property=resolve_property)
            
            res = await resolver.run()
            if res: print(res)
            
            res = ""
            resolve_property ='name'
            resolver = SingleEntitySinglePropertyExactMatchResolver(driver=graph_driver,
                                                                    resolve_node_type=resolve_node_type,
                                                                    filter_query=filter,
                                                                    resolve_property=resolve_property)
            res = await resolver.run()
            if res: print(res)

        except Exception as e:
            print(f"An error occurred performing a Single Entity Single Propery Resolution: {e}")

        
        
        # Refresh the Neo4j schema
        graph_database.refresh_schema()

        # Query for document nodes without embeddings
        document_nodes = graph_database.query("""
            MATCH (n:Document)
            WHERE n.embedding IS NULL
            RETURN n.id AS node_id, n.text AS text
        """)

        # Embed the text property and add the embedding to the document node properties
        for document_node in document_nodes:
            node_id = document_node["node_id"]
            text = document_node["text"]

            try:
                # Use the retry-enabled embedding generation
                document_embedding = generate_embedding_with_retry(embedding_model, text)
                
                # Convert the embedding to a flat list
                flat_embedding = [float(value) for value in document_embedding]
                
                # Update the node properties with the new embedding
                graph_database.query("""
                    MATCH (n:Document)
                    WHERE n.id = $nodeid
                    SET n.embedding = $embedding
                    RETURN n.id, n.embedding
                """, params={"nodeid": node_id, "embedding": flat_embedding})

            except Exception as inner_e:
                logger.error(f"Error processing embedding for node {node_id}: {inner_e}")
                continue
            
    except Exception as e:
        logger.error(f"An error occurred creating graph document: {e}", exc_info=True)


def process_response_with_user(graph_transformer, document, user:str):
    """Appends a 'user' property to each node created in the graph document assigning it to the user logged in"""
    
    # Generate the graph document using the original method
    graph_document = graph_transformer.process_response(document)
    
    # Modify nodes to include user property
    for node in graph_document.nodes:
        node.properties['user'] = user
    
    return graph_document

update_graph_memory = sync_wrapper(async_update_graph_memory)
    
def find_missed_document_nodes(graph_database, user_id: str):
    """Finds all document nodes without children"""
    filter_query = """MATCH (d:Document) 
                      WHERE d.user = $user_id and not (d)--() 
                      RETURN d """
    params = {"user_id": user_id}

    nodes_to_reprocess = graph_database.query(filter_query, params)
    if nodes_to_reprocess is not None:
        documents = []
        for node in nodes_to_reprocess:
            document = Document(page_content=node['d']['text'], metadata={
            "source": node['d']['source'],
            "user": node['d']['user'],
            "reprocessed": datetime.now().isoformat(),
            "reprocessed_user": user_id,
            "timestamp": node['d']['timestamp']
            })
            documents.append(document)
    
    return documents
    



def fetch_neo4j_memory(graph_database, user_id='default', limit=1000):
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

def get_structured_chat_history(graph_database, user_id: str = 'default', limit: int = 1000) -> str:
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



def summarize_sessions(elapsed_time, sessions, llm):

    today = datetime.now()

    summary_prompt = ["system", f"""
    Today is {today} and its been {elapsed_time}, since you last chatted with the user.  
     
    1. You are a helpful AI driven performance coach and expert in neuroscience and the growth mindset. 
    2. Your purpose is to help human users achieve the goals they identify through the application of neuroscience and the growth mindset.
    3. Your name is Mel (a Mindset-oriented, Educator and Learner)
    4. If no chat sessions are available you are meeting the user for the first time so ask the user how they would like you to address them. 
    5. Only introduce yourself if chat sessions are not available. 
    6. If its been more than one day since chatting with the user, welcome them back.
    7. Only welcome them back if it has been more than 24 hours since you last chatted with them.
    8. Address the user by name.
    9. If chat sessions are available but ONLY if chat sessions are available summarize them in one or two sentences
       and recommend a next step, then ask how the human user would like to proceed.

    Render any results in markdown in the voice, style, and manner of Mel Robbins although you are not Mel Robbins.

    Chat Sessions:
    {sessions}
    """]
    summary = llm.invoke(summary_prompt).content
    return summary

def register_new_user(username: str, email: str, campaign:str) -> None:
    """
    Example user registration method
    
    :param username: User's unique username
    :param email: User's email address
    """
    try:
        # Simulate user registration process
        print(f"Registering user: {username}")
        
        # Send registration event to Reddit
        response = reddit_client.send_conversion_event(
            event_type = 'other',
            test_mode=False,
            custom_event_name=campaign,
            user_data={'email':email,
                       'username':username},
            event_metadata={'start':'2024-12-03',
                            'stop':'2024-12-31'}
        )
        print("Registration event tracking successful:", response)
    
    except Exception as e:
        print(f"Error tracking user registration: {e}")

def is_existing_user(graph_database, user_id: str) -> bool:
    query = f"""
    MATCH (n) WHERE n.user = '{user_id}'
    RETURN DISTINCT n.user
    """
    result = graph_database.query(query)
    if len(result)>0:
        return True
    else:
        return False

