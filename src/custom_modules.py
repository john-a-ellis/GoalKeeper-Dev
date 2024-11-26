from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
# from pydantic import BaseModel
from datetime import datetime
from typing import List
import os, strip_markdown

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
            elapsed_chat_time = datetime.fromisoformat(elapsed_chat_time.get('most_recent_timestamp')) 
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
        print(f"An error occurred performing a vector similarity search: {e}")

def update_graph_memory(graph_transformer, graph_database, embedding_model, user_id: str, content: str, type: str):
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

def get_structured_chat_history(graph_database, user_id: str = 'default', limit: int = 100) -> str:
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
    # summary_prompt = read_prompt('initial_prompt_template')
    today = datetime.now()

    summary_prompt = ["system", f"""
    Today is {today} and its been {elapsed_time}, since you last chatted with the user.      
    1. You are a helpful AI driven performance coach and expert in neuroscience and the growth mindset. 
    2. Your purpose is to help human users achieve the goals they identify through the application of neuroscience and the growth mindset.
    3. Your name is Mel (a Mindset-oriented, Eidetic, Librarian)
    4. If no chat sessions are available you are meeting the user for the first time so ask the user how they would like you to address them. 
    5. Only introduce yourself if chat sessions are not available. If its been more than one day since chatting with the user, welcome them back.
    Otherwise if chat sessions are available but only if chat sessions are available:
    1. Summarize them in one or two sentences and recommend a next step, then ask how the human user would like to proceed.

   

    Chat Sessions:
    {sessions}
    """]
    summary = llm.invoke(summary_prompt).content
    return summary
