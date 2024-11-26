from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.graphs.graph_document import GraphDocument
from typing import Dict, List, Optional
from langchain_experimental.graph_transformers import LLMGraphTransformer
import json

def create_graph_aware_transformer(
    llm,
    prompt: PromptTemplate,
    allowed_nodes: List[str],
    allowed_relationships: List[str],
    node_properties: Dict,
    relationship_properties: Dict,
    graph_database: Neo4jGraph,
    strict_mode: bool = True
):
    """Factory function to create a graph-aware transformer with enhanced context."""
    
    def get_existing_nodes_context() -> str:
        """Retrieve context about existing nodes from the graph database."""
        # Query for MainParticipants
        participants_query = """
        MATCH (p:MainParticipant)
        RETURN p.participant_id as id, p.name as name, p.participant_type as type
        """
        
        # Query for ValueBasedGoals
        goals_query = """
        MATCH (g:ValueBasedGoal)
        RETURN g.title as title, g.description as description
        """
        
        # Query for CoreValues
        values_query = """
        MATCH (v:CoreValue)
        RETURN v.name as name, v.domain as domain
        """
        
        # Execute queries and format results
        participants = graph_database.query(participants_query)
        goals = graph_database.query(goals_query)
        values = graph_database.query(values_query)
        
        context = f"""
        Existing Graph Context:
        
        1. Main Participants:
        {json.dumps(participants, indent=2)}
        
        2. Existing Goals:
        {json.dumps(goals, indent=2)}
        
        3. Core Values:
        {json.dumps(values, indent=2)}
        
        When creating new nodes and relationships:
        - Reference existing participants by their participant_id
        - Check if goals/values align with existing ones before creating new nodes
        - Maintain consistency with existing terminology and classifications
        - If referencing existing entities, use their exact names/identifiers
        """
        return context

    # Enhance the original prompt with graph context
    graph_context = get_existing_nodes_context()
    enhanced_prompt = prompt.template.replace(
        "Text to Process:",
        f"{graph_context}\n\nText to Process:"
    )
    
    # Create new prompt template with the enhanced content
    enhanced_prompt_template = PromptTemplate(
        template=enhanced_prompt,
        input_variables=prompt.input_variables
    )

    # Create the base transformer with enhanced prompt
   
    base_transformer = LLMGraphTransformer(
        llm=llm,
        prompt=enhanced_prompt_template,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
        strict_mode=strict_mode
    )

    # Wrap the transform method to include validation
    original_transform = base_transformer.process_response

    def enhanced_transform(input_text: str) -> GraphDocument:
        # Get the graph document from original transform
        graph_doc = original_transform(input_text)
        
        # Validate nodes against existing ones
        validated_nodes = []
        for node in graph_doc.nodes:
            if node.get("type") == "MainParticipant":
                existing = graph_database.query(
                    """
                    MATCH (p:MainParticipant)
                    WHERE p.participant_id = $participant_id
                    RETURN p
                    """,
                    {"participant_id": node.get("participant_id")}
                )
                if existing:
                    # Use existing node reference
                    node["id"] = existing[0]["p"].id
            validated_nodes.append(node)
        
        # Update the graph document with validated nodes
        graph_doc.nodes = validated_nodes
        return graph_doc

    # Replace the transform method
    base_transformer.transform = enhanced_transform
    
    return base_transformer