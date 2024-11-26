#collect data for Entity Relationship Plot
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_cytoscape as cyto

def get_graph_data(graph_database, user_id):
    # driver = GraphDatabase.driver(url, auth=(user, password))
    # with graph_database as session:
    result = graph_database.query(f"""
MATCH (n:!Chunk)-[r]->(m) 
        WHERE n.user = '{user_id}'
        OPTIONAL MATCH (m)-[r2]->(o)
        RETURN elementId(n) AS source, elementId(m) AS target,
               labels(n) AS source_labels, labels(m) AS target_labels,
               type(r) AS relationship_type, n.text AS text, m.id as target_id, o.id as related_id,
               elementId(o) AS related_target, labels(o) AS related_target_labels, type(r2) AS related_relationship_type
        """)
        
    return [record for record in result]

def gen_entity_graph(graph_database, user_id = 'default'):
    my_nodes, my_edges = create_cyto_graph_data(graph_database, user_id)
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

def create_cyto_graph_data(graph_database, user_id):
    this = get_graph_data(graph_database, user_id)
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

