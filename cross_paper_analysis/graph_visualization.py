
import json
import networkx as nx
import random
import plotly.graph_objects as go

def layout_algorithm(json_data):
    """
    Implements the Twin-Mirror Layered Layout Algorithm.
    """
    G = nx.Graph()
    
    # 1. Parse Nodes and assign attributes
    for r_id, r_data in json_data.items():
        if not r_id.startswith("researcher"): continue
        
        x_bias = 0.3 if r_id == "researcher_1" else 0.7
        
        hlg = r_data.get('hlg_data', {})
        for level, nodes in hlg.items():
            if not level.startswith("Level"): continue
            
            if level == "Level3": y_bias = 0.9
            elif level == "Level2": y_bias = 0.5
            else: y_bias = 0.1
            
            for node_name in nodes:
                if not G.has_node(node_name):
                    G.add_node(node_name, 
                               researcher=r_id, 
                               level=level, 
                               x_target=x_bias, 
                               y_target=y_bias,
                               importance=1)

    # 2. Parse Internal Edges
    for r_id, r_data in json_data.items():
        if not r_id.startswith("researcher"): continue
        for rel in r_data.get('hlg_data', {}).get('Relations', []):
            if G.has_node(rel['source']) and G.has_node(rel['target']):
                G.add_edge(rel['source'], rel['target'], 
                           type='internal', 
                           weight=rel.get('confidence', 1),
                           desc=rel.get('explanation', ''))
            
    # 3. Parse Cross-Researcher Edges
    cross_rels = json_data.get('cross_researcher_relations', [])
    for rel in cross_rels:
        if G.has_node(rel['source']) and G.has_node(rel['target']):
            G.add_edge(rel['source'], rel['target'], 
                       type='cross_researcher', 
                       weight=rel.get('confidence', 1) * 3,
                       desc=rel.get('explanation', ''))
            G.nodes[rel['source']]['importance'] += 5
            G.nodes[rel['target']]['importance'] += 5

    # 4. Compute Layout
    pos = {}
    for node, data in G.nodes(data=True):
        pos[node] = [
            data['x_target'] + random.uniform(-0.05, 0.05),
            data['y_target'] + random.uniform(-0.05, 0.05)
        ]
    
    # Use spring layout with different k values implicitly by pre-adjusting positions?
    # No, networkx spring_layout uses a single k. 
    # Strategy: Run layout globally but with a larger k (repulsion) to spread R1,
    # then manually compress R2 if needed, or rely on constraints.
    # Let's increase k generally to spread nodes out.
    pos = nx.spring_layout(G, pos=pos, fixed=None, k=0.4, iterations=100, weight='weight')
    
    # 5. Post-process to enforce constraints
    for node, coords in pos.items():
        data = G.nodes[node]
        
        if data['researcher'] == 'researcher_1':
            # R1 has fewer nodes (61), give them more room to breathe in the X-axis
            # Spread them out: min 0.1, max 0.45
            coords[0] = min(0.45, max(0.1, coords[0]))
        else:
            # R2 has many nodes (290), keep them tighter
            # min 0.55, max 0.9
            coords[0] = max(0.55, min(0.9, coords[0]))
            
        if data['level'] == 'Level3':
            coords[1] = max(0.75, min(1.0, coords[1]))
        elif data['level'] == 'Level2':
            coords[1] = max(0.35, min(0.65, coords[1]))
        else:
            coords[1] = max(0.0, min(0.25, coords[1]))
            
    return G, pos

def visualize_graph(G, pos):
    """
    Renders the graph using Plotly.
    """
    edge_traces = {
        'internal': go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'),
        'cross_researcher': go.Scatter(x=[], y=[], line=dict(width=2, color='gold'), hoverinfo='text', mode='lines', text=[])
    }
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_type = edge[2].get('type', 'internal')
        
        trace = edge_traces.get(edge_type, edge_traces['internal'])
        trace['x'] += tuple([x0, x1, None])
        trace['y'] += tuple([y0, y1, None])
        if edge_type == 'cross_researcher':
            trace['text'] += tuple([edge[2].get('desc', '')])


    node_traces = {
        'researcher_1': go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                                   marker=dict(showscale=False, colorscale='YlGnBu', reversescale=True, color=[],
                                               size=[], opacity=[], sizemin=1, symbol='circle', line=dict(width=0))),
        'researcher_2': go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                                   marker=dict(color=[], size=[], opacity=[], symbol='circle', line=dict(width=0)))
    }
    
    level_colors = {'Level1': '#9B9B9B', 'Level2': '#50C878', 'Level3': '#E74C3C'}

    # Sort nodes by importance descending so larger nodes are drawn first (background)
    # and smaller nodes are drawn last (foreground) to prevent occlusion
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('importance', 1), reverse=True)

    for node, data in sorted_nodes:
        x, y = pos[node]
        researcher = data.get('researcher')
        trace = node_traces.get(researcher)
        if not trace: continue

        trace['x'] += tuple([x])
        trace['y'] += tuple([y])
        trace['text'] += tuple([f"{node}<br>Level: {data.get('level')}<br>Importance: {data.get('importance')}"])
        
        # Adjust size and opacity based on importance
        # User request: Smaller dots, no outline, use opacity for importance
        base_size = 3
        importance = data.get('importance', 1)
        # Reduce size variation
        size = base_size + importance * 1.5 
        
        # Opacity logic: Low importance -> very transparent
        opacity = 0.2 if importance <= 1 else 0.9

        trace['marker']['color'] += tuple([level_colors.get(data.get('level'))])
        trace['marker']['size'] += tuple([size])
        trace['marker']['opacity'] += tuple([opacity])
       
        
    fig = go.Figure(data=[edge_traces['internal'], edge_traces['cross_researcher'], 
                           node_traces['researcher_1'], node_traces['researcher_2']],
                    layout=go.Layout(
                        title=dict(text='Cross-Paper Knowledge Graph Visualization', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Add horizontal separator lines & annotations
    fig.update_layout(
        shapes=[
            go.layout.Shape(
                type="line", xref="paper", yref="paper",
                x0=0, y0=0.7, x1=1, y1=0.7,
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot")
            ),
            go.layout.Shape(
                type="line", xref="paper", yref="paper",
                x0=0, y0=0.3, x1=1, y1=0.3,
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot")
            )
        ],
        annotations=list(fig.layout.annotations) + [
            go.layout.Annotation(
                x=0.01, y=0.95, xref="paper", yref="paper",
                text="<b>Level 3: Problems & Goals</b>", showarrow=False,
                xanchor="left", font=dict(size=10, color="grey")
            ),
            go.layout.Annotation(
                x=0.01, y=0.65, xref="paper", yref="paper",
                text="<b>Level 2: Frameworks & Models</b>", showarrow=False,
                xanchor="left", font=dict(size=10, color="grey")
            ),
            go.layout.Annotation(
                x=0.01, y=0.25, xref="paper", yref="paper",
                text="<b>Level 1: Techniques</b>", showarrow=False,
                xanchor="left", font=dict(size=10, color="grey")
            )
        ]
    )

    output_path = "d:/workspace/ResearchAgent/cross_paper_analysis/cross_paper_visualization_2.png"
    fig.write_image(output_path, scale=2)
    print(f"Graph visualization saved to {output_path}")

if __name__ == "__main__":
    with open('d:/workspace/ResearchAgent/cross_paper_analysis/cross_于洋李想_1217.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    graph, positions = layout_algorithm(data)
    visualize_graph(graph, positions)
