import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import textwrap
import numpy as np

def visualize_proposal_graph(proposal_data, output_path):
    """
    Visualizes a research proposal using a 7-region layout with a high-quality "infographic" style.
    """
    
    # --- 1. Setup Data & Graph ---
    G = nx.Graph()
    topic_name = proposal_data.get('title', 'Topic')
    
    # Add Nodes
    # Main Topic
    G.add_node(topic_name, type='topic', group='center', level='main_topic')
    
    # Sub-concepts
    for concept in proposal_data.get('sub_concepts', []):
        G.add_node(concept, type='sub_concept', group='center', level='sub_concept')
        # Ensure edge exists (Topic <-> Sub-concept)
        if not any(c['source'] == concept and c['target'] == topic_name for c in proposal_data.get('connections', [])) and \
           not any(c['target'] == concept and c['source'] == topic_name for c in proposal_data.get('connections', [])):
             G.add_edge(topic_name, concept)

    # Researcher Nodes
    for node in proposal_data.get('r1_nodes', []):
        G.add_node(node['name'], type='r1_node', group='r1', level=node['level'])
        
    for node in proposal_data.get('r2_nodes', []):
        G.add_node(node['name'], type='r2_node', group='r2', level=node['level'])
        
    # Add Edges from JSON
    for conn in proposal_data.get('connections', []):
        src = conn['source']
        tgt = conn['target']
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt)

    # --- 2. Calculate Layout (Fixed Regions) ---
    pos = {}
    
    # Region X-Centers
    x_centers = {
        'r1_Level3': 0.1, 'r1_Level2': 0.23, 'r1_Level1': 0.36,
        'center': 0.5,
        'r2_Level1': 0.64, 'r2_Level2': 0.77, 'r2_Level3': 0.9
    }
    
    # Canvas Size
    width, height = 20, 12
    
    # Identify Center Sub-concepts first
    center_sub = []
    for node, data in G.nodes(data=True):
        if data.get('group') == 'center' and data.get('type') == 'sub_concept':
            center_sub.append(node)
            
    # 2.1 Place Center Nodes First
    
    # A. Place Sub-concepts (Upper Center)
    n_sub = len(center_sub)
    sub_y_positions = {} # Map sub_concept name -> y
    
    if n_sub > 0:
        # Distribute vertically in Upper Center (0.6 to 0.9)
        # Sort them? Maybe alphabetically or just as is.
        ys = np.linspace(0.85, 0.65, n_sub) if n_sub > 1 else [0.75]
        
        for i, node in enumerate(center_sub):
            # Center X with staggered offset
            # Center band is 0.43 to 0.57. Center is 0.5.
            # Stagger between 0.47 and 0.53
            offset = 0.03 if i % 2 == 0 else -0.03
            x = 0.5 + offset
            y = ys[i]
            pos[node] = (x, y)
            sub_y_positions[node] = y

    # B. Place Main Topic (Lower Center)
    pos[topic_name] = (0.5, 0.35) # Fixed Lower Center

    # 2.2 Place Researcher Nodes
    
    # Helper to find which sub-concept a node connects to
    def get_connected_sub_concept(node_name):
        for sub in center_sub:
            if G.has_edge(node_name, sub):
                return sub
        return None

    # Group remaining nodes by region
    region_nodes = {k: [] for k in x_centers.keys()}
    
    # Pre-assign Y for connected nodes
    connected_nodes_y = {}
    
    for node, data in G.nodes(data=True):
        group = data.get('group')
        level = data.get('level')
        if group == 'center': continue
        
        # Check connection to sub-concept
        connected_sub = get_connected_sub_concept(node)
        
        if connected_sub:
            # Align Y with sub-concept
            target_y = sub_y_positions[connected_sub]
            connected_nodes_y[node] = target_y
            
            # Set Position Immediately
            key = f"{group}_{level}"
            x = x_centers.get(key, 0.1 if group == 'r1' else 0.9)
            pos[node] = (x, target_y)
            
        else:
            # Add to list for distribution later
            key = f"{group}_{level}"
            if key in region_nodes:
                region_nodes[key].append(node)

    # Distribute unconnected nodes (Lower Half)
    # Spread them out more to create a "Fan" effect towards the center topic
    for key, nodes in region_nodes.items():
        n = len(nodes)
        if n == 0: continue
        x = x_centers[key]
        
        # Distribute in Lower Half (0.15 to 0.55) 
        # But vary the range based on distance from center to make it look organic?
        # Simpler: just use full available vertical space below sub-concepts
        # Sub-concepts are > 0.65. So we have 0.1 to 0.6 safely.
        
        if n == 1:
            ys = [0.35]
        else:
            ys = np.linspace(0.55, 0.15, n)
        
        for i, node in enumerate(nodes):
            pos[node] = (x, ys[i])

    # --- 3. Draw with Matplotlib ---
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # A. Draw Background Bands
    # ... (Keep existing colors)
    bounds = [0.0, 0.165, 0.30, 0.43, 0.57, 0.70, 0.835, 1.0]
    bg_colors = ['#ffebee', '#fff0f0', '#fff8f8', '#f8f0fc', '#f8fbff', '#eef6ff', '#e3f2fd']
    
    for i in range(7):
        ax.axvspan(bounds[i], bounds[i+1], color=bg_colors[i], alpha=0.6, zorder=0)
        
    # B. Draw Edges (Curved & Styled)
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Determine Edge Type
        # 1. R -> Sub (Horizontal Aligned)
        # 2. Sub -> Main (Vertical/Aggregating)
        # 3. R -> Main (Direct Fan)
        
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        is_sub_connection = 'sub_concept' in [u_data.get('type'), v_data.get('type')]
        is_main_connection = 'topic' in [u_data.get('type'), v_data.get('type')]
        
        # Default Style
        color = '#999999'
        rad = 0.1
        style = "-"
        width_edge = 1.5
        alpha = 0.7
        
        # Case 1: Sub -> Main (The "Bundled" lines)
        if is_sub_connection and is_main_connection:
            color = '#8e24aa' # Purple
            width_edge = 2.0
            # Curve slightly inward
            # If sub is above (y1 > y2), curve out then in? Or just straight?
            # User asked to "change style". Let's make them slightly curved.
            rad = 0.1 
            # Or use a different arrow style?
            
        # Case 2: R -> Sub (The "Bridge" lines)
        elif is_sub_connection and not is_main_connection:
            # Color based on R side
            if u_data['group'] == 'r1' or v_data['group'] == 'r1':
                color = '#ef5350'
            else:
                color = '#42a5f5'
            
            # Straight if aligned
            if abs(y1 - y2) < 0.01:
                rad = 0.0
            else:
                rad = 0.1
                
        # Case 3: R -> Main (The "Fan" lines)
        elif is_main_connection:
            # Color based on R side
            if u_data['group'] == 'r1' or v_data['group'] == 'r1':
                color = '#ef5350' # Red
            else:
                color = '#42a5f5' # Blue
            
            # Fan Curve
            # If node is to the left (R1), curve up/down to center
            # Target is (0.5, 0.35)
            # If y_node > y_topic: curve down (rad > 0 for left->right?)
            # Let's calibrate rad based on X direction
            
            # x1 is source, x2 is target.
            # We want a nice "arc"
            if x1 < x2: # Left to Center
                rad = -0.2 # Curve Upwards? 
                if y1 > y2: rad = 0.2 # If source is higher, curve "under"? 
                # Actually, simple arc3 works well.
                # Let's try consistent curvature.
                rad = 0.15
            else: # Right to Center
                rad = -0.15
                
        else:
            # Other internal connections?
            rad = 0.1

        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            arrowstyle="-",
            linewidth=width_edge,
            alpha=alpha,
            zorder=1,
            mutation_scale=15
        )
        ax.add_patch(arrow)
        
    # C. Draw Nodes
    for node, (x, y) in pos.items():
        data = G.nodes[node]
        ntype = data.get('type')
        group = data.get('group')
        
        # Node Style
        if ntype == 'topic':
            # Large Rectangle
            box_style = patches.BoxStyle("Round", pad=0.1)
            # Center it?
            # FancyBboxPatch takes (x,y) as bottom-left corner usually? 
            # No, we can center it.
            
            # Wrapper for text
            wrapped_text = "\n".join(textwrap.wrap(node, width=15))
            
            # Draw Text with Bbox
            t = ax.text(x, y, wrapped_text, 
                    ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=1", fc="#7b1fa2", ec="#4a148c", lw=2),
                    zorder=10)
            
        else:
            # Circles for others
            # Determine color/size
            if group == 'r1':
                fc = 'white'
                ec = '#e53935' # Red 600
                tc = 'black'
            elif group == 'r2':
                fc = 'white'
                ec = '#1e88e5' # Blue 600
                tc = 'black'
            else: # sub_concept
                fc = '#f3e5f5' # Purple 50
                ec = '#8e24aa' # Purple 600
                tc = '#4a148c'
                
            size = 0.035 # Radius
            
            # Draw Circle
            circle = patches.Circle((x, y), radius=size, fc=fc, ec=ec, lw=2, zorder=5)
            # Scale aspect ratio?
            # ax is 0-1, 0-1. But figure is 20x12. 
            # Circle will look like ellipse if not corrected.
            # Matplotlib 'Circle' is in data coordinates.
            # We can use Scatter for true circles, OR adjust radius.
            # Better: use Bbox with Circle style for text.
            
            wrapped_text = "\n".join(textwrap.wrap(node, width=10))
            
            ax.text(x, y, wrapped_text,
                   ha='center', va='center',
                   fontsize=9, color=tc,
                   bbox=dict(boxstyle="circle,pad=0.5", fc=fc, ec=ec, lw=2),
                   zorder=10)

    # D. Labels for Regions (Bottom)
    labels = [
        (0.08, "R1\nGoals (L3)"), (0.23, "R1\nFrameworks (L2)"), (0.36, "R1\nTechniques (L1)"),
        (0.5, "Collaboration Zone"),
        (0.64, "R2\nTechniques (L1)"), (0.77, "R2\nFrameworks (L2)"), (0.92, "R2\nGoals (L3)")
    ]
    for x, text in labels:
        ax.text(x, 0.02, text, ha='center', va='bottom', fontsize=10, color='gray', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
