import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import textwrap
import numpy as np
import math

def visualize_paper_graph(proposal_data, output_path):
    """
    Visualizes a multi-paper research synthesis using a radial/star layout.
    Center: Topic
    Inner Ring: Sub-concepts
    Outer Sectors: Papers and their nodes
    """
    
    # --- 1. Setup Data & Graph ---
    G = nx.Graph()
    topic_name = proposal_data.get('title', 'Topic')
    
    # Add Topic Node
    G.add_node(topic_name, type='topic', group='center')
    
    # Add Sub-concepts
    sub_concepts = proposal_data.get('sub_concepts', [])
    for concept in sub_concepts:
        G.add_node(concept, type='sub_concept', group='center')
        # Ensure edge to Topic
        G.add_edge(concept, topic_name)

    # Add Paper Nodes
    # proposal_data['paper_nodes'] expected format: { "paper_id": [ {name, level}, ... ], ... }
    # Or list of objects with paper_id?
    # Let's assume the workflow will produce a structure where we know which paper a node belongs to.
    # Current design in workflow planning: 
    # "paper_nodes": { "paper_id_1": [{"name": "...", "level": "..."}], ... }
    
    paper_nodes_map = proposal_data.get('paper_nodes', {})
    paper_ids = list(paper_nodes_map.keys())
    
    # Add Nodes
    for pid, nodes in paper_nodes_map.items():
        for node in nodes:
            n_name = node['name']
            n_level = node.get('level', 'Level1')
            G.add_node(n_name, type='paper_node', group=pid, level=n_level)
            
    # Add Edges
    for conn in proposal_data.get('connections', []):
        src = conn['source']
        tgt = conn['target']
        # Only add if nodes exist (some might be filtered out)
        if (G.has_node(src) or src == topic_name) and (G.has_node(tgt) or tgt == topic_name):
             G.add_node(src) # Ensure existence
             G.add_node(tgt)
             G.add_edge(src, tgt)

    # --- 2. Calculate Layout (Radial) ---
    pos = {}
    width, height = 16, 16 # Square canvas for radial
    center = (0.5, 0.5)
    
    # A. Topic Center
    pos[topic_name] = center
    
    # B. Sub-concepts (Inner Ring)
    n_sub = len(sub_concepts)
    if n_sub > 0:
        radius_sub = 0.15
        angle_step = 2 * np.pi / n_sub
        # Rotate slightly to not align perfectly with axis if needed
        start_angle = np.pi / 2 
        
        for i, node in enumerate(sub_concepts):
            angle = start_angle + i * angle_step
            x = center[0] + radius_sub * np.cos(angle)
            y = center[1] + radius_sub * np.sin(angle)
            pos[node] = (x, y)
            
    # C. Papers (Outer Sectors)
    n_papers = len(paper_ids)
    if n_papers > 0:
        # Assign angle sector to each paper
        sector_step = 2 * np.pi / n_papers
        sector_centers = {}
        for i, pid in enumerate(paper_ids):
            # Angle for this paper's center
            theta = np.pi / 2 + i * sector_step # Start from top
            sector_centers[pid] = theta
            
        # Place Paper Nodes within their sectors
        for pid, nodes in paper_nodes_map.items():
            if not nodes: continue
            
            theta_c = sector_centers[pid]
            # Spread nodes within a wedge around theta_c
            # Wedge width depends on n_papers. E.g. +/- sector_step/2 * 0.8 (margin)
            wedge_width = (sector_step / 2) * 0.7
            
            # Distribute nodes by Level?
            # Level 3 (Goals) -> Outer most? Or Inner most?
            # Usually L3 is high level, L1 is technique.
            # Let's put L1 closer to center (more specific/connected to sub-concepts?), L3 further out?
            # Or random distribution in the sector.
            
            # Simple approach: Random within sector bounds
            # Radius: 0.25 to 0.45
            
            # Sort nodes to have deterministic layout?
            sorted_nodes = sorted(nodes, key=lambda x: x['name'])
            n_p_nodes = len(sorted_nodes)
            
            for j, node_info in enumerate(sorted_nodes):
                node = node_info['name']
                # Determine radius based on Level?
                lvl = node_info.get('level', '')
                if 'Level3' in lvl: r_base = 0.42
                elif 'Level2' in lvl: r_base = 0.35
                elif 'Level1' in lvl: r_base = 0.28
                else: r_base = 0.35
                
                # Add some jitter to radius
                r = r_base + np.random.uniform(-0.02, 0.02)
                
                # Angle: distribute across the wedge
                # If 1 node, at center. If multiple, spread.
                if n_p_nodes > 1:
                    # Map j to -1 to 1 range
                    # normalized_pos = (j / (n_p_nodes - 1)) * 2 - 1 # -1 to 1
                    # But we want to mix them up so they aren't lines
                    # Use a pseudo-random offset based on name hash or index
                    offset_factor = ((j * 1.618) % 1) * 2 - 1 # -1 to 1
                else:
                    offset_factor = 0
                
                angle = theta_c + offset_factor * wedge_width
                
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                pos[node] = (x, y)

    # --- 3. Draw ---
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Background Sectors (Optional: Color code papers)
    # Define a palette
    paper_colors = ['#ffebee', '#e3f2fd', '#e8f5e9', '#fff3e0', '#f3e5f5', '#fff8e1', '#e0f7fa']
    paper_strokes = ['#e57373', '#64b5f6', '#81c784', '#ffb74d', '#ba68c8', '#ffd54f', '#4dd0e1']
    
    if n_papers > 0:
        for i, pid in enumerate(paper_ids):
            theta = sector_centers[pid]
            color = paper_colors[i % len(paper_colors)]
            # Draw a wedge or just a label
            
            # Label the sector with Paper ID
            label_r = 0.48
            lx = center[0] + label_r * np.cos(theta)
            ly = center[1] + label_r * np.sin(theta)
            
            ax.text(lx, ly, pid, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color=paper_strokes[i % len(paper_strokes)],
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=paper_strokes[i % len(paper_strokes)], alpha=0.9))

    # Edges
    for u, v in G.edges():
        if u not in pos or v not in pos: continue
        
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Style
        color = '#bdbdbd'
        width_edge = 1.0
        alpha = 0.6
        
        u_type = G.nodes[u].get('type') if u in G.nodes else ''
        v_type = G.nodes[v].get('type') if v in G.nodes else ''
        
        # Highlight Topic <-> Sub connections
        if 'topic' in [u_type, v_type]:
            color = '#7b1fa2'
            width_edge = 2.0
            alpha = 0.8
        
        # Highlight Sub <-> Paper connections
        elif 'sub_concept' in [u_type, v_type] and 'paper_node' in [u_type, v_type]:
            color = '#546e7a'
            width_edge = 1.5
            alpha = 0.7
            
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width_edge, alpha=alpha, zorder=1)

    # Nodes
    for node, (x, y) in pos.items():
        if node not in G.nodes: continue
        data = G.nodes[node]
        ntype = data.get('type')
        group = data.get('group')
        
        if ntype == 'topic':
            # Main Center
            wrapped = "\n".join(textwrap.wrap(node, width=15))
            ax.text(x, y, wrapped, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white',
                   bbox=dict(boxstyle="circle,pad=0.8", fc="#7b1fa2", ec="#4a148c", lw=2),
                   zorder=10)
        
        elif ntype == 'sub_concept':
            # Sub concepts
            wrapped = "\n".join(textwrap.wrap(node, width=10))
            ax.text(x, y, wrapped, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='#4a148c',
                   bbox=dict(boxstyle="round,pad=0.4", fc="#f3e5f5", ec="#8e24aa", lw=1.5),
                   zorder=9)
            
        elif ntype == 'paper_node':
            # Paper Nodes
            # Find color based on group (pid)
            try:
                idx = paper_ids.index(group)
                ec = paper_strokes[idx % len(paper_strokes)]
            except:
                ec = '#9e9e9e'
                
            fc = 'white'
            wrapped = "\n".join(textwrap.wrap(node, width=12))
            
            # Font size smaller
            ax.text(x, y, wrapped, ha='center', va='center',
                   fontsize=8, color='#424242',
                   bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec=ec, lw=1),
                   zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
