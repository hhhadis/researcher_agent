
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pandas as pd
from pyvis.network import Network
import matplotlib.colors as mcolors
import json

def parse_file(file_path):
    year = 9999
    title = "Unknown Title"
    paper_id = "Unknown ID"
    keywords = {'technical': [], 'ethical': []}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Year: "):
                try:
                    year = int(line.replace("Year: ", "").strip())
                except:
                    pass
            elif line.startswith("Title: "):
                title = line.replace("Title: ", "").strip()
            elif line.startswith("PaperId: "):
                paper_id = line.replace("PaperId: ", "").strip()
            elif line.startswith("Technical Keywords: "):
                kws = line.replace("Technical Keywords: ", "").strip()
                if kws:
                    keywords['technical'] = [k.strip() for k in kws.split(", ")]
            elif line.startswith("Ethical Keywords: "):
                kws = line.replace("Ethical Keywords: ", "").strip()
                if kws:
                    keywords['ethical'] = [k.strip() for k in kws.split(", ")]
            # Backward compatibility or fallback
            elif line.startswith("Keywords: "):
                kws = line.replace("Keywords: ", "").strip()
                if kws:
                     # Default to technical if unknown? Or ignore?
                     # Let's put in technical for now if mixed
                     keywords['technical'].extend([k.strip() for k in kws.split(", ")])
    return year, keywords, title, paper_id

def normalize_keywords(all_keywords_list):
    """
    Input: list of lists of keywords (raw strings)
    Output: mapping dict {raw_keyword: normalized_keyword}
    """
    # 1. Count lowercased forms
    lowered_counts = Counter()
    for kws in all_keywords_list:
        for kw in kws:
            lowered_counts[kw.lower()] += 1
            
    # 2. Build mapping
    mapping = {}
    
    # We need to traverse all unique raw keywords
    unique_raw = set()
    for kws in all_keywords_list:
        unique_raw.update(kws)
        
    for raw in unique_raw:
        lowered = raw.lower()
        
        # Check for plural: if 'lowered' ends with 's' and 'lowered' without 's' exists in corpus
        if lowered.endswith('s') and lowered[:-1] in lowered_counts:
            # Map to the singular form (lowered)
            mapping[raw] = lowered[:-1]
        else:
            mapping[raw] = lowered
            
    return mapping

def get_topic_root_label(topic_id, topic_info_df):
    if topic_info_df is None: return None
    try:
        tid = int(topic_id)
        row = topic_info_df[topic_info_df['Topic'] == tid]
        if not row.empty:
            # Check for CustomLabel first
            if 'CustomLabel' in row.columns:
                return row.iloc[0]['CustomLabel']
            # Fallback to Name
            name = row.iloc[0]['Name']
            parts = name.split('_')
            if len(parts) > 1:
                return parts[1] # Return first keyword
    except Exception as e:
        pass
    return None

def get_time_tree_pos(G):
    # Custom layout algorithm for Time Tree
    # Group nodes by Year
    layers = {}
    for n in G.nodes():
        y = G.nodes[n]['year']
        if y not in layers: layers[y] = []
        layers[y].append(n)
        
    sorted_years = sorted(layers.keys())
    
    # Assign Y (Top=Earliest)
    y_scale = 1000
    if len(sorted_years) > 1:
        min_y, max_y = sorted_years[0], sorted_years[-1]
        y_coords = {y: (1.0 - (y - min_y)/(max_y - min_y)) * y_scale for y in sorted_years}
    else:
        y_coords = {sorted_years[0]: 50}
        
    pos = {}
    width_per_node = 20 # Spacing between nodes
    
    # Process layers top-down
    for i, year in enumerate(sorted_years):
        nodes = layers[year]
        
        # Determine X based on parents
        node_stats = []
        for n in nodes:
            parents = list(G.predecessors(n))
            # Filter parents that have position assigned (from previous layers)
            valid_parents = [p for p in parents if p in pos]
            if valid_parents:
                avg_x = sum(pos[p][0] for p in valid_parents) / len(valid_parents)
            else:
                # If root or no parents yet
                avg_x = 0
            node_stats.append((n, avg_x))
            
        # Sort by preferred X
        node_stats.sort(key=lambda x: x[1])
        
        # Assign final X with separation
        current_xs = [x[1] for x in node_stats]
        
        # Apply spacing (Sweep Left->Right)
        for j in range(1, len(current_xs)):
            if current_xs[j] < current_xs[j-1] + width_per_node:
                current_xs[j] = current_xs[j-1] + width_per_node
                
        # Re-center to align with parents mean
        if current_xs:
            center_actual = (current_xs[0] + current_xs[-1]) / 2
            # Desired center is mean of preferred Xs
            center_desired = sum([x[1] for x in node_stats]) / len(node_stats) if node_stats else 0
            
            shift = center_desired - center_actual
            current_xs = [x + shift for x in current_xs]
            
        # Assign to pos
        for k, (n, _) in enumerate(node_stats):
            pos[n] = (current_xs[k], y_coords[year])
            
    return pos

def create_interactive_network(G, output_html_path, topic_name):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    # Configure options for Hierarchical Layout
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "nodeSpacing": 200,
                "levelSeparation": 200,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True
            }
        },
        "physics": {
            "enabled": False,
            "hierarchicalRepulsion": {
                "nodeDistance": 200
            }
        },
        "interaction": {
            "navigationButtons": True,
            "keyboard": True,
            "hover": True
        }
    }
    
    years_list = sorted(list(set([G.nodes[n]['year'] for n in G.nodes()])))
    year_to_level = {y: i for i, y in enumerate(years_list)}
    
    min_year = years_list[0] if years_list else 2020
    max_year = years_list[-1] if years_list else 2020
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_year, vmax=max_year)
    
    for n in G.nodes():
        year = G.nodes[n]['year']
        count = G.nodes[n].get('count', 1)
        core = G.nodes[n].get('core_number', 0)
        papers_info = G.nodes[n].get('papers', 'No papers')
        
        # Explicitly set level based on year to ensure visual hierarchy matches time/color
        level = year_to_level.get(year, 0)
        
        rgba = cmap(norm(year))
        color_hex = mcolors.to_hex(rgba)
        
        # Tooltip
        title_html = (
            f"<b>{n}</b><br>"
            f"Year: {year}<br>"
            f"Count: {count}<br>"
            f"Core: {core}<br>"
            f"<hr><b>Related Papers:</b><br>"
            f"{papers_info}"
        )
        
        # PyVis uses 'value' for size
        # Added 'level' to enforce year-based layering
        net.add_node(n, label=n, title=title_html, color=color_hex, value=count, group=year, level=level)
        
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 1)
        net.add_edge(u, v, value=weight, title=f"Weight: {weight}", arrowStrikethrough=False)
        
    # Apply options
    net.set_options(json.dumps(options))
    
    # Custom injection of CSS for tooltip
    # We need to save first, then append/modify the file
    try:
        net.save_graph(output_html_path)
        
        # Inject custom CSS for vis-tooltip
        with open(output_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        custom_css = """
        <style>
            div.vis-tooltip {
                max-width: 400px !important;
                white-space: normal !important;
                word-wrap: break-word !important;
                overflow-wrap: break-word !important;
                font-size: 14px;
            }
        </style>
        """
        
        if "</head>" in html_content:
            html_content = html_content.replace("</head>", f"{custom_css}</head>")
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        print(f"Interactive network saved to {output_html_path}")
    except Exception as e:
        print(f"Failed to save interactive network: {e}")

def build_concept_graph(topic_dir, keyword_type='technical', topic_info_df=None):
    """
    Builds the concept graph for a given topic and keyword type.
    Returns: G (NetworkX DiGraph), first_year (dict), node_papers (dict)
    """
    topic_name = os.path.basename(topic_dir)
    
    # 1. First Pass: Read all data to memory for normalization
    file_data = [] # List of (year, [keywords])
    all_keywords_lists = []
    
    for filename in os.listdir(topic_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(topic_dir, filename)
            year, keywords_dict, title, paper_id = parse_file(file_path)
            keywords = keywords_dict.get(keyword_type, [])
            if keywords:
                file_data.append((year, keywords, title, paper_id))
                all_keywords_lists.append(keywords)
                
    if not file_data:
        print(f"No {keyword_type} keywords found in {topic_dir}")
        return None, None, None

    # 2. Normalize keywords
    mapping = normalize_keywords(all_keywords_lists)

    # 2.5 Filter Keywords using k-shell (New Step)
    # Build a temporary full graph to calculate core numbers
    G_temp = nx.Graph()
    
    # We need word counts for tie-breaking
    word_counts = Counter()
    
    for year, keywords, _, _ in file_data:
        norm_keywords = sorted(list(set([mapping[k] for k in keywords])))
        
        # Update word counts
        for k in norm_keywords:
            word_counts[k] += 1
            
        # Add edges to temp graph
        for i in range(len(norm_keywords)):
            for j in range(i + 1, len(norm_keywords)):
                w1, w2 = norm_keywords[i], norm_keywords[j]
                if G_temp.has_edge(w1, w2):
                    G_temp[w1][w2]['weight'] += 1
                else:
                    G_temp.add_edge(w1, w2, weight=1)
    
    # Remove self-loops if any (though logic above prevents them)
    G_temp.remove_edges_from(nx.selfloop_edges(G_temp))
    
    # Calculate core numbers
    try:
        core_numbers = nx.core_number(G_temp)
    except Exception as e:
        # Fallback if graph is empty or other error
        print(f"k-shell calculation failed: {e}, falling back to frequency")
        core_numbers = {node: 0 for node in G_temp.nodes()}

    # Strategy: Sort by Core Number (desc), then Word Count (desc)
    MAX_NODES = 50
    
    # Create a list of (node, core_num, count)
    node_stats = []
    for node in G_temp.nodes():
        node_stats.append((node, core_numbers.get(node, 0), word_counts.get(node, 0)))
        
    # Sort: Primary key = Core Num (desc), Secondary key = Count (desc)
    node_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Keep top MAX_NODES
    valid_keywords = set([x[0] for x in node_stats[:MAX_NODES]])
    
    # 3. Build Stats
    first_year = {}
    cooccurrence = defaultdict(lambda: defaultdict(int))
    node_papers = defaultdict(list) # Stores list of formatted strings for display
    node_paper_ids = defaultdict(set) # Stores set of paper IDs for logic
    
    for year, keywords, title, paper_id in file_data:
        # Apply mapping and deduplicate per document
        norm_keywords = sorted(list(set([mapping[k] for k in keywords])))
        
        # Filter: Only keep valid keywords
        norm_keywords = [k for k in norm_keywords if k in valid_keywords]
        
        # Update first year and papers
        for k in norm_keywords:
            if k not in first_year or year < first_year[k]:
                first_year[k] = year
            # Add paper info
            node_papers[k].append(f"[{year}] {title}")
            node_paper_ids[k].add(paper_id)
                
        # Update co-occurrence
        for i in range(len(norm_keywords)):
            for j in range(i + 1, len(norm_keywords)):
                w1, w2 = norm_keywords[i], norm_keywords[j]
                cooccurrence[w1][w2] += 1
                cooccurrence[w2][w1] += 1
                
    # 4. Build Tree (Forest first)
    G = nx.DiGraph()
    sorted_keywords = sorted(first_year.keys(), key=lambda k: first_year[k])
    
    for kw in sorted_keywords:
        # Add node with papers info
        # Deduplicate papers list just in case
        unique_papers = sorted(list(set(node_papers[kw])))
        # Limit paper list size in tooltip to avoid huge popups
        papers_html = "<br>".join(unique_papers[:10])
        if len(unique_papers) > 10:
            papers_html += f"<br>...and {len(unique_papers)-10} more"
            
        G.add_node(kw, year=first_year[kw], papers=papers_html, paper_count=len(unique_papers), count=word_counts.get(kw, 0), core_number=core_numbers.get(kw, 0))
        
        candidates = []
        for neighbor, weight in cooccurrence[kw].items():
            if neighbor in first_year and first_year[neighbor] < first_year[kw]:
                candidates.append((neighbor, weight))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            parent, weight = candidates[0]
            G.add_edge(parent, kw, weight=weight)
            
    # 5. Enforce Single Tree (Find Best Real Root)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    if not roots and G.number_of_nodes() > 0:
        pass

    # Extract Topic ID from directory name (e.g., topic_0 -> 0)
    try:
        topic_id = int(os.path.basename(topic_dir).split('_')[-1])
    except:
        topic_id = -999
        
    root_label = get_topic_root_label(topic_id, topic_info_df)
    
    # If no root found or explicit label available, add super-root
    if (len(roots) > 1) or (root_label):
        final_root = root_label if root_label else "Topic Root"
        
        # Add root node
        if final_root not in G:
            min_year = min(first_year.values()) if first_year else 2020
            G.add_node(final_root, year=min_year, count=100, core_number=100) # Dummy high stats
            
        for r in roots:
            if r != final_root:
                G.add_edge(final_root, r, weight=1)
                
    return G, first_year, node_papers, node_paper_ids

def create_cooccurrence_network(topic_dir, output_image_path, topic_info_df=None, output_html_path=None, keyword_type='technical'):
    """
    Creates a concept tree where each concept connects to the most strongly co-occurring
    concept that appeared strictly earlier.
    Merged semantic duplicates and ensures a single tree structure.
    """
    topic_name = os.path.basename(topic_dir)
    
    G, first_year, node_papers, node_paper_ids = build_concept_graph(topic_dir, keyword_type, topic_info_df)
    
    if G is None or G.number_of_nodes() == 0:
        return

    # 6. Visualize
    plt.figure(figsize=(40, 40))
    
    pos = None
    # Try Graphviz first (best for trees)
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        # prog='dot' is for directed acyclic graphs (hierarchical)
        pos = graphviz_layout(G, prog='dot')
    except Exception as e:
        print(f"Graphviz layout failed ({e}), using fallback.")
        pos = None
        
    if pos is None:
        # Fallback: Custom Time Tree Layout (Python implementation)
        try:
            pos = get_time_tree_pos(G)
        except Exception as e:
            print(f"Custom layout failed ({e}), using spring.")
            pos = nx.spring_layout(G, k=2.0)
    
    # Get years for color mapping
    node_years = []
    for n in G.nodes():
        node_years.append(first_year.get(n, 2020))
            
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_years, cmap=plt.cm.viridis, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, arrowstyle='->', arrowsize=10, edge_color='gray', alpha=0.6)
    # Draw labels with larger font
    labels = {n: n.replace('$', '').replace('\\', '') for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif', font_weight='bold')
    
    plt.title(f"Concept Tree for {topic_name}", size=30)
    plt.axis('off')
    plt.savefig(output_image_path, format="PNG", dpi=300)
    plt.close()
    print(f"Concept Tree saved to {output_image_path}")
    
    if output_html_path:
        create_interactive_network(G, output_html_path, topic_name)

import glob
import pandas as pd

def process_all_topics(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create html output dir
    html_dir = os.path.join(base_dir, 'network_htmls')
    os.makedirs(html_dir, exist_ok=True)
    
    # Load topic_info
    topic_info_path = os.path.join(base_dir, "topic_info.csv")
    topic_info_df = None
    if os.path.exists(topic_info_path):
        try:
            topic_info_df = pd.read_csv(topic_info_path)
        except Exception as e:
            print(f"Error reading topic_info.csv: {e}")
    
    # Find all topic directories
    topic_dirs = glob.glob(os.path.join(base_dir, "topic_*"))
    
    print(f"Found {len(topic_dirs)} topic directories.")
    
    generated_htmls = []
    
    for topic_dir in topic_dirs:
        if not os.path.isdir(topic_dir):
            continue
            
        topic_name = os.path.basename(topic_dir)
        
        # Iterate over both keyword types
        for k_type in ['technical', 'ethical']:
            print(f"Processing {topic_name} - {k_type}...")
            
            output_image_path = os.path.join(output_dir, f"mst_core_network_{topic_name}_{k_type}.png")
            output_html_path = os.path.join(html_dir, f"network_{topic_name}_{k_type}.html")
            
            try:
                create_cooccurrence_network(topic_dir, output_image_path, topic_info_df, output_html_path, keyword_type=k_type)
                
                # Add to index list
                if os.path.exists(output_html_path):
                    label = topic_name
                    if topic_info_df is not None:
                        try:
                            # Extract ID from topic_N
                            parts = topic_name.split('_')
                            if len(parts) > 1 and parts[1].isdigit():
                                tid = int(parts[1])
                                row = topic_info_df[topic_info_df['Topic'] == tid]
                                if not row.empty and 'CustomLabel' in row.columns:
                                    label = f"{topic_name}: {row.iloc[0]['CustomLabel']}"
                        except Exception as e:
                            pass
                    
                    # Append type to label
                    label = f"{label} ({k_type.capitalize()})"
                    generated_htmls.append((topic_name, label, f"network_{topic_name}_{k_type}.html"))
                    
            except Exception as e:
                print(f"Failed to process {topic_name} ({k_type}): {e}")

    # Generate index.html
    try:
        index_path = os.path.join(html_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("<html><head><title>Research Topics Network Index</title>")
            f.write("<style>body{font-family:sans-serif; padding:20px;} li{margin:5px 0;} a{text-decoration:none; color:#0366d6;} a:hover{text-decoration:underline;}</style>")
            f.write("</head><body>")
            f.write("<h1>Research Topics Interactive Networks</h1>")
            f.write("<ul>")
            
            # Sort by topic ID
            def sort_key(item):
                try:
                    return int(item[0].split('_')[1])
                except:
                    return 0
            generated_htmls.sort(key=sort_key)
            
            for name, label, link in generated_htmls:
                f.write(f'<li><a href="{link}" target="_blank">{label}</a></li>')
            f.write("</ul></body></html>")
        print(f"Index generated at {index_path}")
    except Exception as e:
        print(f"Failed to generate index.html: {e}")

if __name__ == '__main__':
    base_dir = r'd:/workspace/ResearchAgent/core_concept_tree_2'
    output_dir = os.path.join(base_dir, 'network_images')
    process_all_topics(base_dir, output_dir)
