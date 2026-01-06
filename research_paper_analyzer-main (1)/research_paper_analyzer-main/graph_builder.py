"""
Graph Builder Module
Constructs interactive network graphs from hierarchical logic graph data
"""

import networkx as nx
from pyvis.network import Network
from typing import Dict, List, Tuple
import json
import math


class GraphBuilder:
    """Builds interactive graphs from hierarchical logic graph data."""
    
    def __init__(self, layout: str = "linear"):
        """
        Initialize the graph builder.
        
        Args:
            layout: Layout type - "linear" (horizontal lines) or "circular" (concentric circles)
        """
        self.G = nx.DiGraph()
        self.layout = layout  # "linear" or "circular"
        # Level colors for single-paper mode (fill colors)
        self.level_colors = {
            "Level3": "#4A90E2",  # Blue - research field problems & challenges
            "Level2": "#9B59B6",  # Purple - mathematical/conceptual frameworks
            "Level1": "#9B9B9B"  # Gray - concrete technical implementations
        }
        # Level border colors for multi-paper mode (distinct, noticeable borders)
        self.level_border_colors = {
            "Level3": "#FFFFFF",  # White - distinct border for problems
            "Level2": "#808080",  # Gray - distinct border for frameworks
            "Level1": "#000000"   # Black - distinct border for solutions
        }
        # Border width is consistent for all levels in multi-paper mode
        self.multi_paper_border_width = 4  # Consistent border thickness
        
        # Edge styling based on relation type
        self.edge_styles = {
            # Problem-to-Formulation (L3 → L2) - Blue tones
            "formulated-as": {"color": "#3498DB", "width": 3, "dashes": False},
            "reduced-to": {"color": "#5DADE2", "width": 2.5, "dashes": False},
            "modeled-as": {"color": "#85C1E9", "width": 2.5, "dashes": False},
            
            # Formulation-to-Solution (L2 → L1) - Green tones
            "solved-by": {"color": "#27AE60", "width": 3, "dashes": False},
            "implemented-via": {"color": "#52BE80", "width": 2.5, "dashes": False},
            "optimized-by": {"color": "#7DCEA0", "width": 2.5, "dashes": False},
            
            # Solution-to-Formulation (L1 → L2) - Gray-Green tones
            "implements": {"color": "#95A5A6", "width": 2, "dashes": True},
            "approximates": {"color": "#A9DFBF", "width": 2, "dashes": True},
            
            # Within-Level - Various
            "causes": {"color": "#E74C3C", "width": 2, "dashes": False},
            "related-to": {"color": "#BDC3C7", "width": 1.5, "dashes": True},
            "improves": {"color": "#16A085", "width": 2, "dashes": False},
            "extends": {"color": "#1ABC9C", "width": 2, "dashes": True},
            "conflicts": {"color": "#C0392B", "width": 2, "dashes": True},
            "is-part-of": {"color": "#7F8C8D", "width": 2, "dashes": True},
            "contributes-to": {"color": "#E67E22", "width": 2, "dashes": False},
            
            # Dependencies - Purple tones
            "requires": {"color": "#8E44AD", "width": 2, "dashes": False},
            "enables": {"color": "#9B59B6", "width": 1.5, "dashes": True},
            
            # Validation - Gold/Yellow tones
            "validates": {"color": "#F39C12", "width": 2.5, "dashes": False},
            "supports": {"color": "#F1C40F", "width": 1.5, "dashes": False},
            
            # Legacy relation types (for backward compatibility)
            "motivates": {"color": "#E67E22", "width": 2, "dashes": False},
            "mitigates": {"color": "#27AE60", "width": 3, "dashes": False},
        }
        
        # Edge color schemes for different relation types
        self.intra_paper_same_level_color = "#2ECC71"  # Green - same level connections
        self.intra_paper_cross_level_color = "#3498DB"  # Blue - cross level connections
        self.cross_paper_color = "#FF00FF"  # Magenta - cross paper connections
    
    def _get_paper_number(self, paper_id: str) -> str:
        """
        Extract paper number from paper_id (e.g., "paper_1" -> "1").
        
        Args:
            paper_id: Paper identifier like "paper_1", "paper_2", etc.
            
        Returns:
            Paper number as string (e.g., "1", "2")
        """
        if not paper_id:
            return ""
        # Extract number from "paper_1" -> "1"
        parts = paper_id.split("_")
        if len(parts) > 1:
            return parts[-1]
        return paper_id
    
    def _wrap_text(self, text: str, max_length: int = 80) -> str:
        """
        Wrap long text at word boundaries to prevent tooltip cutoff.
        
        Args:
            text: Text to wrap
            max_length: Maximum characters per line
            
        Returns:
            Text with line breaks at word boundaries
        """
        if not text or len(text) <= max_length:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            # Calculate space needed: word length + 1 space (if not first word)
            space_needed = word_length + (1 if current_line else 0)
            
            # If adding this word would exceed max_length, start a new line
            if current_length + space_needed > max_length and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += space_needed
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _get_node_level_number(self, node_id: str) -> int:
        """
        Get numeric level (1, 2, or 3) for a node.
        
        Args:
            node_id: Node identifier (may include paper prefix)
            
        Returns:
            Integer level (1, 2, or 3), or 0 if unknown
        """
        if not self.G.has_node(node_id):
            return 0
        
        node_data = self.G.nodes[node_id]
        level = node_data.get('level', '')
        
        if level == 'Level3':
            return 3
        elif level == 'Level2':
            return 2
        elif level == 'Level1':
            return 1
        else:
            return 0
    
    def _determine_edge_color(self, source: str, target: str, is_cross_paper: bool = False, 
                             is_inferred: bool = False, rel_type: str = "") -> dict:
        """
        Determine edge color and styling based on relation type.
        
        Args:
            source: Source node ID
            target: Target node ID
            is_cross_paper: Whether this is a cross-paper relation
            is_inferred: Whether this is an inferred relation
            rel_type: Relation type string
            
        Returns:
            Dictionary with color, width, and dashes styling
        """
        # Cross-paper relations always get magenta
        if is_cross_paper:
            return {
                "color": self.cross_paper_color,
                "width": 4,
                "dashes": [10, 5]
            }
        
        # For intra-paper relations, check if same-level or cross-level
        source_level = self._get_node_level_number(source)
        target_level = self._get_node_level_number(target)
        
        if source_level == target_level and source_level > 0:
            # Same level - Silver (for multi-paper mode)
            color = "#C0C0C0"  # Silver
            width = 2
        elif source_level > 0 and target_level > 0:
            # Cross level - Gold (for multi-paper mode)
            color = "#FFD700"  # Gold
            width = 2.5
        else:
            # Unknown - Gray (fallback)
            color = "#666666"
            width = 2
        
        # Inferred relations use dashed lines
        dashes = True if is_inferred else False
        
        return {
            "color": color,
            "width": width,
            "dashes": dashes
        }
    
    def _parse_evaluation_issues(self, eval_data: Dict, hlg: Dict = None) -> Dict:
        """
        Parse evaluation data to extract issues mapped to specific nodes and relations.
        
        Args:
            eval_data: Evaluation result dictionary
            hlg: Optional HLG data to match node names against
            
        Returns:
            Dictionary with:
                - node_issues: Dict mapping node names to list of issues
                - relation_issues: Dict mapping (source, target) tuples to list of issues
                - missing_concepts: List of missing concepts mentioned
        """
        node_issues = {}
        relation_issues = {}
        missing_concepts = []
        
        if not eval_data:
            return {"node_issues": node_issues, "relation_issues": relation_issues, "missing_concepts": missing_concepts}
        
        # Get all node names from HLG for matching
        all_nodes = set()
        if hlg:
            all_nodes.update(hlg.get("Level3", []))
            all_nodes.update(hlg.get("Level2", []))
            all_nodes.update(hlg.get("Level1", []))
            for inferred in hlg.get("InferredNodes", []):
                all_nodes.add(inferred.get("node", ""))
        
        # Get all relations for matching
        all_relations = []
        if hlg:
            for rel in hlg.get("Relations", []):
                all_relations.append((rel.get("source", ""), rel.get("target", "")))
            for rel in hlg.get("InferredRelations", []):
                all_relations.append((rel.get("source", ""), rel.get("target", "")))
        
        import re
        
        # Parse correctness issues
        correctness = eval_data.get("correctness", {})
        for issue in correctness.get("issues", []):
            issue_type = issue.get("type", "")
            description = issue.get("description", "")
            
            if issue_type == "missing_concept":
                missing_concepts.append(description)
            elif issue_type == "wrong_level":
                # Try to match node names from description against actual nodes
                for node in all_nodes:
                    if node.lower() in description.lower() or description.lower() in node.lower():
                        if node not in node_issues:
                            node_issues[node] = []
                        node_issues[node].append(f"⚠️ Wrong Level: {description}")
                        break
                # Also try regex as fallback
                if not any(node in node_issues for node in all_nodes):
                    match = re.search(r'"([^"]+)"', description)
                    if match:
                        potential_name = match.group(1)
                        for node in all_nodes:
                            if potential_name.lower() in node.lower() or node.lower() in potential_name.lower():
                                if node not in node_issues:
                                    node_issues[node] = []
                                node_issues[node].append(f"⚠️ Wrong Level: {description}")
                                break
            elif issue_type == "hallucination":
                # Try to match nodes
                for node in all_nodes:
                    if node.lower() in description.lower():
                        if node not in node_issues:
                            node_issues[node] = []
                        node_issues[node].append(f"⚠️ Hallucination: {description}")
                # Try to match relations
                for source, target in all_relations:
                    if source.lower() in description.lower() and target.lower() in description.lower():
                        key = (source, target)
                        if key not in relation_issues:
                            relation_issues[key] = []
                        relation_issues[key].append(f"⚠️ Hallucination: {description}")
            elif issue_type == "inaccurate_relation":
                # Try to match relations
                for source, target in all_relations:
                    if source.lower() in description.lower() and target.lower() in description.lower():
                        key = (source, target)
                        if key not in relation_issues:
                            relation_issues[key] = []
                        relation_issues[key].append(f"⚠️ Inaccurate: {description}")
        
        # Parse conciseness issues
        conciseness = eval_data.get("conciseness", {})
        for issue in conciseness.get("issues", []):
            issue_type = issue.get("type", "")
            description = issue.get("description", "")
            
            if issue_type == "redundancy":
                # Try to match redundant nodes
                for node in all_nodes:
                    if node.lower() in description.lower():
                        if node not in node_issues:
                            node_issues[node] = []
                        node_issues[node].append(f"⚠️ Redundant: {description}")
        
        return {"node_issues": node_issues, "relation_issues": relation_issues, "missing_concepts": missing_concepts}
    
    def build_graph(self, hlg: Dict, eval_data: Dict = None) -> nx.DiGraph:
        """
        Build a NetworkX graph from HLG data (including inferred nodes if present).
        
        Args:
            hlg: Hierarchical logic graph dictionary
            eval_data: Optional evaluation data to incorporate feedback into graph
            
        Returns:
            NetworkX directed graph
        """
        self.G = nx.DiGraph()
        
        # Parse evaluation issues if provided
        eval_issues = self._parse_evaluation_issues(eval_data, hlg) if eval_data else {"node_issues": {}, "relation_issues": {}, "missing_concepts": []}
        node_issues = eval_issues["node_issues"]
        relation_issues = eval_issues["relation_issues"]
        
        # Add Level 3 nodes (research field problems & challenges) - PAPER-BASED
        for concept in hlg.get("Level3", []):
            # Check for evaluation issues
            issues = node_issues.get(concept, [])
            issue_text = "\n".join(issues) if issues else ""
            title_base = f"Level 3 - Problem (Paper):\n{concept}"
            title_text = f"{title_base}\n{issue_text}" if issue_text else title_base
            
            # Add warning border if there are issues
            border_color = "#FF6B6B" if issues else None
            border_width = 3 if issues else 2
            
            self.G.add_node(
                concept,
                level="Level3",
                color=self.level_colors["Level3"],
                title=title_text,
                size=30,
                is_inferred=False,
                borderWidth=border_width,
                borderColor=border_color
            )
        
        # Add Level 2 nodes (mathematical/conceptual frameworks) - PAPER-BASED
        for concept in hlg.get("Level2", []):
            # Check for evaluation issues
            issues = node_issues.get(concept, [])
            issue_text = "\n".join(issues) if issues else ""
            title_base = f"Level 2 - Framework (Paper): {concept}"
            title_text = f"{title_base}\n{issue_text}" if issue_text else title_base
            
            # Add warning border if there are issues
            border_color = "#FF6B6B" if issues else None
            border_width = 3 if issues else 2
            
            self.G.add_node(
                concept,
                level="Level2",
                color=self.level_colors["Level2"],
                title=title_text,
                size=25,
                is_inferred=False,
                borderWidth=border_width,
                borderColor=border_color
            )
        
        # Add Level 1 nodes (concrete technical implementations) - PAPER-BASED
        for technique in hlg.get("Level1", []):
            # Check for evaluation issues
            issues = node_issues.get(technique, [])
            issue_text = "\n".join(issues) if issues else ""
            title_base = f"Level 1 - Technical Solution (Paper): {technique}"
            title_text = f"{title_base}\n{issue_text}" if issue_text else title_base
            
            # Add warning border if there are issues
            border_color = "#FF6B6B" if issues else None
            border_width = 3 if issues else 2
            
            self.G.add_node(
                technique,
                level="Level1",
                color=self.level_colors["Level1"],
                title=title_text,
                size=20,
                is_inferred=False,
                borderWidth=border_width,
                borderColor=border_color
            )
        
        # Add PAPER-BASED relations
        for relation in hlg.get("Relations", []):
            source = relation.get("source")
            target = relation.get("target")
            rel_type = relation.get("relation", "relates-to")
            confidence = relation.get("confidence", "N/A")
            explanation = relation.get("explanation", "")
            
            if source and target:
                # Check for evaluation issues
                issues = relation_issues.get((source, target), [])
                issue_text = "\n".join(issues) if issues else ""
                
                # Determine edge styling based on same-level vs cross-level
                style = self._determine_edge_color(source, target, is_cross_paper=False, 
                                                   is_inferred=False, rel_type=rel_type)
                
                # Include confidence and evaluation issues in the edge title
                wrapped_explanation = self._wrap_text(explanation)
                title_base = f"[PAPER] {source}({rel_type}){target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                title_text = f"{title_base}\n{issue_text}" if issue_text else title_base
                
                # Use warning color for edges with issues
                edge_color = "#FF6B6B" if issues else style["color"]
                edge_width = style["width"] + 1 if issues else style["width"]
                
                self.G.add_edge(
                    source,
                    target,
                    label=rel_type,
                    title=title_text,
                    arrows="to",
                    color=edge_color,
                    width=edge_width,
                    dashes=style["dashes"],
                    is_inferred=False
                )
        
        # Add INFERRED nodes (if present)
        for inferred_node in hlg.get("InferredNodes", []):
            node_name = inferred_node.get("node")
            node_level = inferred_node.get("level")
            confidence = inferred_node.get("confidence", "N/A")
            explanation = inferred_node.get("explanation", "")
            
            # Map level to color and attributes
            if node_level == "Level3":
                color = self.level_colors["Level3"]
                size = 30
                level_key = "Level3"
            elif node_level == "Level2":
                color = self.level_colors["Level2"]
                size = 25
                level_key = "Level2"
            else:  # Level1
                color = self.level_colors["Level1"]
                size = 20
                level_key = "Level1"
            
            # Make inferred nodes slightly transparent/lighter
            wrapped_explanation = self._wrap_text(explanation)
            title_text = f"[INFERRED] {node_level}: {node_name}\nConfidence: {confidence}/10\n{wrapped_explanation}"
            
            self.G.add_node(
                node_name,
                level=level_key,
                color=color,
                title=title_text,
                size=size,
                is_inferred=True,
                borderWidth=2,
                shapeProperties={"borderDashes": [5, 5]}  # Dashed border for inferred nodes
            )
        
        # Add INFERRED relations (if present)
        for inferred_relation in hlg.get("InferredRelations", []):
            source = inferred_relation.get("source")
            target = inferred_relation.get("target")
            rel_type = inferred_relation.get("relation", "relates-to")
            confidence = inferred_relation.get("confidence", "N/A")
            explanation = inferred_relation.get("explanation", "")
            
            if source and target:
                # Determine edge styling based on same-level vs cross-level (inferred)
                style = self._determine_edge_color(source, target, is_cross_paper=False, 
                                                   is_inferred=True, rel_type=rel_type)
                
                # Inferred relations always have dashed lines
                wrapped_explanation = self._wrap_text(explanation)
                title_text = f"[INFERRED] [{rel_type}]: {source} → {target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                
                self.G.add_edge(
                    source,
                    target,
                    label=rel_type,
                    title=title_text,
                    arrows="to",
                    color=style["color"],
                    width=style["width"],
                    dashes=True,  # Always dashed for inferred relations
                    is_inferred=True
                )
        
        return self.G
    
    def build_multi_paper_graph(self, multi_paper_data: Dict) -> nx.DiGraph:
        """
        Build a NetworkX graph from multi-paper analysis data.
        
        Args:
            multi_paper_data: Dictionary containing:
                - papers: List of paper dicts with hlg_data and color info
                - cross_paper_relations: List of cross-paper relation dicts
                
        Returns:
            NetworkX directed graph with all papers and cross-paper relations
        """
        self.G = nx.DiGraph()
        
        papers = multi_paper_data.get('papers', [])
        cross_paper_relations = multi_paper_data.get('cross_paper_relations', [])
        
        # Add nodes from each paper with paper-specific colors
        for paper in papers:
            paper_id = paper['id']
            paper_name = paper['name']
            paper_number = self._get_paper_number(paper_id)
            hlg = paper['hlg_data']
            colors = paper['color']
            
            # Level 3 nodes
            for concept in hlg.get("Level3", []):
                node_id = f"{paper_id}::{concept}"
                self.G.add_node(
                    node_id,
                    level="Level3",
                    color=self.level_border_colors["Level3"],  # Fill color = level color
                    borderColor=colors['hex'],  # Border color = paper color
                    borderWidth=self.multi_paper_border_width,
                    title=f"Level 3 - Problem\n[Paper {paper_number}: {paper_name}]:\n{concept}",
                    size=30,
                    is_inferred=False,
                    paper_id=paper_id,
                    original_name=concept,
                    hasShadow=True  # Add shadow to Level 3 nodes for better visibility
                )
            
            # Level 2 nodes
            for concept in hlg.get("Level2", []):
                node_id = f"{paper_id}::{concept}"
                self.G.add_node(
                    node_id,
                    level="Level2",
                    color=self.level_border_colors["Level2"],  # Fill color = level color
                    borderColor=colors['hex'],  # Border color = paper color
                    borderWidth=self.multi_paper_border_width,
                    title=f"Level 2 - Framework [Paper {paper_number}: {paper_name}]:\n {concept}",
                    size=25,
                    is_inferred=False,
                    paper_id=paper_id,
                    original_name=concept
                )
            
            # Level 1 nodes
            for technique in hlg.get("Level1", []):
                node_id = f"{paper_id}::{technique}"
                self.G.add_node(
                    node_id,
                    level="Level1",
                    color=self.level_border_colors["Level1"],  # Fill color = level color
                    borderColor=colors['hex'],  # Border color = paper color
                    borderWidth=self.multi_paper_border_width,
                    title=f"Level 1 - Technical Solution\n[Paper {paper_number}: {paper_name}]:\n{technique}",
                    size=20,
                    is_inferred=False,
                    paper_id=paper_id,
                    original_name=technique
                )
            
            # Add intra-paper relations
            for relation in hlg.get("Relations", []):
                source = relation.get("source")
                target = relation.get("target")
                rel_type = relation.get("relation", "relates-to")
                confidence = relation.get("confidence", "N/A")
                explanation = relation.get("explanation", "")
                
                if source and target:
                    source_id = f"{paper_id}::{source}"
                    target_id = f"{paper_id}::{target}"
                    
                    # Determine edge styling based on same-level vs cross-level
                    style = self._determine_edge_color(source_id, target_id, is_cross_paper=False, 
                                                       is_inferred=False, rel_type=rel_type)
                    
                    wrapped_explanation = self._wrap_text(explanation)
                    title_text = f"[Paper {paper_number}: {paper_name}]\n {source} \n({rel_type}) \n{target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                    
                    self.G.add_edge(
                        source_id,
                        target_id,
                        label=rel_type,
                        title=title_text,
                        arrows="to",
                        color=style["color"],
                        width=style["width"],
                        dashes=style["dashes"],
                        is_cross_paper=False
                    )
            
            # Add inferred nodes (if present)
            for inferred_node in hlg.get("InferredNodes", []):
                node_name = inferred_node.get("node")
                node_level = inferred_node.get("level")
                confidence = inferred_node.get("confidence", "N/A")
                explanation = inferred_node.get("explanation", "")
                
                node_id = f"{paper_id}::{node_name}"
                
                # Map level to size and colors (switched: fill = level, border = paper)
                if node_level == "Level3":
                    fill_color = self.level_border_colors["Level3"]  # Fill color = level color
                    border_color = colors['hex']  # Border color = paper color
                    size = 30
                    level_key = "Level3"
                elif node_level == "Level2":
                    fill_color = self.level_border_colors["Level2"]  # Fill color = level color
                    border_color = colors['hex']  # Border color = paper color
                    size = 25
                    level_key = "Level2"
                else:
                    fill_color = self.level_border_colors["Level1"]  # Fill color = level color
                    border_color = colors['hex']  # Border color = paper color
                    size = 20
                    level_key = "Level1"
                
                paper_number = self._get_paper_number(paper_id)
                wrapped_explanation = self._wrap_text(explanation)
                title_text = f"[INFERRED- Paper {paper_number}: {paper_name}] {node_level}:\n{node_name}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                
                self.G.add_node(
                    node_id,
                    level=level_key,
                    color=fill_color,  # Fill color = paper color
                    borderColor=border_color,  # Border color = level color
                    borderWidth=self.multi_paper_border_width,
                    title=title_text,
                    size=size,
                    is_inferred=True,
                    shapeProperties={"borderDashes": [5, 5]},  # Dashed border for inferred
                    paper_id=paper_id,
                    original_name=node_name
                )
            
            # Add inferred intra-paper relations (if present)
            for inferred_relation in hlg.get("InferredRelations", []):
                source = inferred_relation.get("source")
                target = inferred_relation.get("target")
                rel_type = inferred_relation.get("relation", "relates-to")
                confidence = inferred_relation.get("confidence", "N/A")
                explanation = inferred_relation.get("explanation", "")
                
                if source and target:
                    source_id = f"{paper_id}::{source}"
                    target_id = f"{paper_id}::{target}"
                    
                    # Determine edge styling based on same-level vs cross-level (inferred)
                    style = self._determine_edge_color(source_id, target_id, is_cross_paper=False, 
                                                       is_inferred=True, rel_type=rel_type)
                    
                    paper_number = self._get_paper_number(paper_id)
                    wrapped_explanation = self._wrap_text(explanation)
                    title_text = f"[INFERRED - Paper {paper_number}: {paper_name}]\n [{rel_type}]: {source} → {target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                    
                    self.G.add_edge(
                        source_id,
                        target_id,
                        label=rel_type,
                        title=title_text,
                        arrows="to",
                        color=style["color"],
                        width=style["width"],
                        dashes=True,
                        is_cross_paper=False
                    )
        
        # Add cross-paper relations (distinct styling)
        for cross_rel in cross_paper_relations:
            source = cross_rel.get("source")
            source_paper = cross_rel.get("source_paper")
            target = cross_rel.get("target")
            target_paper = cross_rel.get("target_paper")
            rel_type = cross_rel.get("relation", "relates-to")
            confidence = cross_rel.get("confidence", "N/A")
            explanation = cross_rel.get("explanation", "")
            
            if source and target and source_paper and target_paper:
                source_id = f"{source_paper}::{source}"
                target_id = f"{target_paper}::{target}"
                
                # Get source and target paper names and numbers
                source_paper_data = next((p for p in papers if p['id'] == source_paper), None)
                target_paper_data = next((p for p in papers if p['id'] == target_paper), None)
                source_paper_name = source_paper_data['name'] if source_paper_data else source_paper
                target_paper_name = target_paper_data['name'] if target_paper_data else target_paper
                source_paper_number = self._get_paper_number(source_paper)
                target_paper_number = self._get_paper_number(target_paper)
                
                # Cross-paper edges have special styling (thick magenta dashed line)
                wrapped_explanation = self._wrap_text(explanation)
                title_text = f"[CROSS-PAPER] \n[Paper {source_paper_number}: {source_paper_name}] \n{source}\n({rel_type})\n[Paper {target_paper_number}: {target_paper_name}]\n {target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                
                self.G.add_edge(
                    source_id,
                    target_id,
                    label=rel_type,
                    title=title_text,
                    arrows="to",
                    color="#FF00FF",  # Magenta for cross-paper
                    width=4,  # Thicker line
                    dashes=[10, 5],  # Distinct dashed pattern
                    is_cross_paper=True
                )
        
        return self.G
    
    def build_cross_researcher_graph(self, cross_researcher_data: Dict) -> nx.DiGraph:
        """
        Build a NetworkX graph from cross-researcher analysis data.
        
        Args:
            cross_researcher_data: Dictionary containing:
                - researcher_1: {name, hlg_data, original_data}
                - researcher_2: {name, hlg_data, original_data}
                - cross_researcher_relations: List of cross-researcher relation dicts
                
        Returns:
            NetworkX directed graph with both researchers' HLGs and cross-researcher relations
        """
        self.G = nx.DiGraph()
        
        researcher_1 = cross_researcher_data.get("researcher_1", {})
        researcher_2 = cross_researcher_data.get("researcher_2", {})
        cross_researcher_relations = cross_researcher_data.get("cross_researcher_relations", [])
        
        r1_name = researcher_1.get("name", "R1")
        r2_name = researcher_2.get("name", "R2")
        r1_hlg = researcher_1.get("hlg_data", {})
        r2_hlg = researcher_2.get("hlg_data", {})
        
        # Add nodes from Researcher 1 (R1) - use orange/red tones (hue ~0.05, researcher range 0.0-0.45)
        # Using orange color from researcher palette to avoid conflicts with clusters (0.55-1.0)
        r1_colors = {"hex": "#FF6B35", "l3": "#FF6B35", "l2": "#FF8C42", "l1": "#FFB88C"}
        
        # R1 Level 3 nodes
        for concept in r1_hlg.get("Level3", []):
            node_id = f"R1::{concept}"
            self.G.add_node(
                node_id,
                level="Level3",
                color=self.level_border_colors["Level3"],
                borderColor=r1_colors["hex"],
                borderWidth=self.multi_paper_border_width,
                title=f"Level 3 - Problem\n[Paper {r1_name}: {r1_name}]:\n{concept}",
                size=30,
                is_inferred=False,
                paper_id="R1",
                original_name=concept,
                hasShadow=True,
                researcher="R1"
            )
        
        # R1 Level 2 nodes
        for concept in r1_hlg.get("Level2", []):
            node_id = f"R1::{concept}"
            self.G.add_node(
                node_id,
                level="Level2",
                color=self.level_border_colors["Level2"],
                borderColor=r1_colors["hex"],
                borderWidth=self.multi_paper_border_width,
                title=f"Level 2 - Framework [Paper {r1_name}: {r1_name}]:\n {concept}",
                size=25,
                is_inferred=False,
                paper_id="R1",
                original_name=concept,
                researcher="R1"
            )
        
        # R1 Level 1 nodes
        for technique in r1_hlg.get("Level1", []):
            node_id = f"R1::{technique}"
            self.G.add_node(
                node_id,
                level="Level1",
                color=self.level_border_colors["Level1"],
                borderColor=r1_colors["hex"],
                borderWidth=self.multi_paper_border_width,
                title=f"Level 1 - Technical Solution\n[Paper {r1_name}: {r1_name}]:\n{technique}",
                size=20,
                is_inferred=False,
                paper_id="R1",
                original_name=technique,
                researcher="R1"
            )
        
        # Add R1 intra-researcher relations
        for relation in r1_hlg.get("Relations", []):
            source = relation.get("source")
            target = relation.get("target")
            rel_type = relation.get("relation", "relates-to")
            confidence = relation.get("confidence", "N/A")
            explanation = relation.get("explanation", "")
            
            if source and target:
                source_id = f"R1::{source}"
                target_id = f"R1::{target}"
                
                style = self._determine_edge_color(source_id, target_id, is_cross_paper=False, 
                                                   is_inferred=False, rel_type=rel_type)
                
                wrapped_explanation = self._wrap_text(explanation)
                title_text = f"[Paper {r1_name}: {r1_name}]\n {source} \n({rel_type}) \n{target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                
                self.G.add_edge(
                    source_id,
                    target_id,
                    label=rel_type,
                    title=title_text,
                    arrows="to",
                    color=style["color"],
                    width=style["width"],
                    dashes=style["dashes"],
                    is_cross_paper=False,
                    researcher="R1"
                )
        
        # Add nodes from Researcher 2 (R2) - use red tones (hue ~0.0, researcher range 0.0-0.45)
        # Using red color from researcher palette to avoid conflicts with clusters (0.55-1.0)
        r2_colors = {"hex": "#E74C3C", "l3": "#E74C3C", "l2": "#EC7063", "l1": "#F1948A"}
        
        # R2 Level 3 nodes
        for concept in r2_hlg.get("Level3", []):
            node_id = f"R2::{concept}"
            self.G.add_node(
                node_id,
                level="Level3",
                color=self.level_border_colors["Level3"],
                borderColor=r2_colors["hex"],
                borderWidth=self.multi_paper_border_width,
                title=f"Level 3 - Problem\n[Paper {r2_name}: {r2_name}]:\n{concept}",
                size=30,
                is_inferred=False,
                paper_id="R2",
                original_name=concept,
                hasShadow=True,
                researcher="R2"
            )
        
        # R2 Level 2 nodes
        for concept in r2_hlg.get("Level2", []):
            node_id = f"R2::{concept}"
            self.G.add_node(
                node_id,
                level="Level2",
                color=self.level_border_colors["Level2"],
                borderColor=r2_colors["hex"],
                borderWidth=self.multi_paper_border_width,
                title=f"Level 2 - Framework [Paper {r2_name}: {r2_name}]:\n {concept}",
                size=25,
                is_inferred=False,
                paper_id="R2",
                original_name=concept,
                researcher="R2"
            )
        
        # R2 Level 1 nodes
        for technique in r2_hlg.get("Level1", []):
            node_id = f"R2::{technique}"
            self.G.add_node(
                node_id,
                level="Level1",
                color=self.level_border_colors["Level1"],
                borderColor=r2_colors["hex"],
                borderWidth=self.multi_paper_border_width,
                title=f"Level 1 - Technical Solution\n[Paper {r2_name}: {r2_name}]:\n{technique}",
                size=20,
                is_inferred=False,
                paper_id="R2",
                original_name=technique,
                researcher="R2"
            )
        
        # Add R2 intra-researcher relations
        for relation in r2_hlg.get("Relations", []):
            source = relation.get("source")
            target = relation.get("target")
            rel_type = relation.get("relation", "relates-to")
            confidence = relation.get("confidence", "N/A")
            explanation = relation.get("explanation", "")
            
            if source and target:
                source_id = f"R2::{source}"
                target_id = f"R2::{target}"
                
                style = self._determine_edge_color(source_id, target_id, is_cross_paper=False, 
                                                   is_inferred=False, rel_type=rel_type)
                
                wrapped_explanation = self._wrap_text(explanation)
                title_text = f"[Paper {r2_name}: {r2_name}]\n {source} \n({rel_type}) \n{target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                
                self.G.add_edge(
                    source_id,
                    target_id,
                    label=rel_type,
                    title=title_text,
                    arrows="to",
                    color=style["color"],
                    width=style["width"],
                    dashes=style["dashes"],
                    is_cross_paper=False,
                    researcher="R2"
                )
        
        # Add cross-researcher relations (distinct styling - thick cyan dashed line)
        for cross_rel in cross_researcher_relations:
            source = cross_rel.get("source")
            source_researcher = cross_rel.get("source_researcher", "R1")
            target = cross_rel.get("target")
            target_researcher = cross_rel.get("target_researcher", "R2")
            rel_type = cross_rel.get("relation", "relates-to")
            confidence = cross_rel.get("confidence", "N/A")
            explanation = cross_rel.get("explanation", "")
            
            if source and target and source_researcher and target_researcher:
                source_id = f"{source_researcher}::{source}"
                target_id = f"{target_researcher}::{target}"
                
                # Cross-researcher edges have special styling (thick cyan dashed line)
                wrapped_explanation = self._wrap_text(explanation)
                title_text = f"[CROSS-RESEARCHER] \n[Paper {source_researcher}: {source_researcher}] \n{source}\n({rel_type})\n[Paper {target_researcher}: {target_researcher}]\n {target}\nConfidence: {confidence}/10\n{wrapped_explanation}"
                
                self.G.add_edge(
                    source_id,
                    target_id,
                    label=rel_type,
                    title=title_text,
                    arrows="to",
                    color="#00FFFF",  # Cyan for cross-researcher
                    width=5,  # Thicker line
                    dashes=[10, 5],  # Distinct dashed pattern
                    is_cross_paper=True,
                    is_cross_researcher=True
                )
        
        return self.G
    
    def to_pyvis(self, height: str = "600px", width: str = "100%") -> Network:
        """
        Convert NetworkX graph to PyVis interactive visualization with hierarchical layout.
        
        Args:
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            PyVis Network object
        """
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="black"
        )
        
        # Configure layout (no bouncy physics, free dragging on both axes!)
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": false
            }
          },
          "physics": {
            "enabled": false
          },
          "nodes": {
            "font": {
              "size": 16,
              "face": "Arial"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shape": "box",
            "margin": 10,
            "mass": 1
          },
          "edges": {
            "font": {
              "size": 12,
              "align": "middle"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            },
            "smooth": {
              "enabled": true,
              "type": "cubicBezier",
              "roundness": 0.5
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "multiselect": true,
            "navigationButtons": true,
            "keyboard": {
              "enabled": true,
              "speed": {
                "x": 10,
                "y": 10,
                "zoom": 0.02
              }
            },
            "tooltipDelay": 100
          },
          "manipulation": {
            "enabled": false
          }
        }
        """)
        
        # Add nodes with attributes and explicit positioning
        # Count nodes per level for better spacing
        nodes_by_level = {"Level3": [], "Level2": [], "Level1": []}
        for node, attrs in self.G.nodes(data=True):
            level = attrs.get("level", "Level1")
            if level in nodes_by_level:
                nodes_by_level[level].append((node, attrs))
        
        # Check if this is multi-paper mode (nodes have paper_id attribute)
        is_multi_paper = any(attrs.get("paper_id") for _, attrs in self.G.nodes(data=True))
        
        # Pre-calculate positions for all nodes
        node_positions = {}
        
        if self.layout == "circular" and is_multi_paper:
            # Multi-paper circular layout: group nodes by paper, then by level within each paper
            # Group nodes by paper
            nodes_by_paper = {}
            for node, attrs in self.G.nodes(data=True):
                paper_id = attrs.get("paper_id")
                if paper_id:
                    if paper_id not in nodes_by_paper:
                        nodes_by_paper[paper_id] = {"Level3": [], "Level2": [], "Level1": []}
                    level = attrs.get("level", "Level1")
                    if level in nodes_by_paper[paper_id]:
                        nodes_by_paper[paper_id][level].append((node, attrs))
            
            # Count total nodes per level across all papers for adaptive radius
            total_nodes_per_level = {"Level3": 0, "Level2": 0, "Level1": 0}
            for paper_id, levels in nodes_by_paper.items():
                for level, node_list in levels.items():
                    total_nodes_per_level[level] += len(node_list)
            
            # Calculate adaptive radius based on node density per level
            # Formula: radius = base_radius * (1 + density_factor * log(num_nodes + 1))
            # density_factor controls how aggressively radius scales with node count
            density_factor = 0.3  # Adjust this to control scaling sensitivity
            
            base_radius_map = {
                "Level3": 1000,
                "Level2": 650,
                "Level1": 300
            }
            
            adaptive_radius_map = {}
            for level, base_radius in base_radius_map.items():
                num_nodes = total_nodes_per_level.get(level, 0)
                # Use natural logarithm for smooth scaling
                adaptive_radius = base_radius * (1 + density_factor * math.log(num_nodes + 1))
                adaptive_radius_map[level] = adaptive_radius
            
            # Assign angle ranges to each paper
            num_papers = len(nodes_by_paper)
            angle_range_per_paper = 360.0 / num_papers if num_papers > 0 else 360.0
            
            # Calculate positions for each node
            for node, attrs in self.G.nodes(data=True):
                paper_id = attrs.get("paper_id")
                level = attrs.get("level", "Level1")
                
                if paper_id and paper_id in nodes_by_paper:
                    # Get paper index and its angle range
                    paper_ids = sorted(nodes_by_paper.keys())
                    paper_index = paper_ids.index(paper_id)
                    paper_start_angle = paper_index * angle_range_per_paper
                    
                    # Get nodes in this paper and level
                    paper_level_nodes = nodes_by_paper[paper_id][level]
                    node_index = next(i for i, (n, _) in enumerate(paper_level_nodes) if n == node)
                    num_nodes_in_level = len(paper_level_nodes)
                    
                    # Calculate angle within paper's region
                    # Each paper occupies a distinct region, all levels share that region
                    # Distribute nodes evenly across the entire paper's angle range
                    if num_nodes_in_level > 1:
                        # Multiple nodes at this level: distribute evenly across paper's region
                        angle_step = angle_range_per_paper / num_nodes_in_level
                        # Start from beginning of paper's region, add small offset to center nodes
                        angle_offset = angle_range_per_paper * 0.05  # 5% offset from start
                        angle_within_paper = node_index * angle_step + angle_offset
                    else:
                        # Single node: place at center of paper's region
                        angle_within_paper = angle_range_per_paper * 0.5
                    
                    # Final angle: paper's start + position within paper's region
                    angle_deg = paper_start_angle + angle_within_paper
                    angle = angle_deg * (math.pi / 180)  # Convert to radians
                    
                    # Use adaptive radius based on total node count at this level
                    radius = adaptive_radius_map.get(level, 200)
                    
                    # Calculate x, y position on circle
                    node_positions[node] = {
                        "x": radius * math.cos(angle),
                        "y": radius * math.sin(angle)
                    }
                else:
                    # Fallback: Node doesn't have paper_id or isn't in nodes_by_paper
                    # Position it based on level with a default radius
                    fallback_radius_map = {
                        "Level3": 1000,
                        "Level2": 650,
                        "Level1": 300
                    }
                    fallback_radius = fallback_radius_map.get(level, 200)
                    # Place at a default angle (e.g., 0 degrees)
                    node_positions[node] = {
                        "x": fallback_radius * math.cos(0),
                        "y": fallback_radius * math.sin(0)
                    }
        
        for node, attrs in self.G.nodes(data=True):
            level = attrs.get("level", "Level1")
            
            if self.layout == "circular":
                if is_multi_paper and node in node_positions:
                    # Use pre-calculated position from paper-based grouping
                    x_pos = node_positions[node]["x"]
                    y_pos = node_positions[node]["y"]
                else:
                    # Single-paper circular layout: distribute by level with adaptive radius
                    # Calculate adaptive radius based on node density per level
                    # Formula: radius = base_radius * (1 + density_factor * log(num_nodes + 1))
                    density_factor = 0.3  # Adjust this to control scaling sensitivity
                    
                    base_radius_map = {
                        "Level3": 1000,
                        "Level2": 650,
                        "Level1": 300
                    }
                    
                    # Get nodes in this level for angular distribution
                    if level in nodes_by_level:
                        level_nodes = nodes_by_level[level]
                        node_index = next(i for i, (n, _) in enumerate(level_nodes) if n == node)
                        num_nodes = len(level_nodes)
                        
                        # Calculate adaptive radius based on node count at this level
                        base_radius = base_radius_map.get(level, 200)
                        # Use natural logarithm for smooth scaling
                        radius = base_radius * (1 + density_factor * math.log(num_nodes + 1))
                        
                        # Calculate angle: evenly distribute around circle
                        # Add offset per level to avoid alignment (Level3: 0°, Level2: 30°, Level1: 60°)
                        level_offset = {"Level3": 0, "Level2": 30, "Level1": 60}.get(level, 0)
                        angle_step = 360.0 / num_nodes if num_nodes > 0 else 0
                        angle = (node_index * angle_step + level_offset) * (math.pi / 180)  # Convert to radians
                        
                        # Calculate x, y position on circle
                        x_pos = radius * math.cos(angle)
                        y_pos = radius * math.sin(angle)
                    else:
                        # Fallback: Node's level not in nodes_by_level
                        # Position it based on level with a default radius and angle
                        base_radius = base_radius_map.get(level, 200)
                        radius = base_radius
                        # Place at a default angle (0 degrees) to avoid center
                        x_pos = radius * math.cos(0)
                        y_pos = radius * math.sin(0)
            else:
                # Linear layout: horizontal lines (original)
                level_y_map = {
                    "Level3": -400,  # Top level
                    "Level2": -100,  # Second level
                    "Level1": 200  # Bottom level
                }
                y_pos = level_y_map.get(level, 0)
                
                # Calculate x-coordinate based on position within level
                if level in nodes_by_level:
                    level_nodes = nodes_by_level[level]
                    node_index = next(i for i, (n, _) in enumerate(level_nodes) if n == node)
                    num_nodes = len(level_nodes)
                    
                    # Spread nodes horizontally
                    if num_nodes > 1:
                        spacing = 300
                        total_width = (num_nodes - 1) * spacing
                        x_pos = node_index * spacing - total_width / 2
                    else:
                        x_pos = 0
                else:
                    x_pos = 0
            
            # Check if this is an inferred node
            is_inferred = attrs.get("is_inferred", False)
            
            # Get display label (remove paper prefix for multi-paper mode)
            display_label = attrs.get("original_name", node)
            if display_label == node and "::" in node:
                # This is from multi-paper mode, extract the concept name
                display_label = node.split("::", 1)[1] if "::" in node else node
            
            # Apply dashed border for inferred nodes, or warning border for issues
            fill_color = attrs.get("color", "#97C2FC")
            border_color = attrs.get("borderColor")
            border_width = attrs.get("borderWidth", 2)
            
            # PyVis/vis.js requires color as an object with background and border
            if border_color:
                # Multi-paper mode: fill = paper color, border = level color
                node_color = {
                    "background": fill_color,
                    "border": border_color
                }
                # Use consistent border width for multi-paper mode, or specified width for single-paper
                actual_border_width = border_width if border_width != 2 else self.multi_paper_border_width
            else:
                # Single-paper mode: use fill color, default border
                node_color = fill_color
                actual_border_width = border_width
            
            node_params = {
                "label": display_label,
                "color": node_color,
                "title": attrs.get("title", node),
                "size": attrs.get("size", 20),
                "x": x_pos,
                "y": y_pos,
                "fixed": False,  # Allow free dragging in both x and y
                "borderWidth": actual_border_width
            }
            
            # Add shadow to Level 3 nodes for better separation from background
            if attrs.get("level") == "Level3":
                node_params["shadow"] = {
                    "enabled": True,
                    "color": "rgba(0,0,0,0.3)",
                    "size": 10,
                    "x": 3,
                    "y": 3
                }
            
            if is_inferred:
                node_params["shapeProperties"] = {"borderDashes": [5, 5]}
            
            net.add_node(node, **node_params)
        
        # Add edges with attributes (including color and width)
        for source, target, attrs in self.G.edges(data=True):
            net.add_edge(
                source,
                target,
                label=attrs.get("label", ""),
                title=attrs.get("title", ""),
                color=attrs.get("color", "#666666"),
                width=attrs.get("width", 2),
                dashes=attrs.get("dashes", False)
            )
        
        return net
    
    def save_html(self, filename: str = "graph.html"):
        """
        Save the graph as an interactive HTML file.
        
        Args:
            filename: Output filename
        """
        net = self.to_pyvis()
        net.save_graph(filename)
        return filename
    
    def to_json(self, filename: str = None) -> str:
        """
        Export graph to JSON format.
        
        Args:
            filename: Optional filename to save JSON
            
        Returns:
            JSON string representation
        """
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node, attrs in self.G.nodes(data=True):
            graph_data["nodes"].append({
                "id": node,
                "label": node,
                **attrs
            })
        
        for source, target, attrs in self.G.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                **attrs
            })
        
        json_str = json.dumps(graph_data, indent=2, ensure_ascii=False)
        
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_str)
        
        return json_str
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "level3_nodes": len([n for n, d in self.G.nodes(data=True) if d.get("level") == "Level3"]),
            "level2_nodes": len([n for n, d in self.G.nodes(data=True) if d.get("level") == "Level2"]),
            "level1_nodes": len([n for n, d in self.G.nodes(data=True) if d.get("level") == "Level1"]),
            "density": nx.density(self.G),
            "is_connected": nx.is_weakly_connected(self.G) if self.G.number_of_nodes() > 0 else False
        }


if __name__ == "__main__":
    # Test with example data showcasing new edge types
    example_hlg = {
        "Level3": ["Representation Learning", "Conditional Generation"],
        "Level2": ["Sparse Tag Distribution", "Unbalanced User Behavior", "Behavior-conditioned Finetuning", "Low-rank Parameter Adaptation"],
        "Relations": [
            # Problem flow
            {"source": "Unbalanced User Behavior", "target": "Sparse Tag Distribution", "relation": "causes"},
            {"source": "Sparse Tag Distribution", "target": "Behavior-conditioned Finetuning", "relation": "motivates"},
            
            # Solution flow
            {"source": "Behavior-conditioned Finetuning", "target": "Sparse Tag Distribution", "relation": "mitigates"},
            {"source": "Low-rank Parameter Adaptation", "target": "Behavior-conditioned Finetuning", "relation": "improves"},
            
            # Dependencies
            {"source": "Behavior-conditioned Finetuning", "target": "Low-rank Parameter Adaptation", "relation": "requires"}
        ],
        "Level1": ["LoRA", "Multi-beam Generation", "Weighted Aggregation"]
    }
    
    builder = GraphBuilder()
    G = builder.build_graph(example_hlg)
    
    print("Graph Statistics:")
    print(json.dumps(builder.get_statistics(), indent=2))
    
    # Save as HTML
    builder.save_html("test_graph.html")
    print("\nGraph saved as test_graph.html")


