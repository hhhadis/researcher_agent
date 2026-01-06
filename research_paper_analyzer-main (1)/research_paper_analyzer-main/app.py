"""
Research Logic Graph Extractor - Streamlit Application
Main application for extracting and visualizing hierarchical logic graphs from research papers
"""

import streamlit as st
import os
import json
import tempfile
import colorsys
import numpy as np
from pdf_extractor import extract_text_from_pdf_bytes, get_pdf_metadata
from llm_parser import LLMParser
from graph_builder import GraphBuilder
import streamlit.components.v1 as components


# Page configuration
st.set_page_config(
    page_title="Research Logic Graph Extractor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Load custom CSS for better styling."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f8ff;
        border-left: 4px solid #4A90E2;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0fff4;
        border-left: 4px solid #50C878;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff5f5;
        border-left: 4px solid #E85D75;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    # Single paper mode
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'hlg_data' not in st.session_state:
        st.session_state.hlg_data = None
    if 'graph_html' not in st.session_state:
        st.session_state.graph_html = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Multi-paper mode
    if 'multi_papers' not in st.session_state:
        st.session_state.multi_papers = []  # List of {id, name, text, hlg_data, color}
    if 'multi_graph_data' not in st.session_state:
        st.session_state.multi_graph_data = None
    if 'multi_graph_html' not in st.session_state:
        st.session_state.multi_graph_html = None
    if 'multi_processing_complete' not in st.session_state:
        st.session_state.multi_processing_complete = False
    
    # Embedding mode
    if 'embedding_data' not in st.session_state:
        st.session_state.embedding_data = None
    
    # Cross-researcher mode
    if 'cross_researcher_data' not in st.session_state:
        st.session_state.cross_researcher_data = None
    if 'researcher_1_hlg' not in st.session_state:
        st.session_state.researcher_1_hlg = None
    if 'researcher_2_hlg' not in st.session_state:
        st.session_state.researcher_2_hlg = None
    if 'cross_researcher_graph_html' not in st.session_state:
        st.session_state.cross_researcher_graph_html = None
    
    # Debug: Log state status (comment out in production)
    # st.sidebar.markdown("---")
    # st.sidebar.caption("🔍 State Debug Info")
    # st.sidebar.caption(f"extracted_text: {'✅' if st.session_state.extracted_text else '❌'}")
    # st.sidebar.caption(f"hlg_data: {'✅' if st.session_state.hlg_data else '❌'}")
    # st.sidebar.caption(f"processing_complete: {st.session_state.processing_complete}")


def main():
    """Main application function."""
    load_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🧠 Research Logic Graph Extractor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">建立論文的問題→形式化→解法三層邏輯圖 (Problem→Formulation→Solution)</div>', unsafe_allow_html=True)
    
    # Mode Selection
    mode = st.radio(
        "Select Mode:",
        ["📄 Single Paper Mode", "📚 Multi-Paper Mode", "👥 Cross-Researcher Mode", "🌐 Embedding Mode"],
        horizontal=True,
        help="Single Paper: Analyze one paper | Multi-Paper: Compare multiple papers | Cross-Researcher: Compare two researchers' HLGs | Embedding: 3D visualization of multi-paper nodes"
    )
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        This tool uses LLMs with a **two-pass approach** to extract and visualize the logical structure of research papers.
        
        **Two-Pass (or Three-Pass) Analysis:**
        - 📝 **Pass 1**: Extract all concepts (nodes) from paper
        - 🔗 **Pass 2**: Find ALL relations **grounded in paper text**
        - 🔮 **Pass 3 (Optional)**: Infer contextual nodes & relations
          - Adds concepts not in paper but relevant to research context
          - Infers logical connections and broader research context
          - Visually distinguished with dashed borders/lines
        - ✅ Every element has confidence scores and explanations
        
        **Smart Extraction:**
        - 🎯 Prioritizes key sections (Abstract, Methods, Conclusions)
        - 🚫 Skips low-value content (References, Acknowledgments)
        - ⚡ Optimizes content for LLM analysis
        
        **Three Levels (Problem → Formulation → Solution):**
        - 🔵 **Level 3**: Research field problems & challenges
        - 🟣 **Level 2**: Mathematical/conceptual formulations
        - ⚫ **Level 1**: Technical solutions & algorithms
        
        **Relation Types:**
        
        *Problem → Problem (L3→L3):* causes, contributes-to, related-to, extends, conflicts
        
        *Problem → Formulation (L3→L2):* formulated-as, reduced-to, modeled-as
        
        *Formulation → Solution (L2→L1):* solved-by, implemented-via, optimized-by
        
        *Solution → Formulation (L1→L2):* implements, approximates
        
        *Within-Level (L2→L2, L1→L1):* related-to, improves, extends, conflicts, is-part-of
        
        *Dependencies:* requires, enables
        
        *Validation:* validates, supports
        
        *...or any other type the LLM deems appropriate!*
        """)
        
        st.markdown("---")
        
        # Model selection
        st.header("⚙️ Settings")
        model = st.selectbox(
            "LLM Model",
            [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4-turbo",
                "openai/gpt-4o",
                "meta-llama/llama-3.1-70b-instruct"
            ],
            index=0
        )
        
        max_chars = st.slider(
            "Max Characters to Process",
            min_value=5000,
            max_value=30000,
            value=15000,
            step=1000,
            help="Longer texts may take more time and tokens"
        )
        
        enable_inference = st.checkbox(
            "🔮 Enable Pass 3: Contextual Inference",
            value=False,
            help="Infer additional context nodes and relations beyond what's in the paper. Uses extra tokens."
        )
        
        st.session_state.model = model
        st.session_state.max_chars = max_chars
        st.session_state.enable_inference = enable_inference
        
        # Status indicator
        st.markdown("---")
        st.header("📊 Current Status")
        
        if mode == "📄 Single Paper Mode":
            if st.session_state.hlg_data:
                st.success("✅ Analysis Complete")
                st.caption("Your analysis data is available in all tabs")
            elif st.session_state.extracted_text:
                st.info("📝 Text Extracted - Ready to Analyze")
            else:
                st.warning("⏳ No data yet - Upload a PDF to start")
        elif mode == "📚 Multi-Paper Mode":
            if st.session_state.multi_graph_data:
                st.success(f"✅ Multi-Paper Analysis Complete ({len(st.session_state.multi_papers)} papers)")
                st.caption("Your analysis data is available in all tabs")
            elif st.session_state.multi_papers:
                st.info(f"📝 {len(st.session_state.multi_papers)} Paper(s) Ready to Analyze")
        elif mode == "🌐 Embedding Mode":
            if st.session_state.embedding_data:
                st.success("✅ Embedding Visualization Ready")
                st.caption("3D embedding plot is available")
            else:
                st.info("📝 Upload a multi-paper JSON file to generate 3D embeddings")
        else:  # Cross-Researcher mode
            if st.session_state.cross_researcher_data:
                st.success("✅ Cross-Researcher Analysis Complete")
                st.caption("Your analysis data is available in all tabs")
            else:
                st.info("📝 Load two HLG JSON files to start cross-researcher analysis")
    
    # ==================== SINGLE PAPER MODE ====================
    if mode == "📄 Single Paper Mode":
        render_single_paper_mode()
    
    # ==================== MULTI-PAPER MODE ====================
    elif mode == "📚 Multi-Paper Mode":
        render_multi_paper_mode()
    
    # ==================== EMBEDDING MODE ====================
    elif mode == "🌐 Embedding Mode":
        render_embedding_mode()
    
    # ==================== CROSS-RESEARCHER MODE ====================
    else:  # Cross-Researcher mode
        render_cross_researcher_mode()


def render_single_paper_mode():
    """Render the single paper analysis mode."""
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📁 Load JSON", "📄 Upload PDF", "🔍 Analysis", "🎨 Visualization", "🔍 Self-Evaluation", "💾 Export"])
    
    # Tab 1: Load JSON
    with tab1:
        st.header("📁 Load Analysis from JSON")
        
        st.markdown("""
        **Quick Start: Upload Previously Exported Analysis**
        
        Skip the PDF extraction and LLM analysis steps by uploading a JSON file that was exported from a previous analysis.
        
        **Benefits:**
        - ✅ Instant visualization without LLM API costs
        - 📊 Share analysis results with collaborators
        - 💾 Review past analyses offline
        - ⚡ No waiting for API calls
        
        **Compatible Files:**
        - Files exported from 'Export' tab → 'Download HLG JSON'
        - Any valid hierarchical logic graph JSON with Level1, Level2, Level3 structure
        """)
        
        st.markdown("---")
        
        uploaded_json = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            key="json_upload_tab",
            help="Upload a hierarchical logic graph JSON file exported from a previous analysis"
        )
        
        if uploaded_json is not None:
            try:
                json_content = json.loads(uploaded_json.read().decode('utf-8'))
                
                # Validate basic structure
                required_keys = ["Level3", "Level2", "Level1"]
                if all(key in json_content for key in required_keys):
                    st.session_state.hlg_data = json_content
                    st.session_state.processing_complete = True
                    
                    st.success(f"✅ Successfully loaded: {uploaded_json.name}")
                    
                    # Show preview of loaded data
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Level 3 Nodes", len(json_content.get("Level3", [])))
                    with col2:
                        st.metric("Level 2 Nodes", len(json_content.get("Level2", [])))
                    with col3:
                        st.metric("Level 1 Nodes", len(json_content.get("Level1", [])))
                    
                    st.info("💡 Go to the 'Visualization' tab to see the interactive graph!")
                    
                    # Show quick preview
                    with st.expander("👁️ Preview Loaded Data"):
                        st.json(json_content)
                else:
                    st.error("❌ Invalid JSON format. Missing required keys: Level3, Level2, or Level1")
                    st.info("💡 Make sure you're uploading a file exported from this application's 'Export' tab.")
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading JSON: {str(e)}")
        else:
            st.info("👆 Upload a JSON file to get started, or go to 'Upload PDF' tab to analyze a new paper")
    
    # Tab 2: Upload PDF
    with tab2:
        st.header("📄 Upload Research Paper")
        st.caption("Start from scratch by uploading a PDF and running LLM analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a research paper in PDF format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Extract Text from PDF"):
                    with st.spinner("Extracting text from PDF (section-aware)..."):
                        try:
                            # Extract text with section-aware prioritization
                            pdf_bytes = uploaded_file.read()
                            text, full_length = extract_text_from_pdf_bytes(
                                pdf_bytes,
                                section_aware=True,
                                max_chars=st.session_state.max_chars
                            )
                            st.session_state.extracted_text = text
                            st.session_state.full_text_length = full_length
                            
                            # Reset analysis when new text is extracted
                            st.session_state.hlg_data = None
                            st.session_state.graph_html = None
                            st.session_state.processing_complete = False
                            
                            # Reset to beginning for metadata extraction
                            uploaded_file.seek(0)
                            
                            # Show extraction info with full length comparison
                            extracted_len = len(text)
                            if full_length > extracted_len:
                                st.success(f"✅ Extracted {extracted_len:,} characters (from {full_length:,} total)")
                                st.info(f"ℹ️ Text truncated to priority sections (limit: {st.session_state.max_chars:,} chars). Full paper has {full_length:,} characters.")
                            else:
                                st.success(f"✅ Extracted {extracted_len:,} characters (full paper)")
                            st.info("ℹ️ New text extracted - please run analysis again")
                            
                            # Show preview
                            with st.expander("📖 Text Preview (first 1000 characters)"):
                                st.text(text[:1000] + "..." if len(text) > 1000 else text)
                        
                        except Exception as e:
                            st.error(f"❌ Error extracting text: {str(e)}")
            
            with col2:
                if st.session_state.extracted_text:
                    extracted_len = len(st.session_state.extracted_text)
                    full_len = st.session_state.get('full_text_length', extracted_len)
                    if full_len > extracted_len:
                        st.info(f"📊 Extracted: {extracted_len:,} characters (from {full_len:,} total, limit: {st.session_state.max_chars:,})")
                    else:
                        st.info(f"📊 Text Length: {extracted_len:,} characters (full paper)")
        
        else:
            st.info("👆 Please upload a PDF file to begin")
    
    # Tab 3: Analysis
    with tab3:
        st.header("🔍 LLM Analysis")
        
        if not st.session_state.extracted_text:
            st.warning("⚠️ Please upload and extract text from a PDF first (Tab: Upload)")
        else:
            st.markdown(f"""
            <div class="info-box">
            <strong>Ready for Analysis</strong><br>
            Text length: {len(st.session_state.extracted_text)} characters<br>
            Model: {st.session_state.model}<br>
            Pass 3 (Inference): {"✅ Enabled" if st.session_state.get('enable_inference', False) else "❌ Disabled"}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Analyze Paper with LLM"):
                with st.spinner("🧠 Analyzing paper structure with LLM... This may take a minute..."):
                    try:
                        # Initialize parser
                        parser = LLMParser(model=st.session_state.model)
                        
                        # Parse the paper (with optional Pass 3)
                        hlg_data = parser.parse_paper(
                            st.session_state.extracted_text,
                            max_chars=st.session_state.max_chars,
                            enable_inference=st.session_state.get('enable_inference', False)
                        )
                        
                        st.session_state.hlg_data = hlg_data
                        st.session_state.processing_complete = True
                        
                        st.success("✅ Analysis complete!")
                        
                        # Automatically run deep evaluation
                        with st.spinner("🔍 Running quality evaluation..."):
                            try:
                                eval_result = parser.deep_evaluate_hlg(
                                    hlg_data,
                                    st.session_state.extracted_text
                                )
                                st.session_state.eval_result = eval_result
                                
                                # Display evaluation summary
                                overall_score = eval_result.get("overall_score", 0)
                                score_color = "#50C878" if overall_score >= 8 else "#FFD700" if overall_score >= 6 else "#FFA500" if overall_score >= 4 else "#E85D75"
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {score_color}30 0%, {score_color}10 100%); 
                                            padding: 15px; border-radius: 10px; border-left: 5px solid {score_color}; margin: 15px 0;">
                                    <h3 style="margin: 0; color: {score_color}; font-size: 1.2em;">📊 Quality Assessment: {overall_score}/10</h3>
                                    <p style="margin: 8px 0 0 0; font-size: 0.95em;">{eval_result.get('overall_assessment', '')}</p>
                                    <p style="margin: 5px 0 0 0; font-size: 0.85em; opacity: 0.7;">
                                        ✓ Correctness: {eval_result.get('correctness', {}).get('score', 0)}/10 | 
                                        ⚡ Conciseness: {eval_result.get('conciseness', {}).get('score', 0)}/10 | 
                                        💡 Insightfulness: {eval_result.get('insightfulness', {}).get('score', 0)}/10
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.info("💡 See the 'Self-Evaluation' tab for detailed feedback and improvement suggestions")
                                
                            except Exception as e:
                                st.warning(f"⚠️ Evaluation could not be completed: {str(e)}")
                                st.info("💡 You can run evaluation manually in the 'Self-Evaluation' tab")
                        
                        # Display token usage (with Pass 3 if enabled)
                        if "_token_usage" in hlg_data:
                            usage = hlg_data["_token_usage"]
                            has_pass3 = 'pass3_prompt_tokens' in usage
                            
                            st.markdown(f"### 🔢 Token Usage ({('Three' if has_pass3 else 'Two')}-Pass Analysis)")
                            
                            # Total tokens
                            col_t1, col_t2, col_t3 = st.columns(3)
                            col_t1.metric("📝 Total Prompt Tokens", f"{usage['prompt_tokens']:,}")
                            col_t2.metric("💬 Total Completion Tokens", f"{usage['completion_tokens']:,}")
                            col_t3.metric("🔢 Total Tokens", f"{usage['total_tokens']:,}")
                            
                            # Pass breakdown
                            with st.expander("📊 Pass-by-Pass Breakdown"):
                                if has_pass3:
                                    col_p1, col_p2, col_p3 = st.columns(3)
                                else:
                                    col_p1, col_p2 = st.columns(2)
                                
                                with col_p1:
                                    st.markdown("**Pass 1: Node Extraction**")
                                    st.write(f"Prompt: {usage.get('pass1_prompt_tokens', 0):,}")
                                    st.write(f"Completion: {usage.get('pass1_completion_tokens', 0):,}")
                                    st.write(f"Subtotal: {usage.get('pass1_prompt_tokens', 0) + usage.get('pass1_completion_tokens', 0):,}")
                                
                                with col_p2:
                                    st.markdown("**Pass 2: Relation Finding**")
                                    st.write(f"Prompt: {usage.get('pass2_prompt_tokens', 0):,}")
                                    st.write(f"Completion: {usage.get('pass2_completion_tokens', 0):,}")
                                    st.write(f"Subtotal: {usage.get('pass2_prompt_tokens', 0) + usage.get('pass2_completion_tokens', 0):,}")
                                
                                if has_pass3:
                                    with col_p3:
                                        st.markdown("**Pass 3: Contextual Inference**")
                                        st.write(f"Prompt: {usage.get('pass3_prompt_tokens', 0):,}")
                                        st.write(f"Completion: {usage.get('pass3_completion_tokens', 0):,}")
                                        st.write(f"Subtotal: {usage.get('pass3_prompt_tokens', 0) + usage.get('pass3_completion_tokens', 0):,}")
                        
                        # Display overall confidence
                        if "overall_confidence" in hlg_data:
                            st.markdown(f"""
                            <div class="info-box">
                            <strong>🎯 Overall Analysis Confidence: {hlg_data['overall_confidence']}/10</strong><br>
                            {hlg_data.get('overall_explanation', 'N/A')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Display results
                        st.subheader("📊 Extracted Logic Graph")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### 🔵 Level 3: Problems")
                            st.caption("Research field problems & challenges")
                            for item in hlg_data.get("Level3", []):
                                st.markdown(f"- {item}")
                        
                        with col2:
                            st.markdown("### 🟣 Level 2: Frameworks")
                            st.caption("Mathematical/conceptual frameworks")
                            for item in hlg_data.get("Level2", []):
                                st.markdown(f"- {item}")
                        
                        with col3:
                            st.markdown("### ⚫ Level 1: Solutions")
                            st.caption("Technical implementations & algorithms")
                            for item in hlg_data.get("Level1", []):
                                st.markdown(f"- {item}")
                        
                        # Relations with confidence scores (PAPER-BASED)
                        st.subheader("🔗 Relations from Paper")
                        relations = hlg_data.get("Relations", [])
                        if relations:
                            for i, rel in enumerate(relations, 1):
                                confidence = rel.get('confidence', 'N/A')
                                explanation = rel.get('explanation', '')
                                source = rel.get('source', '')
                                target = rel.get('target', '')
                                
                                # Get level badges for source and target
                                source_badge = get_node_level_badge(source, hlg_data)
                                target_badge = get_node_level_badge(target, hlg_data)
                                confidence_badge = render_confidence_badge(confidence)
                                
                                st.markdown(f"""
                                **[{i}]** {source_badge} **{source}** → *{rel.get('relation')}* → {target_badge} **{target}** {confidence_badge}
                                """, unsafe_allow_html=True)
                                
                                if explanation:
                                    st.caption(f"💡 {explanation}")
                                
                                st.markdown("")  # Spacing
                        
                        # Display inferred nodes and relations (if Pass 3 was enabled)
                        if "InferredNodes" in hlg_data and hlg_data["InferredNodes"]:
                            st.markdown("---")
                            st.subheader("🔮 Inferred Context (Pass 3)")
                            
                            # Inference confidence
                            if "inference_confidence" in hlg_data:
                                st.markdown(f"""
                                <div class="info-box" style="border-left: 4px solid #9B59B6;">
                                <strong>🎯 Inference Quality: {hlg_data['inference_confidence']}/10</strong><br>
                                {hlg_data.get('inference_explanation', 'N/A')}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Inferred Nodes
                            inferred_nodes = hlg_data.get("InferredNodes", [])
                            if inferred_nodes:
                                st.markdown("### 🔮 Inferred Nodes")
                                st.caption("Contextual concepts not explicitly in paper but relevant to understanding")
                                
                                for i, node in enumerate(inferred_nodes, 1):
                                    node_name = node.get('node', 'N/A')
                                    level = node.get('level', 'N/A')
                                    confidence = node.get('confidence', 'N/A')
                                    explanation = node.get('explanation', '')
                                    
                                    # Color code confidence
                                    if isinstance(confidence, (int, float)):
                                        if confidence >= 8:
                                            confidence_color = "#9B59B6"  # Purple
                                        elif confidence >= 6:
                                            confidence_color = "#BB79C6"  # Light purple
                                        elif confidence >= 4:
                                            confidence_color = "#CDA5D6"  # Lighter purple
                                        else:
                                            confidence_color = "#DDB9E6"  # Very light purple
                                        confidence_badge = f'<span style="background-color: {confidence_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{confidence}/10</span>'
                                    else:
                                        confidence_badge = f'<span style="background-color: #999; color: white; padding: 2px 8px; border-radius: 4px;">N/A</span>'
                                    
                                    st.markdown(f"""
                                    **[{i}]** **{node_name}** ({level}) {confidence_badge}
                                    """, unsafe_allow_html=True)
                                    
                                    if explanation:
                                        st.caption(f"💡 {explanation}")
                                    
                                    st.markdown("")
                            
                            # Inferred Relations
                            inferred_relations = hlg_data.get("InferredRelations", [])
                            if inferred_relations:
                                st.markdown("### 🔮 Inferred Relations")
                                st.caption("Logical connections and transitive relations inferred from graph structure")
                                
                                for i, rel in enumerate(inferred_relations, 1):
                                    confidence = rel.get('confidence', 'N/A')
                                    explanation = rel.get('explanation', '')
                                    source = rel.get('source', '')
                                    target = rel.get('target', '')
                                    
                                    # Get level badges for source and target
                                    source_badge = get_node_level_badge(source, hlg_data)
                                    target_badge = get_node_level_badge(target, hlg_data)
                                    
                                    # Color code confidence (purple theme for inferred)
                                    if isinstance(confidence, (int, float)):
                                        if confidence >= 8:
                                            confidence_color = "#9B59B6"  # Purple
                                        elif confidence >= 6:
                                            confidence_color = "#BB79C6"
                                        elif confidence >= 4:
                                            confidence_color = "#CDA5D6"
                                        else:
                                            confidence_color = "#DDB9E6"
                                        confidence_badge = f'<span style="background-color: {confidence_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{confidence}/10</span>'
                                    else:
                                        confidence_badge = f'<span style="background-color: #999; color: white; padding: 2px 8px; border-radius: 4px;">N/A</span>'
                                    
                                    st.markdown(f"""
                                    **[{i}]** {source_badge} **{source}** → *{rel.get('relation')}* → {target_badge} **{target}** {confidence_badge}
                                    """, unsafe_allow_html=True)
                                    
                                    if explanation:
                                        st.caption(f"💡 {explanation}")
                                    
                                    st.markdown("")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        with st.expander("🐛 Error Details"):
                            st.code(traceback.format_exc())
    
    # Tab 4: Visualization
    with tab4:
        st.header("🎨 Interactive Graph Visualization")
        
        # Show status and clear button
        col_status, col_clear = st.columns([3, 1])
        with col_status:
            if st.session_state.hlg_data:
                st.success("✅ Analysis data loaded - Visualization ready!")
        with col_clear:
            if st.session_state.hlg_data:
                if st.button("🗑️ Clear Data", key="clear_viz_data", help="Clear current analysis data"):
                    st.session_state.hlg_data = None
                    st.session_state.graph_html = None
                    st.session_state.processing_complete = False
                    st.experimental_rerun()
        
        if not st.session_state.hlg_data:
            st.warning("⚠️ No analysis data available")
            
            # Show helpful status
            if st.session_state.extracted_text:
                st.info("💡 Text extracted but not analyzed yet. Go to 'Analysis' tab and click 'Analyze Paper with LLM'")
            else:
                st.info("💡 No data yet. Go to 'Load JSON' tab to upload a previous analysis, or 'Upload PDF' tab to start a new analysis")
        else:
            try:
                # Layout selection
                layout_option = st.radio(
                    "📐 Layout Style:",
                    ["Linear (Horizontal Lines)", "Circular (Concentric Circles)"],
                    horizontal=True,
                    help="Choose how nodes are arranged: Linear = horizontal lines by level, Circular = concentric circles"
                )
                layout = "circular" if layout_option == "Circular (Concentric Circles)" else "linear"
                
                # Build graph with evaluation data if available
                builder = GraphBuilder(layout=layout)
                eval_data = st.session_state.get('eval_result')
                G = builder.build_graph(st.session_state.hlg_data, eval_data=eval_data)
                
                # Show evaluation indicators legend if evaluation data exists
                if eval_data:
                    st.info("""
                    **📊 Evaluation Feedback Integrated:**
                    - 🔴 **Red borders** on nodes indicate evaluation issues (wrong level, hallucination, redundancy)
                    - 🔴 **Red edges** indicate problematic relations (inaccurate, hallucination)
                    - Hover over nodes/edges to see detailed evaluation feedback in tooltips
                    """)
                
                # Display statistics
                stats = builder.get_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Nodes", stats['total_nodes'])
                col2.metric("Total Edges", stats['total_edges'])
                col3.metric("Level 3", stats['level3_nodes'])
                col4.metric("Level 2", stats['level2_nodes'])
                
                st.markdown("---")
                
                # Generate visualization
                with st.spinner("Generating interactive graph..."):
                    net = builder.to_pyvis(height="700px", width="100%")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                        net.save_graph(f.name)
                        st.session_state.graph_html = f.name
                        
                        # Read and display
                        with open(f.name, 'r', encoding='utf-8') as html_file:
                            html_content = html_file.read()
                            components.html(html_content, height=750, scrolling=True)
                
                st.info("💡 **Tip**: Clear hierarchical layout! Level 3 (theory) at top, Level 2 (problems/methods) in middle, Level 1 (techniques) at bottom. Inferred nodes have dashed borders, inferred relations have dashed lines. You can drag nodes and zoom in/out!")
                
            except Exception as e:
                st.error(f"❌ Error creating visualization: {str(e)}")
                import traceback
                with st.expander("🐛 Error Details"):
                    st.code(traceback.format_exc())
    
    # Tab 5: Self-Evaluation
    with tab5:
        st.header("🔍 Self-Evaluation")
        st.caption("Assess how well the HLG represents your paper")
        
        if not st.session_state.hlg_data or not st.session_state.extracted_text:
            st.warning("⚠️ No analysis data available to evaluate")
            
            # Show helpful status
            if st.session_state.extracted_text and not st.session_state.hlg_data:
                st.info("💡 Text extracted but not analyzed yet. Go to 'Analysis' tab and click 'Analyze Paper with LLM'")
            else:
                st.info("💡 No data yet. Go to 'Load JSON' tab to upload a previous analysis, or 'Upload PDF' tab to start a new analysis")
        else:
            st.markdown("""
            Evaluate your Hierarchical Logic Graph across three dimensions:
            - **✓ Correctness**: Factual accuracy and proper grounding
            - **⚡ Conciseness**: Signal-to-noise ratio and clarity
            - **💡 Insightfulness**: Understanding value and narrative quality
            """)
            
            st.markdown("---")
            
            # Two evaluation options
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚡ Quick Evaluation")
                st.caption("Instant • Free • Rule-based heuristics")
                st.markdown("""
                **What it checks:**
                - Node connectivity and isolation
                - Level balance (L3/L2/L1 distribution)
                - Density (nodes per 1000 words)
                - Cross-level connections
                - Label length and clarity
                
                **Best for:** Fast quality check
                """)
                
                if st.button("Run Quick Evaluation", key="quick_eval_btn"):
                    with st.spinner("🔍 Running quick heuristic evaluation..."):
                        try:
                            parser = LLMParser(model=st.session_state.model)
                            eval_result = parser.quick_evaluate_hlg(
                                st.session_state.hlg_data,
                                st.session_state.extracted_text
                            )
                            st.session_state.eval_result = eval_result
                            st.success("✅ Quick evaluation complete!")
                        except Exception as e:
                            st.error(f"❌ Error during evaluation: {str(e)}")
            
            with col2:
                st.subheader("🧠 Deep Evaluation")
                st.caption("~30s • Uses tokens • LLM semantic analysis")
                st.markdown("""
                **What it checks:**
                - Semantic accuracy vs paper
                - Missing critical concepts
                - Hallucinations or errors
                - Insightfulness and narrative
                - Specific improvement suggestions
                
                **Best for:** Comprehensive quality assessment
                """)
                
                if st.button("Run Deep Evaluation", key="deep_eval_btn"):
                    with st.spinner("🧠 Running deep LLM evaluation... This may take ~30 seconds..."):
                        try:
                            parser = LLMParser(model=st.session_state.model)
                            eval_result = parser.deep_evaluate_hlg(
                                st.session_state.hlg_data,
                                st.session_state.extracted_text
                            )
                            st.session_state.eval_result = eval_result
                            st.success("✅ Deep evaluation complete!")
                            
                            # Show token usage
                            if "_token_usage" in eval_result:
                                usage = eval_result["_token_usage"]
                                with st.expander("🔢 Token Usage"):
                                    st.write(f"**Prompt tokens:** {usage['prompt_tokens']:,}")
                                    st.write(f"**Completion tokens:** {usage['completion_tokens']:,}")
                                    st.write(f"**Total tokens:** {usage['total_tokens']:,}")
                        except Exception as e:
                            st.error(f"❌ Error during evaluation: {str(e)}")
                            import traceback
                            with st.expander("🐛 Error Details"):
                                st.code(traceback.format_exc())
            
            # Display evaluation results
            if 'eval_result' in st.session_state and st.session_state.eval_result:
                st.markdown("---")
                eval_result = st.session_state.eval_result
                eval_type = eval_result.get("evaluation_type", "unknown")
                
                # Overall score banner
                overall_score = eval_result.get("overall_score", 0)
                score_color = "#50C878" if overall_score >= 8 else "#FFD700" if overall_score >= 6 else "#FFA500" if overall_score >= 4 else "#E85D75"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {score_color}30 0%, {score_color}10 100%); 
                            padding: 20px; border-radius: 10px; border-left: 5px solid {score_color}; margin: 20px 0;">
                    <h2 style="margin: 0; color: {score_color};">📊 Overall Quality: {overall_score}/10</h2>
                    <p style="margin: 10px 0 0 0; font-size: 1.1em;">{eval_result.get('overall_assessment', '')}</p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;">
                        {'⚡ Quick Heuristic Evaluation' if eval_type == 'quick_heuristic' else '🧠 Deep LLM Evaluation'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Three dimensions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    correctness = eval_result.get("correctness", {})
                    score = correctness.get("score", 0)
                    st.markdown(f"### ✓ Correctness: {score}/10")
                    st.caption(correctness.get("explanation", ""))
                    
                    if correctness.get("strengths"):
                        with st.expander("✓ Strengths"):
                            for strength in correctness["strengths"]:
                                st.markdown(f"- {strength}")
                    
                    if correctness.get("issues"):
                        with st.expander("⚠ Issues Found", expanded=True):
                            for issue in correctness["issues"]:
                                st.markdown(f"**{issue.get('type', 'Issue')}**: {issue.get('description', '')}")
                
                with col2:
                    conciseness = eval_result.get("conciseness", {})
                    score = conciseness.get("score", 0)
                    st.markdown(f"### ⚡ Conciseness: {score}/10")
                    st.caption(conciseness.get("explanation", ""))
                    
                    if conciseness.get("suggestions"):
                        with st.expander("💡 Suggestions", expanded=True):
                            for suggestion in conciseness["suggestions"]:
                                st.markdown(f"- {suggestion}")
                    
                    if conciseness.get("issues"):
                        with st.expander("⚠ Issues Found"):
                            for issue in conciseness["issues"]:
                                st.markdown(f"**{issue.get('type', 'Issue')}**: {issue.get('description', '')}")
                
                with col3:
                    insightfulness = eval_result.get("insightfulness", {})
                    score = insightfulness.get("score", 0)
                    st.markdown(f"### 💡 Insightfulness: {score}/10")
                    st.caption(insightfulness.get("explanation", ""))
                    
                    if insightfulness.get("strengths"):
                        with st.expander("✓ Strengths"):
                            for strength in insightfulness["strengths"]:
                                st.markdown(f"- {strength}")
                    
                    if insightfulness.get("missing_insights"):
                        with st.expander("💭 Could Add", expanded=True):
                            for insight in insightfulness["missing_insights"]:
                                st.markdown(f"- {insight}")
                
                # Improvement suggestions
                if eval_result.get("improvement_suggestions"):
                    st.markdown("---")
                    st.markdown("### 🎯 Prioritized Improvement Suggestions")
                    for i, suggestion in enumerate(eval_result["improvement_suggestions"], 1):
                        st.markdown(f"{i}. {suggestion}")
    
    # Tab 6: Export
    with tab6:
        st.header("💾 Export Results")
        
        # Show status at top of tab
        if st.session_state.hlg_data:
            st.success("✅ Analysis data loaded - Export options ready!")
        
        if not st.session_state.hlg_data:
            st.warning("⚠️ No analysis data available to export")
            
            # Show helpful status
            if st.session_state.extracted_text:
                st.info("💡 Text extracted but not analyzed yet. Go to 'Analysis' tab and click 'Analyze Paper with LLM'")
            else:
                st.info("💡 No data yet. Go to 'Load JSON' tab to upload a previous analysis, or 'Upload PDF' tab to start a new analysis")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Download JSON")
                json_str = json.dumps(st.session_state.hlg_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Download HLG JSON",
                    data=json_str,
                    file_name="hierarchical_logic_graph.json",
                    mime="application/json"
                )
                
                with st.expander("👁️ Preview JSON"):
                    st.json(st.session_state.hlg_data)
            
            with col2:
                st.subheader("🌐 Download Graph HTML")
                
                # Layout selection for HTML export
                export_layout_option = st.radio(
                    "📐 Layout for Export:",
                    ["Linear (Horizontal Lines)", "Circular (Concentric Circles)"],
                    horizontal=True,
                    help="Choose layout style for the exported HTML graph"
                )
                export_layout = "circular" if export_layout_option == "Circular (Concentric Circles)" else "linear"
                
                # Generate HTML with selected layout
                try:
                    builder = GraphBuilder(layout=export_layout)
                    eval_data = st.session_state.get('eval_result')
                    builder.build_graph(st.session_state.hlg_data, eval_data=eval_data)
                    
                    # Generate HTML
                    net = builder.to_pyvis(height="700px", width="100%")
                    
                    # Save to temporary file to get HTML content
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
                    temp_path = temp_file.name
                    temp_file.close()  # Close immediately after getting path
                    
                    net.save_graph(temp_path)
                    
                    # Read the HTML content
                    with open(temp_path, 'r', encoding='utf-8') as html_file:
                        html_content = html_file.read()
                    
                    # Clean up temp file (close file first, then delete)
                    try:
                        os.unlink(temp_path)
                    except:
                        pass  # Ignore errors if file is still locked
                    
                    layout_suffix = "circular" if export_layout == "circular" else "linear"
                    st.download_button(
                        label=f"⬇️ Download Interactive Graph ({export_layout_option})",
                        data=html_content,
                        file_name=f"logic_graph_{layout_suffix}.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating HTML: {str(e)}")
                    st.info("📊 Make sure analysis data is available")
            
            st.markdown("---")
            
            # Export graph structure JSON
            st.subheader("🔗 Download Graph Structure")
            try:
                builder = GraphBuilder()
                builder.build_graph(st.session_state.hlg_data)
                graph_json = builder.to_json()
                
                st.download_button(
                    label="⬇️ Download Graph Structure JSON",
                    data=graph_json,
                    file_name="graph_structure.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Error generating graph structure: {str(e)}")


def render_confidence_badge(confidence):
    """Helper function to render confidence badge with color coding."""
    if isinstance(confidence, (int, float)):
        if confidence >= 8:
            confidence_color = "#50C878"  # Green
        elif confidence >= 6:
            confidence_color = "#FFD700"  # Yellow
        elif confidence >= 4:
            confidence_color = "#FFA500"  # Orange
        else:
            confidence_color = "#E85D75"  # Red
        return f'<span style="background-color: {confidence_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{confidence}/10</span>'
    else:
        return f'<span style="background-color: #999; color: white; padding: 2px 8px; border-radius: 4px;">N/A</span>'


def get_node_level_badge(node_name, hlg_data):
    """Helper function to get level badge for a node based on HLG data."""
    # Check Level 3
    if node_name in hlg_data.get("Level3", []):
        return '<span style="background-color: #4A90E2; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; font-weight: bold;">L3</span>'
    
    # Check Level 2
    if node_name in hlg_data.get("Level2", []):
        return '<span style="background-color: #9B59B6; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; font-weight: bold;">L2</span>'
    
    # Check Level 1
    if node_name in hlg_data.get("Level1", []):
        return '<span style="background-color: #9B9B9B; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; font-weight: bold;">L1</span>'
    
    # Check Inferred Nodes
    for inferred_node in hlg_data.get("InferredNodes", []):
        if inferred_node.get("node") == node_name:
            level = inferred_node.get("level", "")
            if level == "Level3":
                return '<span style="background-color: #4A90E2; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; font-weight: bold; border: 1px dashed white;">L3*</span>'
            elif level == "Level2":
                return '<span style="background-color: #9B59B6; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; font-weight: bold; border: 1px dashed white;">L2*</span>'
            elif level == "Level1":
                return '<span style="background-color: #9B9B9B; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; font-weight: bold; border: 1px dashed white;">L1*</span>'
    
    # Unknown level
    return '<span style="background-color: #666; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.85em;">?</span>'


def hsl_to_hex(h, s, l):
    """Convert HSL color to hex format.
    
    Args:
        h: Hue (0-360)
        s: Saturation (0-100)
        l: Lightness (0-100)
    
    Returns:
        Hex color string (e.g., "#FF0000")
    """
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def generate_paper_color(index, base_colors):
    """Generate a distinct color for a paper.
    
    Args:
        index: Paper index (0-based)
        base_colors: List of base color dictionaries
        
    Returns:
        Color dictionary with name, hex, and level colors
    """
    # Use predefined colors for first 5 papers
    if index < len(base_colors):
        return base_colors[index]
    
    # Generate distinct colors for papers beyond 5 using HSL color space
    # Distribute colors evenly around the color wheel
    num_extra = index - len(base_colors)
    total_extra = num_extra + 1  # How many extra colors we need
    
    # Use golden angle for even distribution (137.5 degrees)
    hue_step = 137.5
    hue = (len(base_colors) * hue_step + num_extra * hue_step) % 360
    
    # Use medium saturation and lightness for good visibility
    saturation = 70
    lightness = 50
    
    # Generate main color
    hex_color = hsl_to_hex(hue, saturation, lightness)
    
    # Generate level-specific colors (slightly adjusted)
    l3_color = hsl_to_hex(hue, saturation, lightness)
    l2p_color = hsl_to_hex((hue + 30) % 360, saturation - 10, lightness + 10)
    l2m_color = hsl_to_hex((hue + 60) % 360, saturation - 5, lightness + 15)
    l1_color = hsl_to_hex(hue, saturation - 40, lightness + 30)
    
    # Generate color name based on hue
    color_names = {
        (0, 30): "Red", (30, 60): "Orange", (60, 90): "Yellow",
        (90, 150): "Green", (150, 210): "Cyan", (210, 270): "Blue",
        (270, 300): "Indigo", (300, 330): "Violet", (330, 360): "Magenta"
    }
    color_name = "Custom"
    for (start, end), name in color_names.items():
        if start <= hue < end:
            color_name = name
            break
    
    return {
        "name": f"{color_name} {index + 1}",
        "hex": hex_color,
        "l3": l3_color,
        "l2p": l2p_color,
        "l2m": l2m_color,
        "l1": l1_color
    }


def render_cross_researcher_mode():
    """Render the cross-researcher analysis mode."""
    st.header("👥 Cross-Researcher Analysis Mode")
    st.markdown("Compare two researchers' hierarchical logic graphs to find cross-researcher relations.")
    
    # Initialize session state
    if 'cross_researcher_data' not in st.session_state:
        st.session_state.cross_researcher_data = None
    if 'researcher_1_hlg' not in st.session_state:
        st.session_state.researcher_1_hlg = None
    if 'researcher_2_hlg' not in st.session_state:
        st.session_state.researcher_2_hlg = None
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Load HLGs", "🔍 Analysis", "🎨 Visualization", "💾 Export"])
    
    # Tab 1: Load HLGs
    with tab1:
        st.markdown("### Load Researcher HLGs")
        st.caption("Upload two HLG JSON files exported from Multi-Paper Mode (one for each researcher)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Researcher 1 (R1)")
            r1_file = st.file_uploader(
                "Upload R1 HLG JSON",
                type=['json'],
                key="r1_hlg_upload",
                help="Upload the HLG JSON file for Researcher 1"
            )
            
            if r1_file is not None:
                try:
                    r1_data = json.load(r1_file)
                    # Validate it's a multi-paper HLG structure
                    if 'papers' in r1_data and 'cross_paper_relations' in r1_data:
                        st.session_state.researcher_1_hlg = r1_data
                        st.success(f"✅ R1 HLG loaded: {len(r1_data.get('papers', []))} papers")
                    else:
                        st.error("❌ Invalid HLG format. Please upload a JSON file exported from Multi-Paper Mode.")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file. Please check the file format.")
        
        with col2:
            st.markdown("#### Researcher 2 (R2)")
            r2_file = st.file_uploader(
                "Upload R2 HLG JSON",
                type=['json'],
                key="r2_hlg_upload",
                help="Upload the HLG JSON file for Researcher 2"
            )
            
            if r2_file is not None:
                try:
                    r2_data = json.load(r2_file)
                    # Validate it's a multi-paper HLG structure
                    if 'papers' in r2_data and 'cross_paper_relations' in r2_data:
                        st.session_state.researcher_2_hlg = r2_data
                        st.success(f"✅ R2 HLG loaded: {len(r2_data.get('papers', []))} papers")
                    else:
                        st.error("❌ Invalid HLG format. Please upload a JSON file exported from Multi-Paper Mode.")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file. Please check the file format.")
        
        # Show loaded HLG summaries
        if st.session_state.researcher_1_hlg and st.session_state.researcher_2_hlg:
            st.markdown("---")
            st.markdown("### HLG Summaries")
            
            col1, col2 = st.columns(2)
            
            with col1:
                r1_hlg = st.session_state.researcher_1_hlg
                r1_papers = r1_hlg.get('papers', [])
                total_r1_nodes = sum(
                    len(p.get('hlg_data', {}).get('Level3', [])) +
                    len(p.get('hlg_data', {}).get('Level2', [])) +
                    len(p.get('hlg_data', {}).get('Level1', []))
                    for p in r1_papers
                )
                st.markdown(f"**Researcher 1 (R1):**")
                st.markdown(f"- Papers: {len(r1_papers)}")
                st.markdown(f"- Total Nodes: {total_r1_nodes}")
                st.markdown(f"- Cross-Paper Relations: {len(r1_hlg.get('cross_paper_relations', []))}")
            
            with col2:
                r2_hlg = st.session_state.researcher_2_hlg
                r2_papers = r2_hlg.get('papers', [])
                total_r2_nodes = sum(
                    len(p.get('hlg_data', {}).get('Level3', [])) +
                    len(p.get('hlg_data', {}).get('Level2', [])) +
                    len(p.get('hlg_data', {}).get('Level1', []))
                    for p in r2_papers
                )
                st.markdown(f"**Researcher 2 (R2):**")
                st.markdown(f"- Papers: {len(r2_papers)}")
                st.markdown(f"- Total Nodes: {total_r2_nodes}")
                st.markdown(f"- Cross-Paper Relations: {len(r2_hlg.get('cross_paper_relations', []))}")
    
    # Tab 2: Analysis
    with tab2:
        st.markdown("### Cross-Researcher Relation Analysis")
        
        if not st.session_state.researcher_1_hlg or not st.session_state.researcher_2_hlg:
            st.warning("⚠️ Please load both HLG files in the 'Load HLGs' tab first.")
        else:
            if st.button("🔍 Find Cross-Researcher Relations"):
                with st.spinner("Analyzing cross-researcher relations..."):
                    try:
                        parser = LLMParser(model=st.session_state.model)
                        
                        # Combine all papers from each researcher into a single HLG structure
                        def combine_researcher_hlg(multi_paper_data):
                            """Combine multiple papers into a single HLG structure."""
                            combined = {
                                "Level3": [],
                                "Level2": [],
                                "Level1": [],
                                "Relations": []
                            }
                            
                            for paper in multi_paper_data.get('papers', []):
                                hlg = paper.get('hlg_data', {})
                                # Collect unique nodes (avoid duplicates)
                                for level in ["Level3", "Level2", "Level1"]:
                                    for node in hlg.get(level, []):
                                        if node not in combined[level]:
                                            combined[level].append(node)
                                
                                # Collect intra-paper relations
                                for rel in hlg.get("Relations", []):
                                    combined["Relations"].append(rel)
                            
                            return combined
                        
                        r1_combined_hlg = combine_researcher_hlg(st.session_state.researcher_1_hlg)
                        r2_combined_hlg = combine_researcher_hlg(st.session_state.researcher_2_hlg)
                        
                        # Find cross-researcher relations
                        cross_researcher_result = parser.find_cross_researcher_relations(
                            r1_combined_hlg,
                            r2_combined_hlg
                        )
                        
                        # Store results
                        st.session_state.cross_researcher_data = {
                            "researcher_1": {
                                "name": "R1",
                                "hlg_data": r1_combined_hlg,
                                "original_data": st.session_state.researcher_1_hlg
                            },
                            "researcher_2": {
                                "name": "R2",
                                "hlg_data": r2_combined_hlg,
                                "original_data": st.session_state.researcher_2_hlg
                            },
                            "cross_researcher_relations": cross_researcher_result.get("relations", []),
                            "overall_confidence": cross_researcher_result.get("overall_confidence", 0),
                            "overall_explanation": cross_researcher_result.get("overall_explanation", ""),
                            "_token_usage": cross_researcher_result.get("_token_usage", {})
                        }
                        
                        st.success(f"✅ Cross-Researcher Analysis Complete!")
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display results if available
            if st.session_state.cross_researcher_data:
                st.markdown("---")
                st.markdown("### Analysis Results")
                
                relations = st.session_state.cross_researcher_data.get("cross_researcher_relations", [])
                overall_confidence = st.session_state.cross_researcher_data.get("overall_confidence", 0)
                overall_explanation = st.session_state.cross_researcher_data.get("overall_explanation", "")
                
                col1, col2 = st.columns(2)
                col1.metric("Cross-Researcher Relations", len(relations))
                col2.metric("Overall Confidence", f"{overall_confidence}/10")
                
                if overall_explanation:
                    st.info(f"💡 {overall_explanation}")
                
                if relations:
                    st.markdown("#### Cross-Researcher Relations")
                    for i, rel in enumerate(relations, 1):
                        source = rel.get("source", "N/A")
                        source_researcher = rel.get("source_researcher", "N/A")
                        target = rel.get("target", "N/A")
                        target_researcher = rel.get("target_researcher", "N/A")
                        rel_type = rel.get("relation", "N/A")
                        confidence = rel.get("confidence", "N/A")
                        explanation = rel.get("explanation", "")
                        
                        st.markdown(f"**[{i}] [{source_researcher}] {source} → *{rel_type}* → [{target_researcher}] {target}** (Confidence: {confidence}/10)")
                        if explanation:
                            st.caption(f"💡 {explanation}")
                        st.markdown("")
                else:
                    st.info("ℹ️ No cross-researcher relations found.")
    
    # Tab 3: Visualization
    with tab3:
        st.markdown("### Cross-Researcher Graph Visualization")
        
        if not st.session_state.cross_researcher_data:
            st.warning("⚠️ Please run analysis in the 'Analysis' tab first.")
        else:
            # Layout selection
            layout_option = st.radio(
                "📐 Layout Style:",
                ["Linear (Horizontal Lines)", "Circular (Concentric Circles)"],
                horizontal=True,
                help="Choose how nodes are arranged"
            )
            layout = "circular" if layout_option == "Circular (Concentric Circles)" else "linear"
            
            try:
                builder = GraphBuilder(layout=layout)
                G = builder.build_cross_researcher_graph(st.session_state.cross_researcher_data)
                
                # Display statistics
                stats = builder.get_statistics()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Nodes", stats['total_nodes'])
                col2.metric("Total Edges", stats['total_edges'])
                col3.metric("Cross-Researcher Relations", len(st.session_state.cross_researcher_data.get('cross_researcher_relations', [])))
                
                st.markdown("---")
                
                # Generate visualization
                with st.spinner("Generating interactive graph..."):
                    net = builder.to_pyvis(height="700px", width="100%")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                        net.save_graph(f.name)
                        st.session_state.cross_researcher_graph_html = f.name
                        
                        # Read and display
                        with open(f.name, 'r', encoding='utf-8') as html_file:
                            html_content = html_file.read()
                            components.html(html_content, height=750, scrolling=True)
                
            except Exception as e:
                st.error(f"❌ Error generating visualization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Tab 4: Export
    with tab4:
        st.markdown("### Export Results")
        
        if not st.session_state.cross_researcher_data:
            st.warning("⚠️ No cross-researcher analysis data available to export.")
        else:
            # Export JSON
            json_str = json.dumps(st.session_state.cross_researcher_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Cross-Researcher Analysis (JSON)",
                data=json_str,
                file_name="cross_researcher_analysis.json",
                mime="application/json"
            )
            
            # Export HTML graph if available
            if 'cross_researcher_graph_html' in st.session_state:
                with open(st.session_state.cross_researcher_graph_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    st.download_button(
                        label="📥 Download Interactive Graph (HTML)",
                        data=html_content,
                        file_name="cross_researcher_graph.html",
                        mime="text/html"
                    )


def generate_2d_networkx_visualization(nodes_data, embeddings_3d, edges, nodes_by_paper, selected_paper_ids, 
                                        show_labels, node_size, plot_height, min_confidence=5, show_cluster_labels=False):
    """Generate 2D NetworkX-based visualization using PyVis with clustering."""
    import networkx as nx
    from pyvis.network import Network
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    # Filter to visible nodes only
    visible_indices = []
    visible_node_mapping = {}  # Map original index to visible index
    for i, node in enumerate(nodes_data):
        if node['paper_id'] in selected_paper_ids:
            visible_indices.append(i)
            visible_node_mapping[i] = len(visible_indices) - 1
    
    if not visible_indices:
        return "<html><body>No nodes to visualize</body></html>", {'n_clusters': 0, 'method': 'None', 'silhouette_score': None, 'cluster_colors': []}
    
    # Get embeddings for visible nodes
    visible_embeddings = embeddings_3d[visible_indices]
    
    # Step 1: Auto-detect clusters in original embedding space
    n_nodes = len(visible_embeddings)
    
    # Try different clustering methods and pick the best
    best_clusters = None
    best_score = -1
    best_method = None
    
    # Method 1: DBSCAN (auto-detects clusters)
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=3, metric='cosine' if embeddings_3d.shape[1] > 10 else 'euclidean')
        dbscan_labels = dbscan.fit_predict(visible_embeddings)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        if n_clusters_dbscan > 1:
            # Calculate silhouette score (only for non-noise points)
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:
                score = silhouette_score(visible_embeddings[non_noise_mask], 
                                       dbscan_labels[non_noise_mask])
                if score > best_score:
                    best_score = score
                    best_clusters = dbscan_labels
                    best_method = "DBSCAN"
    except:
        pass
    
    # Method 2: K-means with optimal K (using silhouette score)
    if n_nodes >= 4:  # Need at least 4 points for meaningful clustering
        max_k = min(int(np.sqrt(n_nodes)) + 2, n_nodes // 2, 10)  # Reasonable upper bound
        max_k = max(2, max_k)  # At least 2 clusters
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(visible_embeddings)
                score = silhouette_score(visible_embeddings, kmeans_labels)
                
                if score > best_score:
                    best_score = score
                    best_clusters = kmeans_labels
                    best_method = f"KMeans (k={k})"
            except:
                continue
    
    # Fallback: If no good clusters found, use 2 clusters
    if best_clusters is None or len(set(best_clusters)) < 2:
        kmeans = KMeans(n_clusters=min(2, n_nodes), random_state=42, n_init=10)
        best_clusters = kmeans.fit_predict(visible_embeddings)
        best_method = "KMeans (k=2, fallback)"
    
    # Step 2: Reduce to 2D using t-SNE (better for preserving clusters than PCA)
    if visible_embeddings.shape[1] > 2:
        # Use t-SNE with cluster initialization for better results
        perplexity = min(30, max(5, n_nodes - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   init='pca', learning_rate='auto', n_iter=1000)
        embeddings_2d = tsne.fit_transform(visible_embeddings)
    else:
        embeddings_2d = visible_embeddings[:, :2]
    
    # Generate distinct colors for clusters
    # Use hue range 0.55 to 1.0 (cyan to red spectrum) to avoid conflicts with researchers
    unique_clusters = sorted(set(best_clusters))
    # Remove -1 (noise) from cluster count for color generation
    valid_clusters = [c for c in unique_clusters if c != -1]
    n_clusters = len(valid_clusters)
    
    cluster_colors = {}
    import colorsys
    
    # Generate colors for valid clusters
    if n_clusters > 0:
        for idx, cluster_id in enumerate(valid_clusters):
            # Map to hue range 0.55-1.0 (cyan to red) to separate from researchers (0.0-0.45)
            if n_clusters > 1:
                hue = 0.55 + (idx / max(n_clusters - 1, 1)) * 0.45
            else:
                hue = 0.775  # Middle of cluster range for single cluster
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)  # Increased saturation and brightness for better visibility
            cluster_colors[cluster_id] = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
    else:
        # Fallback: if no valid clusters, assign a default color
        if len(unique_clusters) > 0:
            for cluster_id in unique_clusters:
                if cluster_id != -1:
                    cluster_colors[cluster_id] = "#FF6B6B"  # Default red color
    
    # Handle noise points (DBSCAN label -1)
    noise_color = "#cccccc"
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with positions and cluster assignments
    node_positions = {}
    node_attributes = {}
    
    # Create mapping from cluster_id to display index
    cluster_id_to_index = {}
    for display_idx, cluster_id in enumerate(valid_clusters):
        cluster_id_to_index[cluster_id] = display_idx + 1
    
    for idx, orig_idx in enumerate(visible_indices):
        node = nodes_data[orig_idx]
        node_name = node['node_name']
        cluster_id = int(best_clusters[idx])
        
        x, y = float(embeddings_2d[idx, 0]), float(embeddings_2d[idx, 1])
        
        # Use cluster color, or noise color if cluster_id is -1
        if cluster_id == -1:
            cluster_color = noise_color
            cluster_label = "Noise"
        else:
            cluster_color = cluster_colors.get(cluster_id, noise_color)
            cluster_display_idx = cluster_id_to_index.get(cluster_id, cluster_id)
            cluster_label = f"Cluster {cluster_display_idx}"
        
        G.add_node(node_name)
        node_positions[node_name] = (x, y)
        node_attributes[node_name] = {
            'paper': node['paper_name'],
            'paper_color': node.get('paper_color', node.get('researcher_color', '#4A90E2')),
            'researcher_id': node.get('researcher_id', 'Unknown'),
            'researcher_color': node.get('researcher_color', '#4A90E2'),
            'cluster_color': cluster_color,
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'paper_id': node['paper_id']
        }
    
    # Add edges
    visible_node_names = {node['node_name'] for node in nodes_data if node['paper_id'] in selected_paper_ids}
    
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        confidence = edge.get('confidence', 5)
        
        # Filter by confidence
        if confidence < min_confidence:
            continue
        
        if source in visible_node_names and target in visible_node_names:
            if source in node_positions and target in node_positions:
                source_paper = edge.get('source_paper', '')
                target_paper = edge.get('target_paper', '')
                is_cross_paper = source_paper != target_paper
                
                G.add_edge(source, target, 
                          is_cross_paper=is_cross_paper,
                          relation=edge.get('relation', 'related-to'),
                          confidence=confidence)
    
    # Create PyVis network
    net = Network(
        height=f"{plot_height}px",
        width="100%",
        directed=False,
        notebook=False,
        bgcolor="#ffffff",
        font_color="black"
    )
    
    # Configure physics for balanced force-directed layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 250},
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.08,
          "springLength": 200,
          "springConstant": 0.03,
          "damping": 0.12,
          "avoidOverlap": 0.8
        }
      },
      "nodes": {
        "font": {
          "size": 13,
          "face": "Arial"
        },
        "borderWidth": 4,
        "shape": "dot",
        "scaling": {
          "min": 15,
          "max": 35
        },
        "margin": 8
      },
      "edges": {
        "width": 1.5,
        "smooth": {
          "type": "continuous",
          "roundness": 0.5
        },
        "arrows": {
          "to": {
            "enabled": false
          }
        },
        "color": {
          "opacity": 0.4
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)
    
    # Add nodes with fixed positions (scaled for better visualization)
    # Scale positions to fit in a reasonable range
    if node_positions:
        all_x = [pos[0] for pos in node_positions.values()]
        all_y = [pos[1] for pos in node_positions.values()]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        # Scale to balanced range (-600 to 600)
        scale = 500 / max(x_range, y_range) if max(x_range, y_range) > 0 else 1
        
        for node_name, (x, y) in node_positions.items():
            attrs = node_attributes[node_name]
            # Scale and center positions
            scaled_x = (x - (x_min + x_max) / 2) * scale
            scaled_y = (y - (y_min + y_max) / 2) * scale
            
            # Truncate long labels for readability
            display_label = ""
            if show_cluster_labels:
                # Show cluster label instead of node name
                display_label = attrs['cluster_label']
            elif show_labels:
                if len(node_name) > 25:
                    display_label = node_name[:22] + "..."
                else:
                    display_label = node_name
            
            # Make nodes bigger - increased size multiplier
            adjusted_size = max(15, min(node_size * 1.2, 35))
            
            # Fill color = cluster, Border color = researcher
            # Ensure cluster_color exists and is valid
            cluster_color = attrs.get('cluster_color', '#000000')
            if not cluster_color or (cluster_color == '#cccccc' and attrs.get('cluster_id') != -1):
                # Fallback: use a default color if cluster color is missing
                cluster_color = '#FF6B6B'
            
            # PyVis requires color as a dictionary with background and border keys
            node_color = {
                "background": cluster_color,  # Fill color shows cluster
                "border": attrs['researcher_color']  # Border color shows researcher
            }
            
            net.add_node(
                node_name,
                label=display_label,
                color=node_color,  # Pass as dictionary with background and border
                title=f"{node_name}<br>Researcher: {attrs.get('researcher_id', 'Unknown')}<br>Paper: {attrs['paper']}<br>Cluster: {attrs['cluster_label']}",
                x=scaled_x,
                y=scaled_y,
                size=adjusted_size,
                physics=True,  # Enable physics for better spacing
                borderWidth=4  # Thicker border to make cluster color more visible
            )
    
    # Add edges with reduced visibility for less crowding
    for source, target, data in G.edges(data=True):
        is_cross_paper = data.get('is_cross_paper', False)
        confidence = data.get('confidence', 5)
        
        # Balanced edge visibility
        if is_cross_paper:
            edge_color = {"color": "#ff6666", "highlight": "#ff0000", "opacity": 0.5}
            edge_width = 2.5
        else:
            edge_color = {"color": "#aaaaaa", "highlight": "#888888", "opacity": 0.35}
            edge_width = 1.5
        
        net.add_edge(
            source,
            target,
            color=edge_color,
            width=edge_width,
            title=f"{data.get('relation', 'related-to')} (confidence: {confidence}/10)",
            hidden=False
        )
    
    # Generate HTML
    html_content = net.generate_html()
    
    # Count nodes per cluster for statistics and build node lists
    cluster_counts = {}
    cluster_nodes = {}  # Map cluster_label -> list of node names
    
    for node_name, attrs in node_attributes.items():
        cluster_label = attrs['cluster_label']
        cluster_counts[cluster_label] = cluster_counts.get(cluster_label, 0) + 1
        
        if cluster_label not in cluster_nodes:
            cluster_nodes[cluster_label] = []
        cluster_nodes[cluster_label].append({
            'name': node_name,
            'paper': attrs['paper'],
            'researcher_id': attrs.get('researcher_id', 'Unknown'),
            'researcher_color': attrs.get('researcher_color', '#4A90E2')
        })
    
    # Return HTML and cluster info
    cluster_info = {
        'n_clusters': n_clusters + (1 if -1 in best_clusters else 0),  # Include noise as a "cluster" for display
        'method': best_method,
        'silhouette_score': float(best_score) if best_score > -1 else None,
        'cluster_colors': list(cluster_colors.values()),
        'cluster_counts': cluster_counts,
        'cluster_color_map': {attrs['cluster_label']: attrs['cluster_color'] 
                             for attrs in node_attributes.values()},
        'cluster_nodes': cluster_nodes  # Add node lists per cluster
    }
    
    return html_content, cluster_info


def generate_threejs_visualization(nodes_data, embeddings_3d, edges, nodes_by_paper, selected_paper_ids, 
                                    show_labels, node_size, plot_height):
    """Generate Three.js-based 3D visualization HTML (DEPRECATED - use 2D instead)."""
    
    # Prepare node data
    nodes_js = []
    node_id_to_index = {}
    
    for i, node in enumerate(nodes_data):
        paper_id = node['paper_id']
        if paper_id not in selected_paper_ids:
            continue
            
        node_id_to_index[node['node_name']] = i
        nodes_js.append({
            'name': node['node_name'],
            'x': float(embeddings_3d[i, 0]),
            'y': float(embeddings_3d[i, 1]),
            'z': float(embeddings_3d[i, 2]),
            'color': node['paper_color'],
            'paper': node['paper_name']
        })
    
    # Prepare edge data
    edges_js = []
    visible_node_names = {node['node_name'] for node in nodes_data if node['paper_id'] in selected_paper_ids}
    
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in visible_node_names and target in visible_node_names:
            if source in node_id_to_index and target in node_id_to_index:
                source_paper = edge.get('source_paper', '')
                target_paper = edge.get('target_paper', '')
                is_cross_paper = source_paper != target_paper
                edges_js.append({
                    'source': source,
                    'target': target,
                    'isCrossPaper': is_cross_paper
                })
    
    # Calculate center and scale for camera
    if nodes_js:
        all_x = [n['x'] for n in nodes_js]
        all_y = [n['y'] for n in nodes_js]
        all_z = [n['z'] for n in nodes_js]
        center_x = (max(all_x) + min(all_x)) / 2
        center_y = (max(all_y) + min(all_y)) / 2
        center_z = (max(all_z) + min(all_z)) / 2
        scale = max(max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)) * 1.5
    else:
        center_x = center_y = center_z = 0
        scale = 10
    
    # Generate HTML with Three.js
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>3D Embedding Visualization</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }}
            #container {{
                width: 100%;
                height: {plot_height}px;
            }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255, 255, 255, 0.9);
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 100;
            }}
            #legend {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.9);
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 100;
                max-width: 200px;
            }}
            .legend-item {{
                margin: 5px 0;
                display: flex;
                align-items: center;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                margin-right: 8px;
                border: 1px solid #ccc;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <div id="info">
            <div><strong>Controls:</strong></div>
            <div>🖱️ Left Click + Drag: Rotate</div>
            <div>🖱️ Right Click + Drag: Pan</div>
            <div>🖱️ Scroll: Zoom</div>
            <div>🖱️ Hover: See node info</div>
        </div>
        <div id="legend">
            <div><strong>Papers:</strong></div>
            <div id="legend-content"></div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            // Simple OrbitControls implementation
            function SimpleOrbitControls(camera, domElement) {{
                this.object = camera;
                this.domElement = domElement;
                this.enableDamping = true;
                this.dampingFactor = 0.05;
                this.minDistance = 0;
                this.maxDistance = Infinity;
                
                let isMouseDown = false;
                let mouseX = 0, mouseY = 0;
                let spherical = new THREE.Spherical();
                spherical.setFromVector3(camera.position);
                
                const onMouseDown = (e) => {{
                    isMouseDown = true;
                    mouseX = e.clientX;
                    mouseY = e.clientY;
                }};
                
                const onMouseUp = () => {{ isMouseDown = false; }};
                
                const onMouseMove = (e) => {{
                    if (!isMouseDown) return;
                    const deltaX = e.clientX - mouseX;
                    const deltaY = e.clientY - mouseY;
                    spherical.theta -= deltaX * 0.01;
                    spherical.phi += deltaY * 0.01;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                    camera.position.setFromSpherical(spherical);
                    camera.lookAt(0, 0, 0);
                    mouseX = e.clientX;
                    mouseY = e.clientY;
                }};
                
                const onWheel = (e) => {{
                    e.preventDefault();
                    spherical.radius += e.deltaY * 0.01;
                    spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, spherical.radius));
                    camera.position.setFromSpherical(spherical);
                    camera.lookAt(0, 0, 0);
                }};
                
                domElement.addEventListener('mousedown', onMouseDown);
                domElement.addEventListener('mouseup', onMouseUp);
                domElement.addEventListener('mousemove', onMouseMove);
                domElement.addEventListener('wheel', onWheel);
                
                this.update = function() {{
                    if (this.enableDamping) {{
                        // Damping would go here
                    }}
                }};
            }}
            
            // Data
            const nodes = {json.dumps(nodes_js)};
            const edges = {json.dumps(edges_js)};
            const nodeSize = {node_size};
            const showLabels = {str(show_labels).lower()};
            
            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf5f5f5);
            
            // Camera
            const container = document.getElementById('container');
            const aspect = container.clientWidth / {plot_height};
            const camera = new THREE.PerspectiveCamera(75, aspect || 1, 0.1, 10000);
            // Position camera to view the data
            const cameraDistance = Math.max(scale * 2, 10);
            camera.position.set(
                {center_x} + cameraDistance * 0.7,
                {center_y} + cameraDistance * 0.7,
                {center_z} + cameraDistance * 0.7
            );
            camera.lookAt({center_x}, {center_y}, {center_z});
            
            // Renderer
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, {plot_height});
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            
            // Controls
            const controls = new SimpleOrbitControls(camera, renderer.domElement);
            controls.minDistance = scale * 0.5;
            controls.maxDistance = scale * 5;
            
            // Lighting - add multiple lights for better visibility
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight1.position.set(1, 1, 1);
            scene.add(directionalLight1);
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight2.position.set(-1, -1, -1);
            scene.add(directionalLight2);
            const pointLight = new THREE.PointLight(0xffffff, 0.5);
            pointLight.position.set({center_x}, {center_y}, {center_z});
            scene.add(pointLight);
            
            // Create node geometry and materials
            const nodeGeometries = {{}};
            const nodeMaterials = {{}};
            const nodeMeshes = [];
            const nodeMap = {{}};
            
            // Group nodes by paper for legend
            const papers = {{}};
            
            nodes.forEach((node, index) => {{
                // Create sphere for node - ensure minimum size
                const sphereRadius = Math.max(nodeSize * 0.05, 0.2);
                const geometry = new THREE.SphereGeometry(sphereRadius, 16, 16);
                const color = new THREE.Color(node.color);
                const material = new THREE.MeshPhongMaterial({{
                    color: color,
                    shininess: 50,
                    transparent: false,
                    opacity: 1.0
                }});
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.set(node.x, node.y, node.z);
                sphere.userData = {{ name: node.name, paper: node.paper }};
                scene.add(sphere);
                nodeMeshes.push(sphere);
                nodeMap[node.name] = sphere;
                
                // Track papers for legend
                if (!papers[node.paper]) {{
                    papers[node.paper] = {{ color: node.color, count: 0 }};
                }}
                papers[node.paper].count++;
                
                // Add label if enabled
                if (showLabels) {{
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = 256;
                    canvas.height = 64;
                    context.fillStyle = 'rgba(0, 0, 0, 0.8)';
                    context.fillRect(0, 0, canvas.width, canvas.height);
                    context.fillStyle = 'white';
                    context.font = '24px Arial';
                    context.textAlign = 'center';
                    context.textBaseline = 'middle';
                    const label = node.name.length > 20 ? node.name.substring(0, 20) + '...' : node.name;
                    context.fillText(label, canvas.width / 2, canvas.height / 2);
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
                    const sprite = new THREE.Sprite(spriteMaterial);
                    sprite.position.set(node.x, node.y + nodeSize * 0.15, node.z);
                    sprite.scale.set(2, 0.5, 1);
                    scene.add(sprite);
                }}
            }});
            
            // Create edges
            const edgeGeometry = new THREE.BufferGeometry();
            const edgePositions = [];
            const edgeColors = [];
            
            edges.forEach(edge => {{
                const sourceNode = nodeMap[edge.source];
                const targetNode = nodeMap[edge.target];
                if (sourceNode && targetNode) {{
                    edgePositions.push(
                        sourceNode.position.x, sourceNode.position.y, sourceNode.position.z,
                        targetNode.position.x, targetNode.position.y, targetNode.position.z
                    );
                    const color = edge.isCrossPaper ? 
                        new THREE.Color(0xff6666) : 
                        new THREE.Color(0x999999);
                    edgeColors.push(color.r, color.g, color.b);
                    edgeColors.push(color.r, color.g, color.b);
                }}
            }});
            
            if (edgePositions.length > 0) {{
                edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
                edgeGeometry.setAttribute('color', new THREE.Float32BufferAttribute(edgeColors, 3));
                const edgeMaterial = new THREE.LineBasicMaterial({{
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.4,
                    linewidth: edge.isCrossPaper ? 3 : 1
                }});
                const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
                scene.add(edgeLines);
            }}
            
            // Create legend
            const legendContent = document.getElementById('legend-content');
            Object.keys(papers).forEach(paperName => {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                const colorBox = document.createElement('div');
                colorBox.className = 'legend-color';
                colorBox.style.backgroundColor = papers[paperName].color;
                const text = document.createElement('span');
                text.textContent = `${{paperName}} (${{papers[paperName].count}})`;
                item.appendChild(colorBox);
                item.appendChild(text);
                legendContent.appendChild(item);
            }});
            
            // Raycaster for hover
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            let hoveredObject = null;
            
            function onMouseMove(event) {{
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(nodeMeshes);
                
                if (intersects.length > 0) {{
                    if (hoveredObject !== intersects[0].object) {{
                        if (hoveredObject) {{
                            hoveredObject.material.emissive.setHex(0x000000);
                        }}
                        hoveredObject = intersects[0].object;
                        hoveredObject.material.emissive.setHex(0x444444);
                        document.body.style.cursor = 'pointer';
                    }}
                }} else {{
                    if (hoveredObject) {{
                        hoveredObject.material.emissive.setHex(0x000000);
                        hoveredObject = null;
                    }}
                    document.body.style.cursor = 'default';
                }}
            }}
            
            function onMouseClick(event) {{
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(nodeMeshes);
                
                if (intersects.length > 0) {{
                    const node = intersects[0].object.userData;
                    alert(`Node: ${{node.name}}\\nPaper: ${{node.paper}}`);
                }}
            }}
            
            container.addEventListener('mousemove', onMouseMove);
            container.addEventListener('click', onMouseClick);
            
            // Handle window resize
            function handleResize() {{
                const width = container.clientWidth;
                const height = {plot_height};
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }}
            window.addEventListener('resize', handleResize);
            // Initial resize to ensure proper sizing
            setTimeout(handleResize, 100);
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    return html_content


def render_embedding_mode():
    """Render the embedding visualization mode."""
    from sklearn.decomposition import PCA
    import numpy as np
    
    st.header("🌐 Embedding Mode - Multi-Researcher Visualization")
    st.markdown("Generate embeddings of Level 3 nodes from multiple researchers and visualize their semantic relationships. Each researcher's nodes are colored the same.")
    
    tab1, tab2 = st.tabs(["📁 Load JSON & Generate Embeddings", "🎨 2D Visualization"])
    
    # Tab 1: Load JSON and generate embeddings
    with tab1:
        st.markdown("### Upload Researcher JSON Files")
        st.caption("Upload multiple JSON files, one for each researcher. First file = R1, second = R2, etc. Nodes from the same researcher will be colored the same.")
        
        uploaded_jsons = st.file_uploader(
            "Choose JSON files (one per researcher)",
            type=['json'],
            accept_multiple_files=True,
            key="embedding_json_upload",
            help="Upload JSON files exported from Single-Paper or Multi-Paper Mode. Each file represents one researcher."
        )
        
        if uploaded_jsons and len(uploaded_jsons) > 0:
            try:
                # Generate distinct colors for researchers
                # Use hue range 0.0 to 0.45 (red to yellow-green spectrum) to avoid conflicts with clusters
                import colorsys
                researcher_colors = {}
                for idx, uploaded_json in enumerate(uploaded_jsons):
                    researcher_id = f"R{idx + 1}"
                    # Generate distinct colors using HSV color space
                    # Map to hue range 0.0-0.45 (red to yellow-green) to separate from clusters (0.55-1.0)
                    if len(uploaded_jsons) > 1:
                        hue = 0.0 + (idx / max(len(uploaded_jsons) - 1, 1)) * 0.45
                    else:
                        hue = 0.225  # Middle of researcher range for single researcher
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    color_hex = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                    researcher_colors[researcher_id] = color_hex
                
                # Process each JSON file
                nodes_data = []  # List of {node_name, paper_id, paper_name, paper_color, researcher_id, researcher_color}
                all_papers = {}  # Track all papers across researchers
                
                # Store JSON contents to avoid re-reading files
                json_contents = []
                for uploaded_json in uploaded_jsons:
                    # Reset file pointer and read content
                    uploaded_json.seek(0)
                    json_contents.append(json.loads(uploaded_json.read().decode('utf-8')))
                
                for idx, json_content in enumerate(json_contents):
                    researcher_id = f"R{idx + 1}"
                    researcher_color = researcher_colors[researcher_id]
                    
                    # Check if it's a multi-paper JSON or single-paper JSON
                    if "papers" in json_content and isinstance(json_content.get("papers"), list):
                        # Multi-paper JSON format
                        for paper in json_content["papers"]:
                            paper_id = paper.get('id', f'unknown_{researcher_id}')
                            paper_name = paper.get('name', 'Unknown')
                            paper_color = paper.get('color', {})
                            color_hex = paper_color.get('hex', '#4A90E2') if isinstance(paper_color, dict) else '#4A90E2'
                            
                            # Store paper info
                            full_paper_id = f"{researcher_id}_{paper_id}"
                            all_papers[full_paper_id] = {
                                'paper_id': paper_id,
                                'paper_name': paper_name,
                                'paper_color': color_hex,
                                'researcher_id': researcher_id,
                                'researcher_color': researcher_color
                            }
                            
                            hlg_data = paper.get('hlg_data', {})
                            level3_nodes = hlg_data.get('Level3', [])
                            
                            for node_name in level3_nodes:
                                nodes_data.append({
                                    'node_name': node_name,
                                    'paper_id': full_paper_id,
                                    'paper_name': paper_name,
                                    'paper_color': color_hex,
                                    'researcher_id': researcher_id,
                                    'researcher_color': researcher_color
                                })
                    else:
                        # Single-paper JSON format
                        # Get filename from the original uploaded file
                        original_file = uploaded_jsons[idx]
                        paper_name = original_file.name.replace('.json', '')
                        paper_id = f'paper_{researcher_id}'
                        full_paper_id = f"{researcher_id}_{paper_id}"
                        
                        all_papers[full_paper_id] = {
                            'paper_id': paper_id,
                            'paper_name': paper_name,
                            'paper_color': researcher_color,  # Use researcher color for single papers
                            'researcher_id': researcher_id,
                            'researcher_color': researcher_color
                        }
                        
                        level3_nodes = json_content.get('Level3', [])
                        for node_name in level3_nodes:
                            nodes_data.append({
                                'node_name': node_name,
                                'paper_id': full_paper_id,
                                'paper_name': paper_name,
                                'paper_color': researcher_color,
                                'researcher_id': researcher_id,
                                'researcher_color': researcher_color
                            })
                
                if len(nodes_data) == 0:
                    st.warning("⚠️ No Level 3 nodes found in the uploaded JSON files. Make sure the papers have been analyzed.")
                else:
                    # Count nodes by researcher
                    researcher_counts = {}
                    for node in nodes_data:
                        researcher_id = node['researcher_id']
                        researcher_counts[researcher_id] = researcher_counts.get(researcher_id, 0) + 1
                    
                    st.success(f"✅ Successfully loaded {len(uploaded_jsons)} researcher file(s)")
                    st.info(f"📊 Found {len(nodes_data)} Level 3 nodes from {len(uploaded_jsons)} researcher(s)")
                    
                    # Show node breakdown by researcher
                    with st.expander("📋 Node Breakdown by Researcher"):
                        for researcher_id in sorted(researcher_counts.keys()):
                            count = researcher_counts[researcher_id]
                            color = researcher_colors[researcher_id]
                            st.markdown(
                                f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                                f'<div style="width: 20px; height: 20px; background-color: {color}; '
                                f'border-radius: 50%; border: 1px solid #ccc; margin-right: 8px;"></div>'
                                f'<span><strong>{researcher_id}</strong>: {count} Level 3 nodes</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Generate embeddings button
                        if st.button("🚀 Generate Embeddings & Create 3D Visualization"):
                            with st.spinner("Generating embeddings for Level 3 nodes..."):
                                try:
                                    # Initialize parser for embeddings
                                    parser = LLMParser(model=st.session_state.get('model', 'anthropic/claude-3.5-sonnet'))
                                    
                                    # Extract node names
                                    node_names = [node['node_name'] for node in nodes_data]
                                    
                                    # Generate embeddings
                                    embeddings = parser.generate_embeddings(
                                        texts=node_names,
                                        model="openai/text-embedding-3-small"
                                    )
                                    
                                    # Convert to numpy array
                                    embeddings_array = np.array(embeddings)
                                    
                                    # Apply PCA to reduce to 3D
                                    if embeddings_array.shape[1] > 3:
                                        pca = PCA(n_components=3)
                                        embeddings_3d = pca.fit_transform(embeddings_array)
                                        
                                        # Show explained variance
                                        explained_variance = sum(pca.explained_variance_ratio_)
                                        st.info(f"📊 PCA: {explained_variance:.1%} of variance explained by first 3 components")
                                    else:
                                        embeddings_3d = embeddings_array
                                        explained_variance = 1.0
                                    
                                    # Extract L3→L3 relations from all uploaded JSONs
                                    edges = []
                                    
                                    # Process each uploaded JSON to extract relations
                                    # Re-read JSON files for edge extraction (reset file pointers)
                                    for idx, uploaded_json in enumerate(uploaded_jsons):
                                        researcher_id = f"R{idx + 1}"
                                        uploaded_json.seek(0)  # Reset file pointer
                                        json_content = json.loads(uploaded_json.read().decode('utf-8'))
                                        
                                        # Check if it's a multi-paper JSON or single-paper JSON
                                        if "papers" in json_content and isinstance(json_content.get("papers"), list):
                                            # Multi-paper JSON format
                                            for paper in json_content["papers"]:
                                                paper_id = paper.get('id', 'unknown')
                                                full_paper_id = f"{researcher_id}_{paper_id}"
                                                hlg_data = paper.get('hlg_data', {})
                                                relations = hlg_data.get('Relations', [])
                                                level3_nodes = set(hlg_data.get('Level3', []))
                                                
                                                for rel in relations:
                                                    source = rel.get('source')
                                                    target = rel.get('target')
                                                    
                                                    # Check if both are Level 3 nodes
                                                    if source in level3_nodes and target in level3_nodes:
                                                        edges.append({
                                                            'source': source,
                                                            'target': target,
                                                            'source_paper': full_paper_id,
                                                            'target_paper': full_paper_id,
                                                            'relation': rel.get('relation', 'related-to'),
                                                            'confidence': rel.get('confidence', 5)
                                                        })
                                            
                                            # Cross-paper relations within the same researcher
                                            cross_paper_relations = json_content.get('cross_paper_relations', [])
                                            for rel in cross_paper_relations:
                                                source = rel.get('source')
                                                target = rel.get('target')
                                                source_paper = rel.get('source_paper')
                                                target_paper = rel.get('target_paper')
                                                
                                                # Check if both are Level 3 nodes
                                                source_is_l3 = False
                                                target_is_l3 = False
                                                
                                                for paper in json_content["papers"]:
                                                    if paper.get('id') == source_paper:
                                                        if source in paper.get('hlg_data', {}).get('Level3', []):
                                                            source_is_l3 = True
                                                    if paper.get('id') == target_paper:
                                                        if target in paper.get('hlg_data', {}).get('Level3', []):
                                                            target_is_l3 = True
                                                
                                                if source_is_l3 and target_is_l3:
                                                    edges.append({
                                                        'source': source,
                                                        'target': target,
                                                        'source_paper': f"{researcher_id}_{source_paper}",
                                                        'target_paper': f"{researcher_id}_{target_paper}",
                                                        'relation': rel.get('relation', 'related-to'),
                                                        'confidence': rel.get('confidence', 5)
                                                    })
                                        else:
                                            # Single-paper JSON format
                                            full_paper_id = f"{researcher_id}_paper_{researcher_id}"
                                            relations = json_content.get('Relations', [])
                                            level3_nodes = set(json_content.get('Level3', []))
                                            
                                            for rel in relations:
                                                source = rel.get('source')
                                                target = rel.get('target')
                                                
                                                # Check if both are Level 3 nodes
                                                if source in level3_nodes and target in level3_nodes:
                                                    edges.append({
                                                        'source': source,
                                                        'target': target,
                                                        'source_paper': full_paper_id,
                                                        'target_paper': full_paper_id,
                                                        'relation': rel.get('relation', 'related-to'),
                                                        'confidence': rel.get('confidence', 5)
                                                    })
                                    
                                    # Store in session state
                                    st.session_state.embedding_data = {
                                        'nodes_data': nodes_data,
                                        'embeddings_3d': embeddings_3d.tolist(),
                                        'edges': edges,
                                        'researcher_colors': researcher_colors,
                                        'all_papers': all_papers,
                                        'pca_explained_variance': explained_variance
                                    }
                                    
                                    st.success(f"✅ Generated embeddings and found {len(edges)} L3→L3 relations!")
                                    st.info("💡 Go to the '3D Visualization' tab to see the interactive plot!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error generating embeddings: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
            
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading JSON: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("👆 Upload JSON file(s) to get started. Each file represents one researcher (R1, R2, etc.)")
    
    # Tab 2: 2D Visualization
    with tab2:
        st.markdown("### 2D Embedding Visualization")
        st.caption("Interactive network graph showing semantic relationships between Level 3 nodes")
        
        if not st.session_state.embedding_data:
            st.warning("⚠️ No embedding data available. Please load a JSON file and generate embeddings in the first tab.")
        else:
            embedding_data = st.session_state.embedding_data
            
            # Validate embedding data structure
            required_keys = ['nodes_data', 'embeddings_3d', 'edges']
            missing_keys = [key for key in required_keys if key not in embedding_data]
            if missing_keys:
                st.error(f"❌ Invalid embedding data structure. Missing keys: {missing_keys}")
                st.info("💡 Please regenerate embeddings in the first tab.")
                return
            
            nodes_data = embedding_data['nodes_data']
            embeddings_3d = np.array(embedding_data['embeddings_3d'])
            edges = embedding_data['edges']
            researcher_colors = embedding_data.get('researcher_colors', {})
            all_papers = embedding_data.get('all_papers', {})
            
            # Validate data
            if not nodes_data or len(nodes_data) == 0:
                st.error("❌ No nodes found in embedding data.")
                st.info("💡 Please regenerate embeddings in the first tab.")
                return
            
            if embeddings_3d.shape[0] != len(nodes_data):
                st.error(f"❌ Mismatch: {len(nodes_data)} nodes but {embeddings_3d.shape[0]} embeddings.")
                st.info("💡 Please regenerate embeddings in the first tab.")
                return
            
            # Group nodes by researcher for coloring and filtering
            nodes_by_researcher = {}
            nodes_by_paper = {}  # Keep for compatibility
            for i, node in enumerate(nodes_data):
                if not isinstance(node, dict) or 'paper_id' not in node:
                    st.error(f"❌ Invalid node structure at index {i}: {node}")
                    return
                
                researcher_id = node.get('researcher_id', 'Unknown')
                researcher_color = node.get('researcher_color', '#4A90E2')
                paper_id = node['paper_id']
                
                # Group by researcher
                if researcher_id not in nodes_by_researcher:
                    nodes_by_researcher[researcher_id] = {
                        'indices': [],
                        'names': [],
                        'colors': [],
                        'researcher_color': researcher_color
                    }
                nodes_by_researcher[researcher_id]['indices'].append(i)
                nodes_by_researcher[researcher_id]['names'].append(node.get('node_name', f'Node_{i}'))
                nodes_by_researcher[researcher_id]['colors'].append(researcher_color)
                
                # Also group by paper for compatibility
                if paper_id not in nodes_by_paper:
                    nodes_by_paper[paper_id] = {
                        'indices': [],
                        'names': [],
                        'colors': [],
                        'paper_name': node.get('paper_name', 'Unknown'),
                        'researcher_id': researcher_id,
                        'researcher_color': researcher_color
                    }
                nodes_by_paper[paper_id]['indices'].append(i)
                nodes_by_paper[paper_id]['names'].append(node.get('node_name', f'Node_{i}'))
                nodes_by_paper[paper_id]['colors'].append(researcher_color)  # Use researcher color
            
            # Debug information (collapsible)
            with st.expander("🔍 Debug Information", expanded=False):
                # Calculate coordinate ranges
                all_x = embeddings_3d[:, 0]
                all_y = embeddings_3d[:, 1]
                all_z = embeddings_3d[:, 2]
                
                debug_info = {
                    "total_nodes": len(nodes_data),
                    "embeddings_shape": embeddings_3d.shape,
                    "total_edges": len(edges),
                    "papers_found": len(nodes_by_paper),
                    "paper_ids": list(nodes_by_paper.keys()),
                    "coordinate_ranges": {
                        "x": {"min": float(np.min(all_x)), "max": float(np.max(all_x)), "range": float(np.max(all_x) - np.min(all_x))},
                        "y": {"min": float(np.min(all_y)), "max": float(np.max(all_y)), "range": float(np.max(all_y) - np.min(all_y))},
                        "z": {"min": float(np.min(all_z)), "max": float(np.max(all_z)), "range": float(np.max(all_z) - np.min(all_z))}
                    },
                    "has_nan": bool(np.any(np.isnan(embeddings_3d))),
                    "has_inf": bool(np.any(np.isinf(embeddings_3d))),
                    "sample_node": nodes_data[0] if nodes_data else None
                }
                st.json(debug_info)
                
                # Warn if coordinates are problematic
                if debug_info["has_nan"]:
                    st.error("❌ Embeddings contain NaN values!")
                if debug_info["has_inf"]:
                    st.error("❌ Embeddings contain infinite values!")
                if debug_info["coordinate_ranges"]["x"]["range"] < 1e-6:
                    st.warning("⚠️ X coordinates have very small range - nodes may be clustered at one point")
                if debug_info["coordinate_ranges"]["y"]["range"] < 1e-6:
                    st.warning("⚠️ Y coordinates have very small range - nodes may be clustered at one point")
                if debug_info["coordinate_ranges"]["z"]["range"] < 1e-6:
                    st.warning("⚠️ Z coordinates have very small range - nodes may be clustered at one point")
            
            # Visualization controls
            st.markdown("#### 🎛️ Display Controls")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                node_size = st.slider("Node Size", min_value=10, max_value=30, value=20, step=1,
                                     help="Adjust the size of node markers")
            
            with col2:
                show_labels = st.checkbox("Show Node Names", value=False,
                                        help="Display node names as text labels")
            
            with col3:
                show_cluster_labels = st.checkbox("Show Cluster Labels", value=False,
                                                 help="Display cluster names on nodes (overrides node names)")
            
            with col4:
                show_edges = st.checkbox("Show Edges", value=True,
                                       help="Display connections between related nodes")
            
            with col5:
                min_confidence = st.slider("Min Edge Confidence", min_value=1, max_value=10, value=5, step=1,
                                          help="Only show edges with confidence >= this value")
            
            # Plot height in a new row
            col_height = st.columns(1)
            with col_height[0]:
                plot_height = st.slider("Plot Height", min_value=600, max_value=1200, value=900, step=50,
                                      help="Adjust the height of the visualization")
            
            # Researcher and Paper filtering
            st.markdown("#### 📚 Filter by Researchers/Papers")
            
            if not nodes_by_researcher:
                st.error("❌ No researchers found in the data. Please check your JSON file.")
                return
            
            # Researcher selection
            researcher_ids = sorted(nodes_by_researcher.keys())
            selected_researchers = st.multiselect(
                "👥 Select Researchers (leave empty to show all)",
                options=researcher_ids,
                default=researcher_ids,
                help="Filter visualization to show only selected researchers. Nodes from the same researcher share the same color."
            )
            
            # Paper selection (optional)
            paper_names = [nodes_by_paper[pid]['paper_name'] for pid in nodes_by_paper.keys()]
            selected_papers = st.multiselect(
                "📄 Select Papers (optional, leave empty to show all)",
                options=paper_names,
                default=[],
                help="Optionally filter by specific papers"
            )
            
            # Filter nodes based on selected researchers and papers
            if not selected_researchers and not selected_papers:
                selected_paper_ids = set(nodes_by_paper.keys())  # Show all if nothing selected
                st.info("💡 Showing all researchers and papers")
            else:
                selected_paper_ids = set()
                
                # Filter by researchers
                if selected_researchers:
                    for pid, data in nodes_by_paper.items():
                        if data.get('researcher_id') in selected_researchers:
                            selected_paper_ids.add(pid)
                else:
                    # If no researchers selected but papers are, start with all
                    selected_paper_ids = set(nodes_by_paper.keys())
                
                # Further filter by papers if specified
                if selected_papers:
                    paper_filtered = {pid for pid, data in nodes_by_paper.items() 
                                     if data['paper_name'] in selected_papers}
                    selected_paper_ids = selected_paper_ids & paper_filtered
            
            # Filter selected_paper_ids to only include papers that exist in nodes_by_paper
            valid_selected_paper_ids = {pid for pid in selected_paper_ids if pid in nodes_by_paper}
            
            if len(valid_selected_paper_ids) < len(selected_paper_ids):
                missing = selected_paper_ids - valid_selected_paper_ids
                st.warning(f"⚠️ Some selected papers not found in data: {missing}")
            
            if not valid_selected_paper_ids:
                st.warning("⚠️ No valid papers selected. Please select at least one paper.")
                return
            
            # Update selected_paper_ids to only valid ones
            selected_paper_ids = valid_selected_paper_ids
            
            # Check if we have any nodes to display after filtering
            total_visible_nodes = sum(len(nodes_by_paper[pid]['indices']) for pid in selected_paper_ids)
            if total_visible_nodes == 0:
                st.warning("⚠️ No nodes to display. Please select at least one paper.")
                return
            
            # Generate 2D NetworkX visualization
            with st.spinner("🔄 Clustering nodes and generating visualization..."):
                html_content, cluster_info = generate_2d_networkx_visualization(
                    nodes_data, embeddings_3d, edges, nodes_by_paper, 
                    selected_paper_ids, show_labels, node_size, plot_height, min_confidence, show_cluster_labels
                )
            
            # Display researcher legend
            if nodes_by_researcher:
                st.markdown("#### 👥 Researchers")
                researcher_cols = st.columns(min(len(nodes_by_researcher), 4))
                for idx, (researcher_id, data) in enumerate(sorted(nodes_by_researcher.items())):
                    col_idx = idx % 4
                    with researcher_cols[col_idx]:
                        color = data['researcher_color']
                        count = len(data['indices'])
                        st.markdown(
                            f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                            f'<div style="width: 20px; height: 20px; background-color: {color}; '
                            f'border-radius: 50%; border: 1px solid #ccc; margin-right: 8px;"></div>'
                            f'<span><strong>{researcher_id}</strong>: {count} nodes</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            
            # Display researcher legend
            if nodes_by_researcher:
                st.markdown("#### 👥 Researchers")
                researcher_cols = st.columns(min(len(nodes_by_researcher), 4))
                for idx, (researcher_id, data) in enumerate(sorted(nodes_by_researcher.items())):
                    col_idx = idx % 4
                    with researcher_cols[col_idx]:
                        color = data['researcher_color']
                        count = len(data['indices'])
                        st.markdown(
                            f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                            f'<div style="width: 20px; height: 20px; background-color: {color}; '
                            f'border-radius: 50%; border: 1px solid #ccc; margin-right: 8px;"></div>'
                            f'<span><strong>{researcher_id}</strong>: {count} nodes</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            
            # Display cluster information and legend
            if cluster_info:
                st.markdown("#### 📊 Cluster Information")
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.info(f"🔵 **{cluster_info['n_clusters']} Clusters** detected")
                with col_info2:
                    st.caption(f"Method: {cluster_info['method']}")
                with col_info3:
                    if cluster_info['silhouette_score']:
                        st.caption(f"Quality Score: {cluster_info['silhouette_score']:.2f}")
                with col_info4:
                    # Prepare JSON export data
                    export_data = {
                        'clustering_info': {
                            'n_clusters': cluster_info['n_clusters'],
                            'method': cluster_info['method'],
                            'silhouette_score': cluster_info['silhouette_score'],
                            'total_nodes': total_visible_nodes
                        },
                        'clusters': []
                    }
                    
                    # Build cluster data for export
                    if cluster_info.get('cluster_nodes'):
                        sorted_clusters = sorted(cluster_info['cluster_nodes'].items(),
                                               key=lambda x: int(x[0].split()[-1]) if x[0].split()[-1].isdigit() else 999)
                        
                        for cluster_label, nodes in sorted_clusters:
                            cluster_color = cluster_info['cluster_color_map'].get(cluster_label, '#cccccc')
                            
                            # Group nodes by paper
                            nodes_by_paper = {}
                            # Group nodes by researcher
                            nodes_by_researcher = {}
                            # Detailed node list with researcher info
                            detailed_nodes = []
                            
                            for node_info in nodes:
                                paper = node_info['paper']
                                researcher_id = node_info.get('researcher_id', 'Unknown')
                                researcher_color = node_info.get('researcher_color', '#4A90E2')
                                
                                if paper not in nodes_by_paper:
                                    nodes_by_paper[paper] = []
                                nodes_by_paper[paper].append(node_info['name'])
                                
                                if researcher_id not in nodes_by_researcher:
                                    nodes_by_researcher[researcher_id] = []
                                nodes_by_researcher[researcher_id].append(node_info['name'])
                                
                                detailed_nodes.append({
                                    'name': node_info['name'],
                                    'paper': paper,
                                    'researcher_id': researcher_id,
                                    'researcher_color': researcher_color
                                })
                            
                            cluster_data = {
                                'cluster_label': cluster_label,
                                'cluster_color': cluster_color,
                                'node_count': len(nodes),
                                'nodes': sorted([node['name'] for node in nodes]),
                                'nodes_by_paper': {paper: sorted(node_names) for paper, node_names in nodes_by_paper.items()},
                                'nodes_by_researcher': {researcher_id: sorted(node_names) for researcher_id, node_names in nodes_by_researcher.items()},
                                'detailed_nodes': sorted(detailed_nodes, key=lambda x: x['name'])
                            }
                            export_data['clusters'].append(cluster_data)
                    
                    # Export button
                    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Export Clusters JSON",
                        data=json_str,
                        file_name="cluster_analysis.json",
                        mime="application/json",
                        help="Download cluster information and node lists as JSON"
                    )
                
                # Cluster legend with counts
                st.markdown("##### 🎨 Cluster Legend")
                if cluster_info.get('cluster_counts'):
                    cluster_items = sorted(cluster_info['cluster_counts'].items(), 
                                         key=lambda x: int(x[0].split()[-1]) if x[0].split()[-1].isdigit() else 999)
                    
                    # Display clusters in columns
                    n_cols = 4
                    cluster_cols = st.columns(n_cols)
                    
                    for idx, (cluster_label, count) in enumerate(cluster_items):
                        col_idx = idx % n_cols
                        with cluster_cols[col_idx]:
                            cluster_color = cluster_info['cluster_color_map'].get(cluster_label, '#cccccc')
                            st.markdown(
                                f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                                f'<div style="width: 20px; height: 20px; background-color: {cluster_color}; '
                                f'border-radius: 50%; border: 1px solid #ccc; margin-right: 8px;"></div>'
                                f'<span><strong>{cluster_label}</strong>: {count} nodes</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                
                # Cluster node lists - detailed breakdown
                st.markdown("##### 📋 Nodes by Cluster")
                if cluster_info.get('cluster_nodes'):
                    # Sort clusters for display
                    sorted_clusters = sorted(cluster_info['cluster_nodes'].items(),
                                           key=lambda x: int(x[0].split()[-1]) if x[0].split()[-1].isdigit() else 999)
                    
                    # Create tabs for each cluster
                    cluster_tabs = st.tabs([f"{label} ({len(nodes)} nodes)" for label, nodes in sorted_clusters])
                    
                    for tab_idx, (cluster_label, nodes) in enumerate(sorted_clusters):
                        with cluster_tabs[tab_idx]:
                            cluster_color = cluster_info['cluster_color_map'].get(cluster_label, '#cccccc')
                            
                            # Group nodes by paper for better organization
                            nodes_by_paper_in_cluster = {}
                            for node_info in nodes:
                                paper = node_info['paper']
                                if paper not in nodes_by_paper_in_cluster:
                                    nodes_by_paper_in_cluster[paper] = []
                                nodes_by_paper_in_cluster[paper].append(node_info['name'])
                            
                            st.markdown(f"**Cluster Color:** <span style='color: {cluster_color}; font-weight: bold; font-size: 1.2em;'>●</span> `{cluster_color}`")
                            st.markdown(f"**Total Nodes:** {len(nodes)}")
                            st.markdown("---")
                            
                            # Display nodes grouped by paper
                            st.markdown("**Nodes grouped by paper:**")
                            for paper, node_names in sorted(nodes_by_paper_in_cluster.items()):
                                with st.expander(f"📄 {paper} ({len(node_names)} nodes)", expanded=True):
                                    for node_name in sorted(node_names):
                                        st.markdown(f"• {node_name}")
                            
                            # Also show flat list option
                            st.markdown("---")
                            st.markdown("**All nodes in this cluster (alphabetical):**")
                            all_node_names = sorted([node['name'] for node in nodes])
                            # Display in columns for better readability
                            n_cols_display = 2
                            name_cols = st.columns(n_cols_display)
                            items_per_col = (len(all_node_names) + n_cols_display - 1) // n_cols_display
                            
                            for col_idx in range(n_cols_display):
                                with name_cols[col_idx]:
                                    start_idx = col_idx * items_per_col
                                    end_idx = min((col_idx + 1) * items_per_col, len(all_node_names))
                                    for node_name in all_node_names[start_idx:end_idx]:
                                        st.markdown(f"• {node_name}")
            
            # Export HTML button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("#### 🎨 2D Network Visualization with Clustering")
                st.caption("Nodes are colored by automatically detected semantic clusters. Node borders show paper colors. Positioned using t-SNE for better cluster separation.")
            with col2:
                st.download_button(
                    label="💾 Export HTML",
                    data=html_content,
                    file_name="2d_embedding_visualization.html",
                    mime="text/html",
                    help="Download the visualization as an HTML file to view in your browser"
                )
            
            # Display the visualization
            components.html(html_content, height=plot_height + 50, scrolling=False)
            
            # Track visible nodes for statistics (with safety check)
            visible_node_count = sum(len(nodes_by_paper[pid]['indices']) 
                                   for pid in selected_paper_ids 
                                   if pid in nodes_by_paper)
            
            # Count visible edges (with safety check)
            visible_node_names = set()
            for paper_id in selected_paper_ids:
                if paper_id in nodes_by_paper:
                    visible_node_names.update(nodes_by_paper[paper_id]['names'])
            
            edge_count = 0
            if show_edges:
                for edge in edges:
                    if (edge.get('source') in visible_node_names and 
                        edge.get('target') in visible_node_names):
                        edge_count += 1
            
            # Skip the rest of the old Plotly code
            return_early = True
            if return_early:
                # Show statistics
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Visible Nodes", visible_node_count, delta=f"{len(nodes_data) - visible_node_count} hidden" if visible_node_count < len(nodes_data) else None)
                with col2:
                    st.metric("Visible Edges", edge_count)
                with col3:
                    st.metric("Papers Shown", len(selected_paper_ids), delta=f"{len(nodes_by_paper) - len(selected_paper_ids)} hidden" if len(selected_paper_ids) < len(nodes_by_paper) else None)
                with col4:
                    st.metric("Variance Explained", f"{embedding_data['pca_explained_variance']:.1%}")
                with col5:
                    st.metric("Node Size", node_size)
                return


def render_multi_paper_mode():
    """Render the multi-paper analysis mode."""
    # Define paper colors for visualization
    PAPER_COLORS = [
        {"name": "Blue", "hex": "#4A90E2", "l3": "#4A90E2", "l2p": "#9B59B6", "l2m": "#50C878", "l1": "#9B9B9B"},
        {"name": "Red", "hex": "#E74C3C", "l3": "#E74C3C", "l2p": "#E67E22", "l2m": "#F39C12", "l1": "#BDC3C7"},
        {"name": "Green", "hex": "#27AE60", "l3": "#27AE60", "l2p": "#16A085", "l2m": "#1ABC9C", "l1": "#95A5A6"},
        {"name": "Purple", "hex": "#8E44AD", "l3": "#8E44AD", "l2p": "#9B59B6", "l2m": "#BB79C6", "l1": "#A9AEAF"},
        {"name": "Orange", "hex": "#D35400", "l3": "#D35400", "l2p": "#E67E22", "l2m": "#F39C12", "l1": "#BDC3C7"}
    ]
    
    st.header("📚 Multi-Paper Comparison Mode")
    st.markdown("Upload and compare multiple research papers to find cross-paper relations and insights.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Load JSON", "📄 Upload Papers", "🔍 Analysis", "🎨 Visualization", "💾 Export"])
    
    # Tab 1: Load JSON
    with tab1:
        st.header("📁 Load Multi-Paper Analysis from JSON")
        
        st.markdown("""
        **Quick Start: Upload Previously Exported Multi-Paper Analysis**
        
        Skip the PDF extraction and LLM analysis steps by uploading a JSON file that was exported from a previous multi-paper analysis.
        
        **Benefits:**
        - ✅ Instant visualization without LLM API costs
        - 📊 Share analysis results with collaborators
        - 💾 Review past comparative analyses offline
        - ⚡ No waiting for expensive multi-paper API calls
        - 🎨 Includes all papers and cross-paper relations
        
        **Compatible Files:**
        - Files exported from 'Export' tab → 'Download Multi-Paper Analysis JSON'
        - Any valid multi-paper analysis JSON with 'papers' array structure
        """)
        
        st.markdown("---")
        
        uploaded_json = st.file_uploader(
            "Choose a multi-paper JSON file",
            type=['json'],
            key="multi_json_upload_tab",
            help="Upload a multi-paper analysis JSON file exported from a previous analysis"
        )
        
        if uploaded_json is not None:
            try:
                json_content = json.loads(uploaded_json.read().decode('utf-8'))
                
                # Validate multi-paper structure
                if "papers" in json_content and isinstance(json_content.get("papers"), list):
                    # Ensure papers have color info
                    for idx, paper in enumerate(json_content["papers"]):
                        if "color" not in paper or not paper["color"]:
                            paper["color"] = generate_paper_color(idx, PAPER_COLORS)
                    
                    st.session_state.multi_graph_data = json_content
                    st.session_state.multi_papers = json_content["papers"]
                    st.session_state.multi_processing_complete = True
                    
                    st.success(f"✅ Successfully loaded: {uploaded_json.name}")
                    
                    # Show preview of loaded data
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Papers Loaded", len(json_content["papers"]))
                    with col2:
                        total_relations = sum(
                            len(p.get('hlg_data', {}).get('Relations', []))
                            for p in json_content["papers"]
                        )
                        st.metric("Intra-Paper Relations", total_relations)
                    with col3:
                        cross_relations = len(json_content.get('cross_paper_relations', []))
                        st.metric("Cross-Paper Relations", cross_relations)
                    
                    st.info("💡 Go to the 'Visualization' tab to see the interactive multi-paper graph!")
                    
                    # Show papers list
                    st.markdown("### 📚 Loaded Papers:")
                    for idx, paper in enumerate(json_content["papers"]):
                        color = paper.get('color', PAPER_COLORS[0])
                        st.markdown(f"**{idx+1}.** {paper.get('name', 'Unknown')} - <span style='background-color: {color['hex']}; color: white; padding: 2px 8px; border-radius: 4px;'>{color['name']}</span>", unsafe_allow_html=True)
                    
                    # Show quick preview
                    with st.expander("👁️ Preview Loaded Data"):
                        # Don't show full text to keep it manageable
                        preview_data = json_content.copy()
                        for paper in preview_data.get("papers", []):
                            if "text" in paper:
                                paper["text"] = f"{paper['text'][:200]}... (truncated)"
                        st.json(preview_data)
                else:
                    st.error("❌ Invalid multi-paper JSON format. Missing 'papers' array")
                    st.info("💡 Make sure you're uploading a file exported from this application's 'Export' tab in Multi-Paper mode.")
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading JSON: {str(e)}")
        else:
            st.info("👆 Upload a multi-paper JSON file to get started, or go to 'Upload Papers' tab to analyze new papers")
    
    # Tab 2: Upload Papers
    with tab2:
        st.header("📄 Upload Research Papers")
        st.markdown("Upload 2-5 papers for comparative analysis")
        st.caption("Start from scratch by uploading PDFs and running multi-paper LLM analysis")
        
        # File uploader for multiple files
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload 2-5 research papers in PDF format"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Display uploaded papers
            st.subheader("📋 Uploaded Papers")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                color = generate_paper_color(idx, PAPER_COLORS)
                
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"**Paper {idx+1}:** {uploaded_file.name}")
                    st.markdown(f"<span style='background-color: {color['hex']}; color: white; padding: 2px 8px; border-radius: 4px;'>{color['name']}</span>", unsafe_allow_html=True)
                
                with col2:
                    # Check if already extracted
                    existing_paper = next((p for p in st.session_state.multi_papers if p['name'] == uploaded_file.name), None)
                    if existing_paper and existing_paper.get('text'):
                        st.success(f"✓ Extracted ({len(existing_paper['text'])} chars)")
                    else:
                        st.info("⏳ Not extracted")
                
                with col3:
                    if st.button(f"🗑️", key=f"remove_{idx}"):
                        st.session_state.multi_papers = [p for p in st.session_state.multi_papers if p['name'] != uploaded_file.name]
                        st.experimental_rerun()
            
            st.markdown("---")
            
            # Extract all papers button
            if st.button("🔍 Extract Text from All Papers"):
                with st.spinner("Extracting text from all papers..."):
                    try:
                        st.session_state.multi_papers = []
                        
                        for idx, uploaded_file in enumerate(uploaded_files):
                            color = generate_paper_color(idx, PAPER_COLORS)
                            
                            # Extract text
                            pdf_bytes = uploaded_file.read()
                            text, full_length = extract_text_from_pdf_bytes(
                                pdf_bytes,
                                section_aware=True,
                                max_chars=st.session_state.max_chars
                            )
                            
                            # Store paper data
                            paper_data = {
                                'id': f"paper_{idx+1}",
                                'name': uploaded_file.name,
                                'text': text,
                                'full_text_length': full_length,
                                'color': color,
                                'hlg_data': None
                            }
                            st.session_state.multi_papers.append(paper_data)
                            
                            # Reset to beginning for next iteration
                            uploaded_file.seek(0)
                        
                        st.success(f"✅ Extracted text from {len(st.session_state.multi_papers)} papers!")
                        
                        # Show preview
                        with st.expander("📖 Preview Extracted Texts"):
                            for paper in st.session_state.multi_papers:
                                extracted_len = len(paper['text'])
                                full_len = paper.get('full_text_length', extracted_len)
                                if full_len > extracted_len:
                                    st.markdown(f"**{paper['name']}** - Extracted: {extracted_len:,} chars (from {full_len:,} total)")
                                else:
                                    st.markdown(f"**{paper['name']}** - {extracted_len:,} chars (full paper)")
                                st.text(paper['text'][:500] + "..." if len(paper['text']) > 500 else paper['text'])
                                st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error extracting text: {str(e)}")
        
        else:
            st.info("👆 Please upload 2 or more PDF files to begin")
        
        # Display current papers in session
        if st.session_state.multi_papers:
            st.markdown("---")
            st.subheader("✅ Papers Ready for Analysis")
            for paper in st.session_state.multi_papers:
                st.markdown(f"- **{paper['name']}** ({len(paper['text'])} chars) - {paper['color']['name']}")
    
    # Tab 3: Analysis
    with tab3:
        st.header("🔍 Multi-Paper Analysis")
        
        if not st.session_state.multi_papers or len(st.session_state.multi_papers) < 2:
            st.warning("⚠️ Please upload and extract text from at least 2 papers (Tab: Upload Papers)")
        else:
            st.markdown(f"""
            <div class="info-box">
            <strong>Ready for Analysis</strong><br>
            Papers: {len(st.session_state.multi_papers)}<br>
            Model: {st.session_state.model}<br>
            Pass 3 (Inference): {"✅ Enabled" if st.session_state.get('enable_inference', False) else "❌ Disabled"}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Analyze All Papers & Find Cross-Paper Relations"):
                with st.spinner("🧠 Analyzing papers... This may take a few minutes..."):
                    try:
                        parser = LLMParser(model=st.session_state.model)
                        
                        # Phase 1: Analyze each paper individually
                        st.markdown("### Phase 1: Analyzing Individual Papers")
                        progress_bar = st.progress(0)
                        
                        for idx, paper in enumerate(st.session_state.multi_papers):
                            st.write(f"📄 Analyzing {paper['name']}...")
                            
                            hlg_data = parser.parse_paper(
                                paper['text'],
                                max_chars=st.session_state.max_chars,
                                enable_inference=st.session_state.get('enable_inference', False)
                            )
                            
                            paper['hlg_data'] = hlg_data
                            progress_bar.progress((idx + 1) / len(st.session_state.multi_papers))
                        
                        st.success("✅ Phase 1 Complete: All papers analyzed individually")
                        
                        # Phase 2: Find cross-paper relations
                        st.markdown("### Phase 2: Finding Cross-Paper Relations")
                        
                        cross_paper_data = parser.find_cross_paper_relations(
                            st.session_state.multi_papers
                        )
                        
                        st.session_state.multi_graph_data = {
                            'papers': st.session_state.multi_papers,
                            'cross_paper_relations': cross_paper_data.get('relations', []),
                            'cross_paper_confidence': cross_paper_data.get('overall_confidence', 'N/A'),
                            'cross_paper_explanation': cross_paper_data.get('overall_explanation', ''),
                            '_cross_paper_token_usage': cross_paper_data.get('_token_usage', {})
                        }
                        
                        st.session_state.multi_processing_complete = True
                        st.success("✅ Phase 2 Complete: Cross-paper relations identified!")
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        # Display per-paper relations in separate blocks
                        for paper in st.session_state.multi_papers:
                            hlg = paper['hlg_data']
                            paper_color = paper['color']['hex']
                            
                            st.markdown("---")
                            
                            # Paper header with color
                            st.markdown(f"""
                            <div style="background-color: {paper_color}; color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                                <h3 style="margin: 0;">📄 {paper['name']}</h3>
                                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">Intra-Paper Relations</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Level 3 Nodes", len(hlg.get('Level3', [])))
                            with col2:
                                st.metric("Level 2 Nodes", len(hlg.get('Level2', [])))
                            with col3:
                                st.metric("Level 1 Nodes", len(hlg.get('Level1', [])))
                            with col4:
                                st.metric("Relations", len(hlg.get('Relations', [])))
                            
                            # Display relations with confidence scores and explanations
                            relations = hlg.get('Relations', [])
                            if relations:
                                st.markdown(f"#### 🔗 {len(relations)} Relations Found")
                                
                                for i, rel in enumerate(relations, 1):
                                    confidence = rel.get('confidence', 'N/A')
                                    explanation = rel.get('explanation', '')
                                    source = rel.get('source', '')
                                    target = rel.get('target', '')
                                    
                                    # Get level badges for source and target
                                    source_badge = get_node_level_badge(source, hlg)
                                    target_badge = get_node_level_badge(target, hlg)
                                    confidence_badge = render_confidence_badge(confidence)
                                    
                                    st.markdown(f"""
                                    **[{i}]** {source_badge} **{source}** → *{rel.get('relation')}* → {target_badge} **{target}** {confidence_badge}
                                    """, unsafe_allow_html=True)
                                    
                                    if explanation:
                                        st.caption(f"💡 {explanation}")
                                    
                                    st.markdown("")  # Spacing
                            else:
                                st.info("No intra-paper relations found.")
                            
                            # Show token usage in expander
                            if "_token_usage" in hlg:
                                with st.expander(f"🔢 Token Usage for {paper['name'][:30]}..."):
                                    usage = hlg["_token_usage"]
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Prompt", f"{usage.get('prompt_tokens', 0):,}")
                                    col2.metric("Completion", f"{usage.get('completion_tokens', 0):,}")
                                    col3.metric("Total", f"{usage.get('total_tokens', 0):,}")
                        
                        # Cross-paper relations (separate block)
                        st.markdown("---")
                        
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                            <h3 style="margin: 0;">🔗 Cross-Paper Relations</h3>
                            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">Relations between concepts from different papers</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        cross_paper_relations = st.session_state.multi_graph_data.get('cross_paper_relations', [])
                        
                        if cross_paper_relations:
                            # Overall confidence
                            st.markdown(f"""
                            <div class="info-box">
                            <strong>🎯 Cross-Paper Analysis Confidence: {st.session_state.multi_graph_data['cross_paper_confidence']}/10</strong><br>
                            {st.session_state.multi_graph_data['cross_paper_explanation']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"#### 🔗 {len(cross_paper_relations)} Cross-Paper Relations Found")
                            
                            for i, rel in enumerate(cross_paper_relations, 1):
                                confidence = rel.get('confidence', 'N/A')
                                explanation = rel.get('explanation', '')
                                source_paper_id = rel.get('source_paper', 'Unknown')
                                target_paper_id = rel.get('target_paper', 'Unknown')
                                source = rel.get('source', '')
                                target = rel.get('target', '')
                                
                                # Find paper colors and names
                                source_paper_data = next((p for p in st.session_state.multi_papers if p['id'] == source_paper_id), None)
                                target_paper_data = next((p for p in st.session_state.multi_papers if p['id'] == target_paper_id), None)
                                
                                source_color = source_paper_data['color']['hex'] if source_paper_data else '#666'
                                target_color = target_paper_data['color']['hex'] if target_paper_data else '#666'
                                source_name = source_paper_data['name'][:25] if source_paper_data else source_paper_id
                                target_name = target_paper_data['name'][:25] if target_paper_data else target_paper_id
                                
                                # Get level badges from respective papers
                                source_level_badge = get_node_level_badge(source, source_paper_data['hlg_data']) if source_paper_data else ''
                                target_level_badge = get_node_level_badge(target, target_paper_data['hlg_data']) if target_paper_data else ''
                                
                                confidence_badge = render_confidence_badge(confidence)
                                
                                st.markdown(f"""
                                **[{i}]** <span style="background-color: {source_color}; color: white; padding: 2px 6px; border-radius: 3px;" title="{source_paper_data['name'] if source_paper_data else source_paper_id}">{source_name}</span> 
                                {source_level_badge} **{source}** → *{rel.get('relation')}* → 
                                <span style="background-color: {target_color}; color: white; padding: 2px 6px; border-radius: 3px;" title="{target_paper_data['name'] if target_paper_data else target_paper_id}">{target_name}</span> 
                                {target_level_badge} **{target}** {confidence_badge}
                                """, unsafe_allow_html=True)
                                
                                if explanation:
                                    st.caption(f"💡 {explanation}")
                                
                                st.markdown("")  # Spacing
                        else:
                            st.info("ℹ️ No cross-paper relations found. This may indicate that the papers cover different topics or use different terminology.")
                        
                        # Overall Token usage summary
                        st.markdown("---")
                        with st.expander("🔢 Overall Token Usage Summary"):
                            st.markdown("#### Per-Paper Analysis (Phase 1)")
                            for paper in st.session_state.multi_papers:
                                if "_token_usage" in paper['hlg_data']:
                                    usage = paper['hlg_data']['_token_usage']
                                    st.markdown(f"**{paper['name'][:40]}...**: {usage.get('total_tokens', 0):,} tokens")
                            
                            st.markdown("#### Cross-Paper Analysis (Phase 2)")
                            if '_cross_paper_token_usage' in st.session_state.multi_graph_data:
                                usage = st.session_state.multi_graph_data['_cross_paper_token_usage']
                                st.markdown(f"**Cross-Paper Relations**: {usage.get('total_tokens', 0):,} tokens")
                            
                            st.markdown("#### Grand Total")
                            total_tokens = sum(
                                p['hlg_data'].get('_token_usage', {}).get('total_tokens', 0) 
                                for p in st.session_state.multi_papers
                            )
                            if '_cross_paper_token_usage' in st.session_state.multi_graph_data:
                                total_tokens += st.session_state.multi_graph_data['_cross_paper_token_usage'].get('total_tokens', 0)
                            st.markdown(f"**🎯 Total Tokens Used**: {total_tokens:,}")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        with st.expander("🐛 Error Details"):
                            st.code(traceback.format_exc())
    
    # Tab 4: Visualization
    with tab4:
        st.header("🎨 Multi-Paper Graph Visualization")
        
        # Show status and clear button
        col_status, col_clear = st.columns([3, 1])
        with col_status:
            if st.session_state.multi_graph_data:
                st.success(f"✅ Multi-paper analysis data loaded - Visualization ready! ({len(st.session_state.multi_papers)} papers)")
        with col_clear:
            if st.session_state.multi_graph_data:
                if st.button("🗑️ Clear Data", key="clear_multi_viz_data", help="Clear current multi-paper analysis data"):
                    st.session_state.multi_graph_data = None
                    st.session_state.multi_graph_html = None
                    st.session_state.multi_papers = []
                    st.session_state.multi_processing_complete = False
                    st.experimental_rerun()
        
        if not st.session_state.multi_graph_data:
            st.warning("⚠️ No multi-paper analysis data available")
            
            # Show helpful status
            if st.session_state.multi_papers:
                st.info(f"💡 {len(st.session_state.multi_papers)} paper(s) ready but not analyzed yet. Go to 'Analysis' tab and click 'Analyze All Papers'")
            else:
                st.info("💡 No data yet. Go to 'Load JSON' tab to upload a previous analysis, or 'Upload Papers' tab to start a new analysis")
        else:
            try:
                # Layout selection
                layout_option = st.radio(
                    "📐 Layout Style:",
                    ["Linear (Horizontal Lines)", "Circular (Concentric Circles)"],
                    horizontal=True,
                    help="Choose how nodes are arranged: Linear = horizontal lines by level, Circular = concentric circles"
                )
                layout = "circular" if layout_option == "Circular (Concentric Circles)" else "linear"
                
                st.markdown("---")
                
                # Paper Filter Toggles
                st.markdown("### 🔘 Paper Filters")
                st.caption("Toggle papers on/off to focus on specific papers or see the complete graph")
                
                active_papers = []
                
                for idx, paper in enumerate(st.session_state.multi_papers):
                    # Create a row with checkbox and colored block containing paper name
                    row_cols = st.columns([1, 11])
                    
                    with row_cols[0]:
                        # Use paper name as part of key to make it unique
                        is_active = st.checkbox(
                            "",
                            value=True,  # All papers active by default
                            key=f"filter_{paper['id']}",
                            help=f"Show/hide {paper['name']}"
                        )
                    
                    with row_cols[1]:
                        # Display paper name inside colored block
                        st.markdown(f"""
                        <div style="background-color: {paper['color']['hex']}; color: white; padding: 8px; border-radius: 5px; text-align: left;">
                        <strong>📄 {paper['name']}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if is_active:
                        active_papers.append(paper['id'])
                
                st.markdown("---")
                
                # Validate at least one paper is active
                if not active_papers:
                    st.warning("⚠️ Please select at least one paper to visualize")
                else:
                    # Filter graph data based on active papers
                    filtered_graph_data = {
                        'papers': [p for p in st.session_state.multi_graph_data['papers'] if p['id'] in active_papers],
                        'cross_paper_relations': [
                            rel for rel in st.session_state.multi_graph_data.get('cross_paper_relations', [])
                            if rel.get('source_paper') in active_papers and rel.get('target_paper') in active_papers
                        ]
                    }
                    
                    # Build filtered multi-paper graph
                    builder = GraphBuilder(layout=layout)
                    G = builder.build_multi_paper_graph(filtered_graph_data)
                    
                    # Display statistics
                    stats = builder.get_statistics()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Nodes", stats['total_nodes'])
                    col2.metric("Total Edges", stats['total_edges'])
                    col3.metric("Active Papers", len(active_papers))
                    col4.metric("Cross-Paper Relations", len(filtered_graph_data.get('cross_paper_relations', [])))
                    
                    st.markdown("---")
                    
                    # Generate visualization with filtered data
                    with st.spinner("Generating interactive graph..."):
                        net = builder.to_pyvis(height="700px", width="100%")
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                            net.save_graph(f.name)
                            st.session_state.multi_graph_html = f.name
                            
                            # Read and display
                            with open(f.name, 'r', encoding='utf-8') as html_file:
                                html_content = html_file.read()
                                components.html(html_content, height=750, scrolling=True)
                    
                    # Dynamic tip based on number of active papers
                    if len(active_papers) == 1:
                        st.info("💡 **Tip**: Viewing single paper mode. Toggle other papers on to see cross-paper relations!")
                    else:
                        st.info("💡 **Tip**: Papers are color-coded! Cross-paper relations shown with thick magenta dashed lines. Toggle papers on/off to focus. You can drag nodes and zoom in/out!")
                
            except Exception as e:
                st.error(f"❌ Error creating visualization: {str(e)}")
                import traceback
                with st.expander("🐛 Error Details"):
                    st.code(traceback.format_exc())
    
    # Tab 5: Export
    with tab5:
        st.header("💾 Export Multi-Paper Results")
        
        # Show status at top of tab
        if st.session_state.multi_graph_data:
            st.success(f"✅ Multi-paper analysis data loaded - Export options ready! ({len(st.session_state.multi_papers)} papers)")
        
        if not st.session_state.multi_graph_data:
            st.warning("⚠️ No multi-paper analysis data available to export")
            
            # Show helpful status
            if st.session_state.multi_papers:
                st.info(f"💡 {len(st.session_state.multi_papers)} paper(s) ready but not analyzed yet. Go to 'Analysis' tab and click 'Analyze All Papers'")
            else:
                st.info("💡 No data yet. Go to 'Load JSON' tab to upload a previous analysis, or 'Upload Papers' tab to start a new analysis")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Download JSON")
                json_str = json.dumps(st.session_state.multi_graph_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Download Multi-Paper Analysis JSON",
                    data=json_str,
                    file_name="multi_paper_analysis.json",
                    mime="application/json"
                )
                
                with st.expander("👁️ Preview JSON"):
                    st.json(st.session_state.multi_graph_data)
            
            with col2:
                st.subheader("🌐 Download Graph HTML")
                
                # Layout selection for HTML export
                export_layout_option = st.radio(
                    "📐 Layout for Export:",
                    ["Linear (Horizontal Lines)", "Circular (Concentric Circles)"],
                    horizontal=True,
                    help="Choose layout style for the exported HTML graph"
                )
                export_layout = "circular" if export_layout_option == "Circular (Concentric Circles)" else "linear"
                
                # Generate HTML with selected layout
                try:
                    # Filter graph data based on active papers (use all papers for export)
                    filtered_graph_data = {
                        'papers': st.session_state.multi_graph_data['papers'],
                        'cross_paper_relations': st.session_state.multi_graph_data.get('cross_paper_relations', [])
                    }
                    
                    builder = GraphBuilder(layout=export_layout)
                    builder.build_multi_paper_graph(filtered_graph_data)
                    
                    # Generate HTML
                    net = builder.to_pyvis(height="700px", width="100%")
                    
                    # Save to temporary file to get HTML content
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
                    temp_path = temp_file.name
                    temp_file.close()  # Close immediately after getting path
                    
                    net.save_graph(temp_path)
                    
                    # Read the HTML content
                    with open(temp_path, 'r', encoding='utf-8') as html_file:
                        html_content = html_file.read()
                    
                    # Clean up temp file (close file first, then delete)
                    try:
                        os.unlink(temp_path)
                    except:
                        pass  # Ignore errors if file is still locked
                    
                    layout_suffix = "circular" if export_layout == "circular" else "linear"
                    st.download_button(
                        label=f"⬇️ Download Interactive Graph ({export_layout_option})",
                        data=html_content,
                        file_name=f"multi_paper_graph_{layout_suffix}.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating HTML: {str(e)}")
                    st.info("📊 Make sure analysis data is available")


if __name__ == "__main__":
    main()

