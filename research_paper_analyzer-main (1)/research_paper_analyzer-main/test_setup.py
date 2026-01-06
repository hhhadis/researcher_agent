"""
Test script to verify all modules can be imported correctly
"""

import sys

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import streamlit
        print("[OK] streamlit")
    except Exception as e:
        print(f"[FAIL] streamlit: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("[OK] openai")
    except Exception as e:
        print(f"[FAIL] openai: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("[OK] PyMuPDF (fitz)")
    except Exception as e:
        print(f"[FAIL] PyMuPDF: {e}")
        return False
    
    try:
        import networkx
        print("[OK] networkx")
    except Exception as e:
        print(f"[FAIL] networkx: {e}")
        return False
    
    try:
        from pyvis.network import Network
        print("[OK] pyvis")
    except Exception as e:
        print(f"[FAIL] pyvis: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("[OK] python-dotenv")
    except Exception as e:
        print(f"[FAIL] python-dotenv: {e}")
        return False
    
    try:
        from pdfminer.high_level import extract_text
        print("[OK] pdfminer.six")
    except Exception as e:
        print(f"[FAIL] pdfminer.six: {e}")
        return False
    
    return True


def test_custom_modules():
    """Test custom modules."""
    print("\nTesting custom modules...")
    
    try:
        import pdf_extractor
        print("[OK] pdf_extractor")
    except Exception as e:
        print(f"[FAIL] pdf_extractor: {e}")
        return False
    
    try:
        import llm_parser
        print("[OK] llm_parser")
    except Exception as e:
        print(f"[FAIL] llm_parser: {e}")
        return False
    
    try:
        import graph_builder
        print("[OK] graph_builder")
    except Exception as e:
        print(f"[FAIL] graph_builder: {e}")
        return False
    
    return True


def test_graph_builder():
    """Test graph builder with example data."""
    print("\nTesting graph builder functionality...")
    
    try:
        from graph_builder import GraphBuilder
        import json
        
        # Load example data
        with open("examples/BETag_output.json", "r", encoding="utf-8") as f:
            example_data = json.load(f)
        
        # Build graph
        builder = GraphBuilder()
        G = builder.build_graph(example_data)
        
        stats = builder.get_statistics()
        print(f"[OK] Graph built successfully")
        print(f"  - Total nodes: {stats['total_nodes']}")
        print(f"  - Total edges: {stats['total_edges']}")
        print(f"  - Level 3 nodes: {stats['level3_nodes']}")
        print(f"  - Level 2 problem nodes: {stats['level2_problem_nodes']}")
        print(f"  - Level 2 method nodes: {stats['level2_method_nodes']}")
        print(f"  - Level 1 nodes: {stats['level1_nodes']}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Graph builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_env_file():
    """Check if .env file exists."""
    print("\nChecking .env file...")
    
    import os
    if os.path.exists(".env"):
        print("[OK] .env file exists")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            print(f"[OK] OPENROUTER_API_KEY is set (length: {len(api_key)})")
        else:
            print("[FAIL] OPENROUTER_API_KEY is not set in .env")
            return False
        
        return True
    else:
        print("[FAIL] .env file not found")
        print("  Please create a .env file with:")
        print("  OPENROUTER_API_KEY=your_api_key_here")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Research Logic Graph Extractor - Setup Test")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test custom modules
    if not test_custom_modules():
        all_passed = False
    
    # Test graph builder
    if not test_graph_builder():
        all_passed = False
    
    # Check env file
    if not check_env_file():
        all_passed = False
        print("\n⚠️  Warning: .env file is missing or incomplete.")
        print("   The application will not work without a valid API key.")
        print("   Please create a .env file before running the app.")
    
    print()
    print("=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Setup is complete.")
        print()
        print("To run the application:")
        print("  1. Ensure .env file has your OPENROUTER_API_KEY")
        print("  2. Run: venv\\Scripts\\activate")
        print("  3. Run: streamlit run app.py")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    print("=" * 60)

