"""
Quick test script to diagnose OpenRouter API connection issues
"""

import os
from dotenv import load_dotenv
from llm_parser import LLMParser

# Load environment variables
load_dotenv()

def main():
    print("=" * 70)
    print("OpenRouter API Connection Test")
    print("=" * 70)
    print()
    
    # Check if API key exists
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[X] ERROR: OPENROUTER_API_KEY not found in environment!")
        print("  Please check your .env file")
        return
    
    print(f"[OK] API Key found (first 10 chars): {api_key[:10]}...")
    print()
    
    # Test different models
    models_to_test = [
        "anthropic/claude-3.5-sonnet",  # Current model
        "anthropic/claude-3-haiku",     # Fallback option
        "openai/gpt-4o-mini",           # Alternative
    ]
    
    for model in models_to_test:
        print(f"\n{'=' * 70}")
        print(f"Testing model: {model}")
        print('=' * 70)
        
        try:
            parser = LLMParser(model=model)
            success = parser.test_api_connection()
            
            if success:
                print(f"\n[OK] Model {model} is working!\n")
                print("You can use this model in your .env file:")
                print(f'  OPENROUTER_MODEL="{model}"')
                break
            else:
                print(f"\n[FAIL] Model {model} failed\n")
                
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize parser with {model}: {str(e)}\n")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()

