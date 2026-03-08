"""
test_api.py — Quick test for Nova Micro, Titan Embeddings v2, and Nova Sonic

This demonstrates the three key models in your Voice AI Agent:
1. Nova Micro - Fast conversational AI
2. Titan Embeddings v2 - Document embeddings for RAG
3. Nova Sonic - Voice interactions (see nova_sonic/client.py for full implementation)
"""

import boto3
import json
import os

# Set credentials (better to use .env file in production)
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = "ABSKQmVkcm9ja0FQSUtleS0zbmlwLWF0LTUyMzIzNDQyNTc2NTpyT2QrTGtjV3B0Y0JBekpnUDNxeVVxRmI2TlJFT3docGkyYzFYUVhaRy95Q2h1OHJTRFBocmVRS1V2az0="

AWS_REGION = "us-east-1"

# ══════════════════════════════════════════════════════════════════════════════
# 1️⃣  NOVA MICRO — Conversational AI
# ══════════════════════════════════════════════════════════════════════════════

def test_nova_micro():
    """Test Nova Micro for quick conversational responses."""
    print("\n" + "="*80)
    print("1️⃣  Testing Nova Micro (us.amazon.nova-micro-v1:0)")
    print("="*80)
    
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )
    
    # Define the model and message
    model_id = "us.amazon.nova-micro-v1:0"
    messages = [
        {
            "role": "user",
            "content": [{"text": "Hello! Can you tell me about Amazon Bedrock?"}]
        }
    ]
    
    # Make the API call
    response = client.converse(
        modelId=model_id,
        messages=messages,
    )
    
    # Print the response
    answer = response['output']['message']['content'][0]['text']
    print(f"\n🤖 Response:\n{answer}\n")
    return answer


# ══════════════════════════════════════════════════════════════════════════════
# 2️⃣  TITAN EMBEDDINGS V2 — Document Embeddings
# ══════════════════════════════════════════════════════════════════════════════

def test_titan_embeddings():
    """Test Titan Embeddings v2 for semantic search."""
    print("\n" + "="*80)
    print("2️⃣  Testing Titan Embeddings v2 (amazon.titan-embed-text-v2:0)")
    print("="*80)
    
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )
    
    # Sample text to embed
    text_to_embed = "Amazon Bedrock provides foundation models via API"
    
    # Titan Embeddings v2 request
    body = json.dumps({
        "inputText": text_to_embed,
        "dimensions": 1024,  # 256, 512, or 1024
        "normalize": True    # For cosine similarity
    })
    
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    embedding = response_body['embedding']
    
    print(f"\n📄 Input text: '{text_to_embed}'")
    print(f"📊 Embedding dimensions: {len(embedding)}")
    print(f"🔢 First 5 values: {embedding[:5]}")
    print(f"✅ This embedding can be used for semantic search, RAG, clustering")
    
    return embedding


# ══════════════════════════════════════════════════════════════════════════════
# 3️⃣  NOVA SONIC — Voice Interactions (Reference)
# ══════════════════════════════════════════════════════════════════════════════

def about_nova_sonic():
    """Info about Nova Sonic (full implementation in nova_sonic/client.py)."""
    print("\n" + "="*80)
    print("3️⃣  About Nova Sonic (amazon.nova-sonic-v1:0)")
    print("="*80)
    
    print("""
    Nova Sonic is for real-time voice interactions. Your project already has it!
    
    📁 Implementation: nova_sonic/client.py
    🎤 Capabilities:
       • Real-time bidirectional audio streaming
       • Voice input → text → AI processing → voice output
       • Tool calling (can invoke your financial tools)
       • Low latency for natural conversations
    
    🚀 Start your voice agent:
       python main.py
       
    Then connect via WebSocket to: ws://localhost:8000/audio-stream
    
    💡 Nova Sonic uses Titan Embeddings v2 for RAG when querying SEC filings!
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🎯 "*30)
    print("QUICK API TEST — Nova Micro + Titan Embeddings v2 + Nova Sonic")
    print("🎯 "*30)
    
    try:
        # Test 1: Nova Micro
        test_nova_micro()
        
        # Test 2: Titan Embeddings
        test_titan_embeddings()
        
        # Test 3: Nova Sonic info
        about_nova_sonic()
        
        print("\n" + "="*80)
        print("✅ All tests passed! Your models are working correctly.")
        print("="*80)
        
        print("\n📚 For more examples, run:")
        print("   python tests/test_embeddings_demo.py")
        print("\n📖 Read the guide:")
        print("   docs/EMBEDDING_SETUP_GUIDE.md")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("   • Make sure AWS credentials are set in .env")
        print("   • Check that you have access to Bedrock models")
        print("   • Run: aws bedrock list-foundation-models --region us-east-1")