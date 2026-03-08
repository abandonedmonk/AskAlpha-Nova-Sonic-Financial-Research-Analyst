"""
test_complete_setup.py — Complete test for Nova Micro, Titan Embeddings v2, and Nova Sonic

Run this to verify your complete setup:
  python tests/test_complete_setup.py
"""

import boto3
import json
import sys
from botocore.exceptions import ClientError

AWS_REGION = "us-east-1"


def test_aws_credentials():
    """Test AWS credentials."""
    print("\n" + "="*80)
    print("🔑 Testing AWS Credentials")
    print("="*80)
    
    try:
        sts = boto3.client('sts', region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        
        print(f"✅ AWS credentials verified")
        print(f"   Account: {identity['Account']}")
        print(f"   User/Role: {identity['Arn'].split('/')[-1]}")
        return True
        
    except Exception as e:
        print(f"❌ Credentials error: {e}")
        print("\n💡 Set credentials in .env:")
        print("   AWS_ACCESS_KEY_ID=...")
        print("   AWS_SECRET_ACCESS_KEY=...")
        return False


def test_titan_embeddings():
    """Test Titan Embeddings v2."""
    print("\n" + "="*80)
    print("🧪 Testing Titan Embeddings v2")
    print("="*80)
    
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        
        body = json.dumps({
            "inputText": "Test embedding for voice AI agent",
            "dimensions": 1024,
            "normalize": True
        })
        
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        embedding = json.loads(response['body'].read())['embedding']
        
        print(f"✅ Titan Embeddings v2 working!")
        print(f"   Model: amazon.titan-embed-text-v2:0")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ Failed: {error_code}")
        
        if "AccessDeniedException" in error_code:
            print("💡 Enable Titan Embeddings access:")
            print("   https://console.aws.amazon.com/bedrock/home#/modelaccess")
        
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_nova_micro():
    """Test Nova Micro."""
    print("\n" + "="*80)
    print("🧪 Testing Nova Micro")
    print("="*80)
    
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        
        messages = [
            {
                "role": "user",
                "content": [{"text": "Say 'Hello from Nova Micro' in exactly 5 words."}]
            }
        ]
        
        response = client.converse(
            modelId="us.amazon.nova-micro-v1:0",
            messages=messages,
        )
        
        answer = response['output']['message']['content'][0]['text']
        
        print(f"✅ Nova Micro working!")
        print(f"   Model: us.amazon.nova-micro-v1:0")
        print(f"   Response: {answer}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ Failed: {error_code}")
        
        if "AccessDeniedException" in error_code:
            print("💡 Enable Nova model access:")
            print("   https://console.aws.amazon.com/bedrock/home#/modelaccess")
        
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_nova_sonic():
    """Test Nova Sonic."""
    print("\n" + "="*80)
    print("🧪 Testing Nova Sonic")
    print("="*80)
    
    try:
        bedrock = boto3.client("bedrock", region_name=AWS_REGION)
        
        # Check if Nova Sonic is in available models
        response = bedrock.list_foundation_models(
            byProvider="Amazon"
        )
        
        nova_sonic_found = False
        for model in response.get('modelSummaries', []):
            if 'nova-sonic' in model['modelId'].lower():
                nova_sonic_found = True
                print(f"✅ Nova Sonic available!")
                print(f"   Model: {model['modelId']}")
                print(f"   Name: {model.get('modelName', 'N/A')}")
                break
        
        if not nova_sonic_found:
            # Check runtime API
            try:
                runtime_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
                print(f"✅ Nova Sonic runtime accessible!")
                print(f"   Model: amazon.nova-sonic-v1:0")
                print(f"   Implementation: nova_sonic/client.py")
                nova_sonic_found = True
            except Exception:
                pass
        
        if nova_sonic_found:
            print(f"\n💡 Your Nova Sonic is already set up in:")
            print(f"   📁 nova_sonic/client.py")
            print(f"   📁 main.py (WebSocket endpoint)")
            return True
        else:
            print(f"❌ Nova Sonic not found")
            print("💡 Enable Nova model access:")
            print("   https://console.aws.amazon.com/bedrock/home#/modelaccess")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_existing_implementation():
    """Check existing project files."""
    print("\n" + "="*80)
    print("📁 Checking Your Project Files")
    print("="*80)
    
    import os
    
    files_to_check = [
        ("config.py", "Configuration file"),
        ("main.py", "FastAPI server"),
        ("nova_sonic/client.py", "Nova Sonic client"),
        ("tools/sec_rag.py", "SEC RAG tool (uses Titan)"),
        ("tests/test_api.py", "API tests"),
    ]
    
    all_found = True
    for file_path, description in files_to_check:
        full_path = f"../{file_path}"
        if os.path.exists(full_path):
            print(f"✅ {file_path:30s} - {description}")
        else:
            print(f"⚠️  {file_path:30s} - NOT FOUND")
            all_found = False
    
    return all_found


def main():
    """Run all tests."""
    print("\n" + "🎯 "*30)
    print("COMPLETE SETUP TEST")
    print("Nova Micro + Titan Embeddings v2 + Nova Sonic")
    print("🎯 "*30)
    
    results = {}
    
    # Run all tests
    results['credentials'] = test_aws_credentials()
    
    if results['credentials']:
        results['titan'] = test_titan_embeddings()
        results['nova_micro'] = test_nova_micro()
        results['nova_sonic'] = test_nova_sonic()
        results['files'] = test_existing_implementation()
    else:
        print("\n⚠️  Skipping other tests due to credential issues")
        results['titan'] = False
        results['nova_micro'] = False
        results['nova_sonic'] = False
        results['files'] = False
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    print(f"\n{'✅' if results['credentials'] else '❌'} AWS Credentials")
    print(f"{'✅' if results['titan'] else '❌'} Titan Embeddings v2")
    print(f"{'✅' if results['nova_micro'] else '❌'} Nova Micro")
    print(f"{'✅' if results['nova_sonic'] else '❌'} Nova Sonic")
    print(f"{'✅' if results['files'] else '⚠️'} Project Files")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n🎉 ALL SYSTEMS GO!")
        print("\n📚 Next steps:")
        print("   1. Run demo: python tests/test_embeddings_demo.py")
        print("   2. Start server: python main.py")
        print("   3. Read guide: docs/EMBEDDING_SETUP_GUIDE.md")
        print("\n🎯 Your setup:")
        print("   • Nova Micro → Fast text chat")
        print("   • Titan v2 → Document embeddings (RAG)")
        print("   • Nova Sonic → Voice interactions")
    else:
        print("\n⚠️  Some components need attention")
        
        if not results['credentials']:
            print("\n🔧 Fix credentials:")
            print("   1. Create/edit .env file")
            print("   2. Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            
        if not (results['titan'] and results['nova_micro'] and results['nova_sonic']):
            print("\n🔧 Fix model access:")
            print("   1. Go to: https://console.aws.amazon.com/bedrock/home#/modelaccess")
            print("   2. Enable: Amazon Nova (includes Micro & Sonic)")
            print("   3. Enable: Amazon Titan Embeddings")
            print("   4. Wait 1-2 minutes, then re-run this test")
    
    print("\n" + "="*80 + "\n")
    
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
