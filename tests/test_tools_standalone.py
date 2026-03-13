"""
test_tools_standalone.py — Quick verification that all 4 tools work independently.

Run this BEFORE test_audio_with_tools.py to catch tool backend issues early.

Usage:
    python tests/test_tools_standalone.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.market_data import get_market_snapshot
from tools.sec_rag import query_sec_filings
from tools.quant_model import run_monte_carlo
from tools.vault_logger import log_insight


async def test_tool_1_market_data():
    """Test Tool 1: query_live_market_data"""
    
    try:
        result = await get_market_snapshot(ticker="AMD")
        print(f"✅ SUCCESS")
        print(f"   Ticker: {result.get('ticker')}")
        print(f"   Price: ${result.get('price')}")
        print(f"   Volume: {result.get('volume'):,}")
        print(f"   Change: {result.get('change_percent')}%")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_2_sec_rag():
    """Test Tool 2: analyze_sec_filings_rag"""
    
    try:
        result = await query_sec_filings(
            company="NVIDIA",
            topic="supply chain constraints"
        )
        print(f"✅ SUCCESS")
        print(f"   Backend: {result.get('backend')}")
        print(f"   Chunks returned: {len(result.get('chunks', []))}")
        if result.get('chunks'):
            print(f"   First 200 chars: {result['chunks'][0][:200]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_3_monte_carlo():
    """Test Tool 3: execute_quantitative_model"""
    
    try:
        result = await run_monte_carlo(
            ticker="NVDA",
            days=30,
            simulations=10_000
        )
        print(f"✅ SUCCESS")
        print(f"   Ticker: {result.get('ticker')}")
        print(f"   Current price: ${result.get('current_price')}")
        print(f"   P10 (worst case): ${result.get('p10')}")
        print(f"   P50 (median): ${result.get('p50')}")
        print(f"   P90 (best case): ${result.get('p90')}")
        print(f"   Mean: ${result.get('mean')}")
        print(f"   Paths: {result.get('simulations'):,}")
        print(f"   Execution mode: {result.get('execution_mode')}")
        print(f"   Engine: {result.get('calculation_engine')}")
        print(f"   Simulation time: {result.get('simulation_time_seconds')}s")
        print(f"   Total time: {result.get('total_time_seconds')}s")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_4_vault_logger():
    """Test Tool 4: log_research_insight"""
    
    try:
        result = await log_insight(
            title="Test Insight from Standalone Tool Test",
            content="This is a test note to verify the vault logger works.",
            tags=["test", "tool-verification"]
        )
        print(f"✅ SUCCESS")
        print(f"   Saved: {result.get('saved')}")
        print(f"   File: {result.get('filepath')}")
        print(f"   Message: {result.get('message')}")
        
        # Verify file actually exists
        from pathlib import Path
        if result.get('saved') and result.get('filepath'):
            file_path = Path(result['filepath'])
            if file_path.exists():
                print(f"   ✓ File verified on disk")
            else:
                print(f"   ⚠ File not found on disk!")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all 4 tool tests."""
    print("\nThis test verifies each tool backend works independently.\n")
    
    results = []
    
    # Test all 4 tools
    results.append(("Tool 1: Market Data", await test_tool_1_market_data()))
    results.append(("Tool 2: SEC RAG", await test_tool_2_sec_rag()))
    results.append(("Tool 3: Monte Carlo", await test_tool_3_monte_carlo()))
    results.append(("Tool 4: Vault Logger", await test_tool_4_vault_logger()))
    
    # Summary
    print("SUMMARY")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tools working")
    
    if passed_count == total_count:
        print("\n🎉 ALL TOOLS WORKING! You're ready to run test_audio_with_tools.py")
    else:
        print("\n⚠️  Fix the failing tools before testing voice integration")
        print("   See error details above for each failed tool")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
