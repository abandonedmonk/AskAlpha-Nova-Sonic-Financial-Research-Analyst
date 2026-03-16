"""
test_web_ui_debug.py - Diagnostic test comparing WebSocket vs direct stream approaches

Run this to understand:
1. Is audio being captured from the browser?
2. Are events flowing back from Nova Sonic?
3. Where is the bottleneck?

Usage:
  python tests/test_web_ui_debug.py
  
  Then in another terminal:
  1. Start backend: python main.py
  2. Open frontend in browser: http://localhost:5173
  3. This script will show you what the backend receives in real-time
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_web_ui_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────

async def monitor_websocket():
    """
    Connect to the backend WebSocket as if we're the frontend,
    then show what events flow through in real-time.
    """
    uri = "ws://localhost:8000/ws/voice"
    logger.info("=" * 80)
    logger.info("WebSocket Monitor Started")
    logger.info("=" * 80)
    logger.info(f"Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✓ Connected to backend")
            logger.info("")
            logger.info("Monitoring events from backend...")
            logger.info("(When you speak in the web UI, events will appear here)")
            logger.info("")
            
            event_count = 0
            event_types_received = {}
            
            try:
                while True:
                    message = await websocket.recv()
                    event_count += 1
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    
                    if isinstance(message, bytes):
                        logger.info(f"[{timestamp}] EVENT #{event_count}: AUDIO CHUNK ({len(message)} bytes)")
                        event_types_received['audio'] = event_types_received.get('audio', 0) + 1
                    else:
                        try:
                            data = json.loads(message)
                            event_type = data.get('type', 'unknown')
                            event_types_received[event_type] = event_types_received.get(event_type, 0) + 1
                            
                            logger.info(f"[{timestamp}] EVENT #{event_count}: {event_type.upper()}")
                            
                            # Show event details
                            if event_type == 'user_transcript':
                                logger.info(f"    🎤 User said: {data.get('text', '')}")
                            elif event_type == 'transcript':
                                text = data.get('text', '')
                                logger.info(f"    🤖 Model says: {text[:100]}")
                            elif event_type == 'tool_call':
                                logger.info(f"    🔧 Tool invoked: {data.get('tool_name', '')}")
                            elif event_type == 'tool_result':
                                logger.info(f"    ✓ Tool {data.get('tool_name', '')} returned")
                            elif event_type == 'response':
                                logger.info(f"    ✓ Response complete")
                            
                            logger.info('')
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse message: {message[:100]}")
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
            finally:
                logger.info("")
                logger.info("=" * 80)
                logger.info("WebSocket Connection Closed")
                logger.info("=" * 80)
                logger.info(f"Total events received: {event_count}")
                logger.info("Events by type:")
                for etype, count in sorted(event_types_received.items()):
                    logger.info(f"  - {etype}: {count}")
                logger.info("")
                
    except ConnectionRefusedError:
        logger.error(f"✗ Cannot connect to {uri}")
        logger.error("Make sure the backend is running: python main.py")
    except Exception as e:
        logger.error(f"✗ Connection error: {e}")


# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Voice AI Agent - Web UI Debug Monitor")
    logger.info("")
    logger.info("This script monitors WebSocket events between frontend and backend.")
    logger.info("It helps diagnose:")
    logger.info("  1. If audio is flowing from browser to server")
    logger.info("  2. If events are flowing back from Nova Sonic")
    logger.info("  3. What types of events are generated")
    logger.info("")
    logger.info("Steps:")
    logger.info("  1. Start the backend: python main.py")
    logger.info("  2. Open frontend: http://localhost:5173")
    logger.info("  3. Run this monitor: python tests/test_web_ui_debug.py")
    logger.info("  4. Speak into the microphone in the web UI")
    logger.info("  5. Watch this terminal for event flow")
    logger.info("")
    
    asyncio.run(monitor_websocket())
