"""
Enhanced Bidirectional Audio Test for AWS Nova Sonic

This test demonstrates and validates:
1. INPUT DIRECTION: Microphone → Nova Sonic (user speech)
2. OUTPUT DIRECTION: Nova Sonic → Speaker (assistant response)

Based on: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
"""

import os
import asyncio
import base64
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import pyaudio
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
RECORD_SECONDS = 10  # Record for 10 seconds, then wait for response


class BidirectionalAudioTest:
    """Test bidirectional audio with Nova Sonic with detailed logging."""
    
    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        self.model_id = model_id
        self.region = region
        self.client = None
        self.stream = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.audio_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            'input': {
                'audio_chunks_sent': 0,
                'transcript_events': 0,
                'user_transcripts': []
            },
            'output': {
                'audio_chunks_received': 0,
                'transcript_events': 0,
                'assistant_transcripts': []
            }
        }
        self.current_role = None
        self.display_text = False
        
    def _initialize_client(self):
        """Initialize the Bedrock client."""
        print("🔧 Initializing Bedrock Runtime client...")
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
        print(f"✅ Client initialized for region: {self.region}")
    
    async def send_event(self, event_json):
        """Send an event to the stream."""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)
    
    async def start_session(self):
        """Start a new session with Nova Sonic."""
        print("\n" + "="*70)
        print("STARTING BIDIRECTIONAL AUDIO SESSION")
        print("="*70)
        
        if not self.client:
            self._initialize_client()
        
        print("\n📡 Opening bidirectional stream...")
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        print("✅ Stream opened successfully")
        
        # Send session start event
        print("📤 Sending sessionStart event...")
        session_start = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7
                    }
                }
            }
        }
        await self.send_event(json.dumps(session_start))
        
        # Send prompt start event
        print("📤 Sending promptStart event...")
        prompt_start = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": CHANNELS,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    }
                }
            }
        }
        await self.send_event(json.dumps(prompt_start))
        
        # Send system prompt
        print("📤 Sending system prompt...")
        text_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
        await self.send_event(json.dumps(text_content_start))
        
        system_prompt = ("You are a friendly assistant. The user and you will engage in a spoken dialog "
                        "exchanging the transcripts of a natural real-time conversation. Keep your responses "
                        "short, generally two or three sentences for chatty scenarios.")
        
        text_input = {
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": system_prompt
                }
            }
        }
        await self.send_event(json.dumps(text_input))
        
        text_content_end = {
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name
                }
            }
        }
        await self.send_event(json.dumps(text_content_end))
        
        print("✅ Session initialization complete")
        
        # Start processing responses
        asyncio.create_task(self._process_responses())
    
    async def start_audio_input(self):
        """Start audio input stream."""
        print("\n" + ">"*70)
        print(">>> [INPUT DIRECTION] Starting audio input stream")
        print(">"*70)
        
        audio_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": CHANNELS,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }
                }
            }
        }
        await self.send_event(json.dumps(audio_content_start))
        print("✅ Audio input stream started")
    
    async def send_audio_chunk(self, audio_bytes):
        """Send an audio chunk to the stream."""
        if not self.is_active:
            return
        
        blob = base64.b64encode(audio_bytes)
        audio_event = {
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": blob.decode('utf-8')
                }
            }
        }
        await self.send_event(json.dumps(audio_event))
        self.stats['input']['audio_chunks_sent'] += 1
        
        # Log first chunk
        if self.stats['input']['audio_chunks_sent'] == 1:
            print("    🎤 First audio chunk sent to Nova Sonic")
    
    async def end_audio_input(self):
        """End audio input stream."""
        audio_content_end = {
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name
                }
            }
        }
        await self.send_event(json.dumps(audio_content_end))
        print(f"✅ Audio input ended ({self.stats['input']['audio_chunks_sent']} chunks sent)")
    
    async def end_session(self):
        """End the session."""
        if not self.is_active:
            return
        
        print("\n📤 Ending session...")
        prompt_end = {
            "event": {
                "promptEnd": {
                    "promptName": self.prompt_name
                }
            }
        }
        await self.send_event(json.dumps(prompt_end))
        
        session_end = {
            "event": {
                "sessionEnd": {}
            }
        }
        await self.send_event(json.dumps(session_end))
        await self.stream.input_stream.close()
        print("✅ Session ended")
    
    async def _process_responses(self):
        """Process responses from the stream."""
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                
                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)
                    
                    if 'event' in json_data:
                        # Handle content start event
                        if 'contentStart' in json_data['event']:
                            content_start = json_data['event']['contentStart']
                            self.current_role = content_start.get('role')
                            
                            # Check for assistant response
                            if self.current_role == 'ASSISTANT':
                                print("\n" + "<"*70)
                                print("<<< [OUTPUT DIRECTION] Assistant response started")
                                print("<"*70)
                            
                            # Check for speculative content
                            if 'additionalModelFields' in content_start:
                                additional_fields = json.loads(content_start['additionalModelFields'])
                                if additional_fields.get('generationStage') == 'SPECULATIVE':
                                    self.display_text = True
                                else:
                                    self.display_text = False
                        
                        # Handle text output event (transcripts)
                        elif 'textOutput' in json_data['event']:
                            text = json_data['event']['textOutput']['content']
                            
                            if self.current_role == "ASSISTANT" and self.display_text:
                                print(f"    📝 [TRANSCRIPT] Assistant: {text}")
                                self.stats['output']['transcript_events'] += 1
                                self.stats['output']['assistant_transcripts'].append(text)
                            elif self.current_role == "USER":
                                print(f"    📝 [TRANSCRIPT] You said: {text}")
                                self.stats['input']['transcript_events'] += 1
                                self.stats['input']['user_transcripts'].append(text)
                        
                        # Handle audio output
                        elif 'audioOutput' in json_data['event']:
                            audio_content = json_data['event']['audioOutput']['content']
                            audio_bytes = base64.b64decode(audio_content)
                            await self.audio_queue.put(audio_bytes)
                            self.stats['output']['audio_chunks_received'] += 1
                            
                            # Log first chunk
                            if self.stats['output']['audio_chunks_received'] == 1:
                                print("    🔊 First audio chunk received from Nova Sonic")
        except Exception as e:
            if self.is_active:
                print(f"❌ Error processing responses: {e}")
    
    async def play_audio(self):
        """Play audio responses."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True
        )
        
        print("🔊 Audio playback ready...")
        
        try:
            while self.is_active:
                audio_data = await self.audio_queue.get()
                stream.write(audio_data)
        except Exception as e:
            if self.is_active:
                print(f"❌ Error playing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    async def capture_audio(self):
        """Capture audio from microphone for a fixed duration."""
        p = pyaudio.PyAudio()
        
        # List available devices
        print("\n🎤 Audio Input Devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"   [{i}] {info['name']}")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print(f"\n🎤 Recording audio for {RECORD_SECONDS} seconds...")
        print("💬 Try saying: 'Hello, can you hear me?' or 'What's the weather like today?'")
        
        await self.start_audio_input()
        
        try:
            chunks_to_record = int(INPUT_SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)
            for i in range(chunks_to_record):
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                
                # Show progress
                if i % 50 == 0:
                    progress = (i / chunks_to_record) * 100
                    print(f"    ▶ Recording... {progress:.0f}%")
                
                await asyncio.sleep(0.01)
            
            print("    ▶ Recording complete!")
            
        except Exception as e:
            print(f"❌ Error capturing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            await self.end_audio_input()
    
    def print_statistics(self):
        """Print session statistics."""
        print("\n" + "="*70)
        print("📊 BIDIRECTIONAL AUDIO SESSION STATISTICS")
        print("="*70)
        
        print("\n[INPUT DIRECTION - User → Nova Sonic]")
        print(f"  • Audio chunks sent: {self.stats['input']['audio_chunks_sent']}")
        print(f"  • User transcript events: {self.stats['input']['transcript_events']}")
        if self.stats['input']['user_transcripts']:
            print(f"  • What you said:")
            for t in self.stats['input']['user_transcripts']:
                print(f"      '{t}'")
        
        print("\n[OUTPUT DIRECTION - Nova Sonic → User]")
        print(f"  • Audio chunks received: {self.stats['output']['audio_chunks_received']}")
        print(f"  • Assistant transcript events: {self.stats['output']['transcript_events']}")
        if self.stats['output']['assistant_transcripts']:
            print(f"  • Assistant responses:")
            for t in self.stats['output']['assistant_transcripts']:
                print(f"      '{t}'")
        
        print("\n[RESULT]")
        if (self.stats['input']['audio_chunks_sent'] > 0 and 
            self.stats['output']['audio_chunks_received'] > 0):
            print("  ✅ BIDIRECTIONAL AUDIO WORKING!")
            print("     → Audio sent to Nova Sonic successfully")
            print("     → Audio received from Nova Sonic successfully")
        else:
            print("  ⚠️  BIDIRECTIONAL AUDIO INCOMPLETE")
            if self.stats['input']['audio_chunks_sent'] == 0:
                print("     ❌ No audio was sent to Nova Sonic")
            if self.stats['output']['audio_chunks_received'] == 0:
                print("     ❌ No audio was received from Nova Sonic")
        
        print("="*70 + "\n")


async def main():
    """Main test function."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     AWS Nova Sonic - Bidirectional Audio Test                    ║
║                                                                    ║
║  This test will:                                                   ║
║  1. ▶ Record your voice for 10 seconds                           ║
║  2. ▶ Send audio to Nova Sonic                                   ║
║  3. ◀ Receive and play Nova Sonic's response                     ║
║  4. 📊 Show statistics                                            ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check AWS credentials
    print("🔍 Checking AWS credentials...")
    print(f"   Loading from: {env_path}")
    if not os.getenv('AWS_ACCESS_KEY_ID'):
        print("⚠️  WARNING: AWS_ACCESS_KEY_ID not found in environment")
        print("   Set it with: $env:AWS_ACCESS_KEY_ID='your-key'")
    if not os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("⚠️  WARNING: AWS_SECRET_ACCESS_KEY not found in environment")
        print("   Set it with: $env:AWS_SECRET_ACCESS_KEY='your-secret'")
    
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("✅ AWS credentials found")
    
    # Create test client
    test = BidirectionalAudioTest()
    
    try:
        # Start session
        await test.start_session()
        
        # Start audio playback task
        playback_task = asyncio.create_task(test.play_audio())
        
        # Capture audio for fixed duration
        await test.capture_audio()
        
        # Wait for response (give model time to process and respond)
        print("\n⏳ Waiting for Nova Sonic to respond (up to 30 seconds)...")
        await asyncio.sleep(30)
        
        # End session
        test.is_active = False
        await test.end_session()
        
        # Wait for playback to finish
        if not playback_task.done():
            playback_task.cancel()
            try:
                await playback_task
            except asyncio.CancelledError:
                pass
        
        # Show statistics
        test.print_statistics()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        test.is_active = False
    
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
