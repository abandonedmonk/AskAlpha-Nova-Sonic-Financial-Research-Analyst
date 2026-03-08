# AWS Nova Sonic - Bidirectional Audio Test Guide

## What is Bidirectional Audio?

Bidirectional audio means **two-way communication**:

1. **INPUT Direction**: Your voice → Microphone → Audio sent to Nova Sonic
2. **OUTPUT Direction**: Nova Sonic's response → Audio received → Speaker plays it back

This is like a real conversation where both sides can speak and listen.

## Test Files

### 1. `test_bidirectional_audio.py` (Recommended)
- **Enhanced test with detailed logging**
- Shows exactly what's happening in both directions
- Records for 10 seconds, then waits for response
- Displays statistics at the end
- Clear visual indicators for input/output

### 2. `nova_sonic_simple.py` (Original)
- Continuous recording until you press Enter
- Less verbose output
- Good for extended conversations

## Prerequisites

1. **AWS Credentials**: Set your AWS credentials in environment variables:
   ```powershell
   $env:AWS_ACCESS_KEY_ID = "your-access-key"
   $env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
   $env:AWS_DEFAULT_REGION = "us-east-1"
   ```

2. **Python Dependencies**: Make sure you have:
   - `pyaudio` for microphone/speaker access
   - `boto3` for AWS
   - AWS SDK packages

3. **Microphone & Speaker**: Ensure your microphone and speakers are working

## Running the Test

### Option 1: Enhanced Test (Recommended)
```powershell
cd d:\Projects\voice-ai-agent\Voice_AI_Agent_Nova
python tests\test_bidirectional_audio.py
```

**What to expect:**
1. Test starts and initializes the bidirectional stream
2. You'll see "[INPUT DIRECTION]" messages when capturing audio
3. Speak clearly into your microphone for 10 seconds
4. You'll see "[OUTPUT DIRECTION]" when Nova Sonic responds
5. Audio will play back through your speakers
6. Transcripts of both your speech and the assistant's response
7. Final statistics showing if bidirectional audio worked

**Try saying:**
- "Hello, can you hear me?"
- "What's the weather like today?"
- "Tell me a joke"

### Option 2: Original Continuous Test
```powershell
cd d:\Projects\voice-ai-agent\Voice_AI_Agent_Nova
python tests\nova_sonic_simple.py
# Press Enter when you want to stop
```

## Expected Output

### Successful Bidirectional Audio Session:
```
>>> [INPUT DIRECTION] Starting audio input stream
    🎤 First audio chunk sent to Nova Sonic
    📝 [TRANSCRIPT] You said: Hello can you hear me

<<< [OUTPUT DIRECTION] Assistant response started
    🔊 First audio chunk received from Nova Sonic
    📝 [TRANSCRIPT] Assistant: Yes I can hear you perfectly

📊 BIDIRECTIONAL AUDIO SESSION STATISTICS
[INPUT DIRECTION - User → Nova Sonic]
  • Audio chunks sent: 320
  • User transcript events: 1

[OUTPUT DIRECTION - Nova Sonic → User]
  • Audio chunks received: 180
  • Assistant transcript events: 1

[RESULT]
  ✅ BIDIRECTIONAL AUDIO WORKING!
```

## Troubleshooting

### No audio input
- Check microphone permissions in Windows Settings
- Try listing audio devices: the test will show available input devices
- Verify microphone works in other applications

### No audio output
- Check speaker/headphone connections
- Verify volume is turned up
- Check Windows audio output settings

### AWS Credentials Error
- Verify credentials are set correctly
- Check if your AWS account has access to Bedrock
- Ensure Nova Sonic is available in your region (us-east-1 recommended)

### "No audio chunks received"
- This might mean Nova Sonic didn't generate a response
- Try speaking more clearly or asking a direct question
- Check AWS CloudWatch logs for errors

## Key Features Being Tested

✅ **Bidirectional Stream Setup**: Can establish a two-way connection  
✅ **Audio Input**: Can capture microphone audio and send to Nova Sonic  
✅ **Audio Output**: Can receive audio from Nova Sonic and play it back  
✅ **Speech Recognition**: Nova Sonic transcribes your speech (USER role)  
✅ **Speech Synthesis**: Nova Sonic generates speech response (ASSISTANT role)  
✅ **Real-time Processing**: Audio flows in both directions simultaneously  

## What the AWS Documentation Says

According to the [AWS Nova Sonic documentation](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html):

- Nova Sonic supports **bidirectional streaming** for real-time voice conversations
- Audio input: 16 kHz, 16-bit, mono LPCM
- Audio output: 24 kHz, 16-bit, mono LPCM
- Supports multiple voice IDs (we use "matthew")
- Provides both audio output and text transcriptions
- Interactive mode enables turn-based conversation

This test validates all these capabilities!
