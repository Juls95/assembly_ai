# Install dependencies first:
#   brew install portaudio
#   pip install "assemblyai[extras]"

import logging
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import json
import sys

# ---
# Word boost list for jargon prioritization in real-time transcription
word_boost = [
    "Alpha", "Arbitrage", "Ask Price", "Bag Holder", "Bear Market", "Bid Price", "Blockchain", "Block Trade", "Bull Market", "Currency Pair", "Day Trading", "DeFi", "Dividend", "DYOR", "FOMO", "Fork", "FUD", "Gas Fee", "Halving", "HODL", "ICO", "Leverage", "Liquidity", "Long Position", "Margin Call", "Market Cap", "Mining", "NFT", "Pips", "Proof of Stake", "Proof of Work", "Pump and Dump", "Rug Pull", "Scalping", "Short Position", "Smart Contract", "Spread", "Staking", "Stop Loss", "Swing Trading", "Take Profit", "Volatility", "Volume", "Wallet", "Whale", "51% Attack", "Airdrop", "Altcoin", "ATH", "ROI"
]

api_key = "d694ab449585476fa390008180983a65"  # Replace with your API key if needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_begin(self, event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(self, event: TurnEvent):
    print("on_turn called")
    if event.transcript:
        print(f"Transcript: {event.transcript} (End of turn: {event.end_of_turn})")
    if event.end_of_turn and not event.turn_is_formatted:
        params = StreamingSessionParameters(format_turns=True)
        self.set_params(params)

def on_terminated(self, event: TerminationEvent):
    print(f"Session terminated: {event.audio_duration_seconds} seconds of audio processed")

def on_error(self, error: StreamingError):
    print(f"Error occurred: {error}")

def main():
    client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            api_host="streaming.assemblyai.com",
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate=8000,
            format_turns=True,
            word_boost=word_boost
        )
    )

    try:
        # Use the correct device_index if needed: aai.extras.MicrophoneStream(sample_rate=16000, device_index=YOUR_INDEX)
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=8000)
        for chunk in microphone_stream:
            print("Audio chunk read:", len(chunk))
            break  # Remove break to keep reading, this is just to test
        client.stream(microphone_stream)
    finally:
        client.disconnect(terminate=True)

def run_batch_tests():
    # Use the same API key as above
    aai.settings.api_key = api_key
    # Use the same word_boost list
    config = aai.TranscriptionConfig(word_boost=word_boost)
    transcriber = aai.Transcriber(config=config)

    with open("test_audio/test_cases.json", "r") as f:
        test_cases = json.load(f)

    for case in test_cases:
        transcript = transcriber.transcribe(case["filename"]).text
        print(f"Expected: {case['expected_transcript']}")
        print(f"Got: {transcript}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_batch_tests()
    else:
        main()