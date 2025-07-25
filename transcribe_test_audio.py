import assemblyai as aai
import json
import os

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
crypto_terms = ["blockchain", "Ethereum", "zero-knowledge proofs", "staking", "DeFi", "NFT", "minting", "royalty"]  # From earlier

config = aai.TranscriptionConfig(word_boost=crypto_terms)
transcriber = aai.Transcriber(config=config)

with open("test_audio/test_cases.json", "r") as f:
    test_cases = json.load(f)

transcripts = []
for case in test_cases:
    audio_path = case["filename"]
    transcript = transcriber.transcribe(audio_path)
    transcripts.append(transcript.text)
    print(f"Transcript for {audio_path}: {transcript.text}")

# Save transcripts for RAG queries
with open("transcripts.json", "w") as f:
    json.dump(transcripts, f)