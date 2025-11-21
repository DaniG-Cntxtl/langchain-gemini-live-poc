import asyncio
from google import generativeai as genai
from google.generativeai import types

genai.configure(api_key="AIzaSyC9vifSNF_RB6hOaWL59ikUk1-6r2AT0uQ")
client = genai.Client()
model = "gemini-2.5-flash-native-audio-preview-09-2025"

config = {
    "response_modalities": [types.GenerationConfig.Modality.AUDIO],
    "output_audio_transcription": {}
}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        message = "Hello? Gemini are you there?"

        await session.send_client_content(
            turns=[{"role": "user", "parts": [{"text": message}]}], turn_complete=True
        )

        async for response in session.receive():
            if response.server_content.model_turn:
                print("Model turn:", response.server_content.model_turn)
            if response.server_content.output_transcription:
                print("Transcript:", response.server_content.output_transcription.text)

if __name__ == "__main__":
    asyncio.run(main())
