import asyncio
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_live import ChatGeminiLive
from typing import List

API_KEY = os.getenv("GOOGLE_GEMINI_KEY")

async def main():
    chat = ChatGeminiLive(api_key=API_KEY)
    
    messages: List[BaseMessage] = []
    print("\n--- Gemini Live Conversation (type 'exit' to quit) ---")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break
            
        messages.append(HumanMessage(content=user_input))
        
        print("Model: ", end="", flush=True)
        
        full_response_content = ""
        async for chunk in chat.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response_content += chunk.content
        
        messages.append(AIMessage(content=full_response_content))
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())