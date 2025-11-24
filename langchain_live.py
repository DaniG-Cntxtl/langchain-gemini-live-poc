import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from google import genai

class ChatGeminiLive(BaseChatModel):
    """
    A LangChain ChatModel wrapper for Google's Gemini Live (Bidirectional) API.
    
    This model specifically supports the 'gemini-live-2.5-flash-preview' model
    via the new google.genai SDK.
    
    It currently operates in a single-turn mode for compatibility with standard LangChain usage:
    1. Connects to the Live API.
    2. Sends the input messages.
    3. Streams the response (Text).
    4. Closes the connection when the turn is complete.
    """
    
    model_name: str = "gemini-live-2.5-flash-preview"
    api_key: str
    response_modalities: List[str] = ["TEXT"]
    _client: Any = None
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self._client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    @property
    def _llm_type(self) -> str:
        return "chat-google-gemini-live"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync generation is not fully supported due to the async nature of the Bidi API."""
        raise NotImplementedError("Sync generation not supported. Use ainvoke or astream.")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation."""
        final_content = ""
        
        async for chunk in self._astream(messages, stop, run_manager, **kwargs):
            msg = chunk.message
            if isinstance(msg.content, str):
                final_content += msg.content
                
        message = AIMessage(content=final_content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            # In a real app, handle system messages etc.
            pass
            
        config = {"response_modalities": ["TEXT"]}
        
        async with self._client.aio.live.connect(model=self.model_name, config=config) as session:
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": last_message.content}]},
                turn_complete=True
            )
            
            async for chunk in session.receive():
                if chunk.text:
                    text_to_yield = chunk.text
                    msg_chunk = AIMessageChunk(content=text_to_yield)
                    if run_manager:
                        await run_manager.on_llm_new_token(text_to_yield, chunk=msg_chunk)
                    yield ChatGenerationChunk(message=msg_chunk)

