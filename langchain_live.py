import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from google.ai import generativelanguage_v1alpha
from google.ai.generativelanguage_v1alpha.types import (
    GenerationConfig, 
    BidiGenerateContentSetup, 
    BidiGenerateContentClientMessage, 
    BidiGenerateContentClientContent, 
    BidiGenerateContentRealtimeInput,
    Content, 
    Part,
    Blob
)

class ChatGeminiLive(BaseChatModel):
    """
    A LangChain ChatModel wrapper for Google's Gemini Live (Bidirectional) API.
    
    This model specifically supports the 'gemini-2.5-flash-native-audio-preview-09-2025' model
    which requires the 'bidiGenerateContent' API.
    
    It currently operates in a single-turn mode for compatibility with standard LangChain usage:
    1. Connects to the Bidi API.
    2. Sends the input messages.
    3. Streams the response (Text and Audio).
    4. Closes the connection when the turn is complete.
    
    Audio data is returned in the `additional_kwargs` of the AIMessageChunk under the key 'audio'.
    """
    
    model_name: str = "models/gemini-2.5-flash-native-audio-preview-09-2025"
    api_key: str
    response_modalities: List[str] = ["AUDIO"]
    _client: Any = None
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self._client = generativelanguage_v1alpha.GenerativeServiceAsyncClient(
            client_options={"api_key": api_key}
        )

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
        final_audio_chunks = []
        
        async for chunk in self._astream(messages, stop, run_manager, **kwargs):
            msg = chunk.message
            if isinstance(msg.content, str):
                final_content += msg.content
            
            # Collect audio chunks (LangChain merges lists in additional_kwargs)
            if "audio_chunks" in msg.additional_kwargs:
                final_audio_chunks.extend(msg.additional_kwargs["audio_chunks"])
                
        final_audio = b"".join(final_audio_chunks)
                
        message = AIMessage(
            content=final_content,
            additional_kwargs={"audio": final_audio} if final_audio else {}
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        
        # Prepare content from messages
        # Current implementation assumes the last message is the user input to send.
        # History support in Bidi API works by sending history in 'turns', but for now
        # let's just send the last message as the new turn.
        
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            # In a real app, handle system messages etc.
            pass
            
        parts = [Part(text=last_message.content)]
        
        # Prepare requests generator
        async def request_generator():
            # 1. Setup
            modalities = []
            if "AUDIO" in self.response_modalities:
                modalities.append(GenerationConfig.Modality.AUDIO)
            if "TEXT" in self.response_modalities:
                modalities.append(GenerationConfig.Modality.TEXT)
                
            setup_msg = BidiGenerateContentClientMessage(
                setup=BidiGenerateContentSetup(
                    model=self.model_name,
                    generation_config=GenerationConfig(
                        response_modalities=[GenerationConfig.Modality.AUDIO]
                    ),
                    system_instruction=Content(
                        parts=[Part(text="You are a helpful AI assistant.")]
                    )
                )
            )
            yield setup_msg
            await asyncio.sleep(0.5)
            
            # 2. Client Content (User Message)
            client_msg = BidiGenerateContentClientMessage(
                client_content=BidiGenerateContentClientContent(
                    turns=[Content(role="user", parts=parts)],
                    turn_complete=True
                )
            )
            yield client_msg
            
        # Call the API
        try:
            stream = await self._client.bidi_generate_content(requests=request_generator())
            
            async for response in stream:
                if response.server_content:
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                chunk = AIMessageChunk(content=part.text)
                                if run_manager:
                                    await run_manager.on_llm_new_token(part.text, chunk=chunk)
                                yield ChatGenerationChunk(message=chunk)
                            
                            if part.inline_data:
                                # Audio data
                                chunk = AIMessageChunk(
                                    content="",
                                    additional_kwargs={"audio_chunks": [part.inline_data.data]}
                                )
                                yield ChatGenerationChunk(message=chunk)
                                
                    if response.server_content.turn_complete:
                        # Turn is done
                        break
                        
        except Exception as e:
            # Handle errors (like disconnection)
            print(f"Error in Gemini Live stream: {e}")
            raise e
