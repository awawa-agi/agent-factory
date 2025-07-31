"""
LLM Provider module for different API providers.

This module contains:
- GenerationStopReason: Enumeration for LLM stop reasons
- GenerationResult: Data class for generation results
- LLMProvider: Abstract base class for LLM providers
- VllmOpenAIProvider: Vllm OpenAI API provider implementation
- AnthropicProvider: Anthropic API provider implementation
"""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# Third-party API clients
from openai import AsyncOpenAI

# Message system imports
from .core.messages import Messages, UsageInfo
from .core.config import ApiConfig, GenerationConfig


# ========== Stop Reason Enumeration ==========

class GenerationStopReason(StrEnum):
    """Enumeration for LLM generation stop reasons"""
    STOP_SEQUENCE = "stop_sequence"      # Hit stop sequence
    END_TURN = "end_turn"                # Natural end of turn
    MAX_TOKENS = "max_tokens"            # Hit max token limit
    
    @classmethod
    def from_openai_finish_reason(cls, finish_reason: str | None, stop_reason: str | None = None) -> 'GenerationStopReason':
        """Convert OpenAI finish_reason to standardized LLM stop reason"""
        if finish_reason == 'stop':
            if stop_reason is not None:
                return cls.STOP_SEQUENCE
            else:
                return cls.END_TURN
        elif finish_reason == 'length':
            return cls.MAX_TOKENS

        else:
            raise ValueError(f"Unknown OpenAI finish_reason: {finish_reason}")
    
    @classmethod
    def from_anthropic_stop_reason(cls, stop_reason: str) -> 'GenerationStopReason':
        """Convert Anthropic stop_reason to standardized LLM stop reason"""
        if stop_reason == "end_turn":
            return cls.END_TURN
        elif stop_reason == "max_tokens":
            return cls.MAX_TOKENS
        elif stop_reason == "stop_sequence":
            return cls.STOP_SEQUENCE
        else:
            raise ValueError(f"Unknown Anthropic stop_reason: {stop_reason}")


# ========== Data Structures ==========

class GenerationResult(BaseModel):
    """LLM generation result"""
    content: str
    stop_reason: GenerationStopReason
    stop_sequence: str | None = None
    num_completion_tokens: int = 0
    usage: UsageInfo | None = None

class TokenizeResult(BaseModel):
    """LLM tokenize result"""
    count: int
    max_model_len: int
    tokens: List[int]
    token_strs: str | None

# ========== LLM Provider Abstract Base Class ==========

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: ApiConfig):
        self.config = config
    
    @abstractmethod
    async def generate(self, messages: Messages, config: GenerationConfig, max_tokens: int, assistant_prefix: str = "") -> GenerationResult:
        """Generate response"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def tokenize(self, messages: Messages) -> List[int]:
        """Tokenize messages"""
        raise NotImplementedError("Subclasses must implement this method")
    
    @property
    def provider_name(self) -> str:
        """Get API format name (used for image processing)"""
        return self.config.provider


# ========== OpenAI Provider ==========

class VllmOpenAIProvider(LLMProvider):
    """Vllm OpenAI API provider"""
    
    def __init__(self, config: ApiConfig):
        super().__init__(config)
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    async def generate(self, messages: Messages, config: GenerationConfig, max_tokens: int, assistant_prefix: str = "") -> GenerationResult:
        """Generate response using OpenAI API"""
        openai_messages = messages.to_openai_format()
        
        gen_args = {
            "model": config.model_name, # type: ignore[attr-defined]
            "messages": openai_messages,
            "stop": config.stop_sequences,
            "max_tokens": max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "extra_body": {
                "top_k": config.top_k,
                "min_p": config.min_p,
                "repetition_penalty": config.repetition_penalty,
            },
        }

        if assistant_prefix:
            gen_args["messages"].append({"role": "assistant", "content": assistant_prefix})
            gen_args["extra_body"]["add_generation_prompt"] = False
            gen_args["extra_body"]["continue_final_message"] = True
        
        if config.stream_output:
            gen_result = await self._stream_generate(gen_args)
        else:
            response = await self.client.chat.completions.create(**gen_args)
            gen_result = self._process_response(response)
        
        if assistant_prefix:
            gen_result.content = f"{assistant_prefix}{gen_result.content}"
        if gen_result.stop_reason == GenerationStopReason.STOP_SEQUENCE:
            gen_result.content = f"{gen_result.content}{gen_result.stop_sequence}"
        
        return gen_result
    
    def _process_response(self, response) -> GenerationResult:
        """Process OpenAI response"""
        choice = response.choices[0]
        stop_reason = GenerationStopReason.from_openai_finish_reason(
            choice.finish_reason, choice.stop_reason
        )

        if type(choice.message.content) != str:
            from loguru import logger
            logger.warning(f"OpenAI response content is not a string: {choice.message.content}")
        
        if type(choice.stop_reason) != str and choice.stop_reason is not None:
            from loguru import logger
            logger.warning(f"OpenAI response stop_reason is not a string: {choice.stop_reason}")

        return GenerationResult(
            content=str(choice.message.content),
            stop_reason=stop_reason,
            stop_sequence=str(choice.stop_reason) if stop_reason == GenerationStopReason.STOP_SEQUENCE else None,
            num_completion_tokens=response.usage.completion_tokens,
            usage=UsageInfo.model_validate(response.usage.to_dict())
        )
    
    async def _stream_generate(self, gen_args: Dict[str, Any], print_response: bool = True) -> GenerationResult:
        """Stream generation"""
        stream = await self.client.chat.completions.create(
            **gen_args,
            stream=True,
            stream_options={"include_usage": True}
        )
        
        content_buffer = ""
        stop_reason, finish_reason, num_completion_tokens = None, None, 0
        usage_info = None
        
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content = delta.content
                    content_buffer += content
                    if print_response:
                        print(content, end="", flush=True)
                
                if hasattr(chunk.choices[0], "stop_reason"):
                    stop_reason = chunk.choices[0].stop_reason # type: ignore[attr-defined]
                if hasattr(chunk.choices[0], "finish_reason"):
                    finish_reason = chunk.choices[0].finish_reason
            
            if chunk.usage:
                num_completion_tokens = chunk.usage.completion_tokens
                usage_info = UsageInfo.model_validate(chunk.usage.to_dict())
        
        stop_reason = GenerationStopReason.from_openai_finish_reason(
            finish_reason, stop_reason
        )

        
        return GenerationResult(
            content=content_buffer,
            stop_reason=stop_reason,
            stop_sequence=str(stop_reason) if stop_reason == GenerationStopReason.STOP_SEQUENCE else None,
            num_completion_tokens=num_completion_tokens,
            usage=usage_info
        )

    async def tokenize(self, messages: Messages) -> TokenizeResult:
        """Tokenize messages asynchronously"""
        import httpx  # Local import to avoid global dependency if not used
        from loguru import logger
        
        url = self.config.base_url.replace("/v1", "/tokenize")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(url, json={"messages": messages.to_openai_format()})
                response.raise_for_status()
                return TokenizeResult.model_validate(response.json())
        except httpx.TimeoutException as e:
            logger.debug(f"Tokenize request timeout for URL {url}: {e}")
            raise
        except Exception as e:
            logger.debug(f"Tokenize request failed for URL {url}: {e}")
            raise


# ========== Anthropic Provider ==========

class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""
    
    def __init__(self, config: ApiConfig):
        super().__init__(config)
        # Lazy import
        from anthropic import AsyncAnthropic # type: ignore[import-untyped]
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    async def generate(self, messages: Messages, config: GenerationConfig, max_tokens: int, assistant_prefix: str = "") -> GenerationResult:
        """Generate response using Anthropic API"""
        system_message, anthropic_messages = messages.to_anthropic_format()
        
        if assistant_prefix:
            anthropic_messages.append({"role": "assistant", "content": assistant_prefix})
        
        gen_args = {
            "model": config.model_name,
            "messages": anthropic_messages,
            "system": system_message,
            "stop_sequences": config.stop_sequences,
            "max_tokens": max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        # Anthropic only supports streaming
        gen_result = await self._stream_generate(gen_args)
        
        if assistant_prefix:
            gen_result.content = f"{assistant_prefix}{gen_result.content}"
        if gen_result.stop_reason == GenerationStopReason.STOP_SEQUENCE:
            gen_result.content = f"{gen_result.content}{gen_result.stop_sequence}"
        
        return gen_result
    
    async def _stream_generate(self, gen_args: Dict[str, Any], print_response: bool = True) -> GenerationResult:
        """Anthropic streaming generation"""
        async with self.client.messages.stream(**gen_args) as stream:
            content_buffer = ""
            async for event in stream:
                if event.type == "text" and print_response:
                    print(event.text, end="", flush=True)
                if event.type == "text":
                    content_buffer += event.text
            
            final_message = await stream.get_final_message()
        
        stop_reason = GenerationStopReason.from_anthropic_stop_reason(final_message.stop_reason)
        
        return GenerationResult(
            content=content_buffer,
            stop_reason=stop_reason,
            stop_sequence=final_message.stop_sequence if stop_reason == GenerationStopReason.STOP_SEQUENCE else None,
            num_completion_tokens=final_message.usage.output_tokens,
            usage=final_message.usage.to_dict()
        )


# ========== Factory Functions ==========

def create_llm_provider(
    config: ApiConfig
) -> LLMProvider:
    """Factory function to create LLM provider"""
    
    if config.provider == "vllm_openai":
        return VllmOpenAIProvider(config)
    elif config.provider == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported API provider: {config.provider}") 