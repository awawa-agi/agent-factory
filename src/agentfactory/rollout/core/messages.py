from __future__ import annotations

import asyncio
import base64
import io
import json
import textwrap
from enum import StrEnum
from pathlib import Path
from typing import (
    Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union, 
    Annotated, TypeVar, TYPE_CHECKING, Literal
)

import httpx
from PIL import Image
from pydantic import BaseModel, Field, RootModel, model_validator, PrivateAttr

from .content import (
    TextContent,
    ImageContent,
    McpPromptContent,
)

# ---------------------------------------------------------------------------
# Core enums and constants
# ---------------------------------------------------------------------------

class MessageRole(StrEnum):
    """Available message roles in conversations."""
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    TOOL = "tool"

# Content union type with proper discrimination
Content = Annotated[Union[TextContent, ImageContent, McpPromptContent], Field(discriminator="type")]

# ---------------------------------------------------------------------------
# Base message class
# ---------------------------------------------------------------------------

class BaseMessage(BaseModel):
    """Base class for all message types with common functionality."""
    
    role: MessageRole = Field(frozen=True)
    content: List[Content] = Field(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_content(cls, data: Any) -> Any:
        """Convert string content to proper TextContent format."""
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            data["content"] = [{"type": "text", "text": data["content"]}]
        return data

    def get_text(self) -> str:
        """Extract and concatenate all text content."""
        return "".join(
            item.text for item in self.content 
            if isinstance(item, TextContent)
        )

    def get_images(self) -> List[ImageContent]:
        """Extract all image content."""
        return [
            item for item in self.content 
            if isinstance(item, ImageContent)
        ]

    def load_images(self) -> None:
        """Eagerly load all images in this message."""
        for image in self.get_images():
            image.fetch_bytes() 

    def update_content(self, content: List[Content]) -> BaseMessage:
        """Update the content of this message and validate."""
        # Since content is frozen, we need to create a new instance
        return type(self).model_validate(
            {"role": self.role, "content": content}
        )


# ---------------------------------------------------------------------------
# Specialized message classes
# ---------------------------------------------------------------------------

class SystemMessage(BaseMessage):
    """System message for setting conversation context."""
    role: Literal[MessageRole.SYSTEM] = Field(default=MessageRole.SYSTEM, frozen=True)

class UserMessage(BaseMessage):
    """User input message."""
    role: Literal[MessageRole.USER] = Field(default=MessageRole.USER, frozen=True)

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[dict[str, Any]] = None

class AssistantMessage(BaseMessage):
    """Assistant response message with optional usage metadata, turn rewards and advantages."""
    role: Literal[MessageRole.ASSISTANT] = Field(default=MessageRole.ASSISTANT, frozen=True)
    usage: Optional[UsageInfo] = Field(default=None)
    turn_reward_components: Optional[Dict[str, float]] = Field(default=None)
    weighted_turn_reward: Optional[float] = Field(default=None)
    
    # MT-GRPO advantage fields for logging
    turn_level_advantage: Optional[float] = Field(default=None, description="Turn-level advantage (A^T)")
    outcome_level_advantage: Optional[float] = Field(default=None, description="Outcome-level advantage (A^O)")
    emt_grpo_advantage: Optional[float] = Field(default=None, description="Final EMT-GRPO advantage (A^EMT)")

class ToolMessage(BaseMessage):
    """Tool execution message with metadata."""
    role: Literal[MessageRole.TOOL] = Field(default=MessageRole.TOOL, frozen=True)
    tool_name: str
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    structured_output: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Message union type with discriminator
# ---------------------------------------------------------------------------

MessageUnion = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator='role')
]

# ---------------------------------------------------------------------------
# Message collection
# ---------------------------------------------------------------------------

class Messages(RootModel[List[MessageUnion]]):
    """Collection of messages with rich utility methods."""
    
    root: List[MessageUnion]

    @model_validator(mode="before")
    @classmethod
    def parse_string_input(cls, data: Any) -> Any:
        """If input is a string, try to parse it as JSON."""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse JSON string: {e}")
        return data

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[MessageUnion]:
        return iter(self.root)

    def __getitem__(self, index: int) -> MessageUnion:
        return self.root[index]

    def __setitem__(self, index: int, value: MessageUnion) -> None:
        """Set message at a specific index."""
        self.root[index] = value

    def __bool__(self) -> bool:
        return bool(self.root)

    # ---------------------------------------------------------------------------
    # Mutation methods
    # ---------------------------------------------------------------------------

    def append(self, message: Union[MessageUnion, Dict[str, Any]]) -> None:
        """Add message to the end of collection."""
        if isinstance(message, dict):
            # Use Pydantic's discriminated union to create proper message type
            temp_messages = Messages.model_validate([message])
            message = temp_messages.root[0]
        self.root.append(message)

    def extend(self, messages: Union[Sequence[Union[MessageUnion, Dict[str, Any]]], "Messages"]) -> None:
        """Add multiple messages to collection."""
        if isinstance(messages, Messages):
            self.root.extend(messages.root)
        else:
            for msg in messages:
                self.append(msg)

    def insert(self, index: int, message: Union[MessageUnion, Dict[str, Any]]) -> None:
        """Insert message at specific position."""
        if isinstance(message, dict):
            # Use Pydantic's discriminated union to create proper message type
            temp_messages = Messages.model_validate([message])
            message = temp_messages.root[0]
        self.root.insert(index, message)

    # ---------------------------------------------------------------------------
    # Query and filtering methods
    # ---------------------------------------------------------------------------

    def find_last(self, role: Optional[MessageRole] = None) -> Optional[MessageUnion]:
        """Find the last message, optionally filtered by role."""
        if role is None:
            return self.root[-1] if self.root else None
        
        for message in reversed(self.root):
            if message.role == role:
                return message
        return None

    def last_assistant_message(self) -> AssistantMessage | None:
        """Find the last assistant message."""
        for message in reversed(self.root):
            if isinstance(message, AssistantMessage):
                return message
        return None

    def filter_by_role(self, role: MessageRole) -> List[MessageUnion]:
        """Get all messages with specified role."""
        return [msg for msg in self.root if msg.role == role]

    def truncate_after_last_assistant(self) -> "Messages":
        """Create copy truncated after the last assistant message (inclusive).
        
        Useful for removing any trailing tool calls or user messages after
        the final assistant response in a conversation.
        
        Returns:
            New Messages instance containing messages up to and including
            the last assistant message. If no assistant messages exist,
            returns a copy of the entire collection.
        """
        last_assistant_index = -1
        
        # Find the last assistant message
        for i in range(len(self.root) - 1, -1, -1):
            if self.root[i].role == MessageRole.ASSISTANT:
                last_assistant_index = i
                break
        
        # If no assistant message found, return full copy
        if last_assistant_index == -1:
            return Messages(root=self.root.copy())
        
        # Return truncated copy
        return Messages(root=self.root[:last_assistant_index + 1])

    # ---------------------------------------------------------------------------
    # Content transformation methods
    # ---------------------------------------------------------------------------

    def strip_images(self) -> "Messages":
        """Create copy with all image content removed."""
        stripped_messages = []
        for message in self.root:
            text_content = [
                item for item in message.content 
                if isinstance(item, TextContent)
            ]
            stripped_messages.append(
                message.model_copy(update={"content": text_content})
            )
        return Messages(root=stripped_messages)

    def load_all_images(self) -> None:
        """Synchronously load all images in all messages."""
        for message in self.root:
            message.load_images()

    # ---------------------------------------------------------------------------
    # Format conversion methods
    # ---------------------------------------------------------------------------

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI API format."""
        return [
            message.model_dump(include={"role", "content"}) 
            for message in self.root
        ]

    def to_anthropic_format(self) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert to Anthropic API format (system_prompt, messages)."""
        system_messages = []
        conversation_messages = []
        
        for message in self.root:
            if message.role == MessageRole.SYSTEM:
                system_messages.append(message.get_text())
            elif message.role in {MessageRole.USER, MessageRole.ASSISTANT}:
                conversation_messages.append(
                    message.model_dump(include={"role", "content"})
                )
        
        system_prompt = "\n\n".join(system_messages) if system_messages else None
        return system_prompt, conversation_messages

    # ---------------------------------------------------------------------------
    # Display and formatting methods
    # ---------------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render conversation as clean Markdown."""
        sections = []
        for message in self.root:
            header = f"**{message.role.value.upper()}**"
            content_parts = []
            
            for item in message.content:
                if isinstance(item, TextContent):
                    content_parts.append(textwrap.indent(item.text.strip(), "  "))
                elif isinstance(item, ImageContent):
                    size_info = f"{item.width}×{item.height}" if item.size else "unknown size"
                    content_parts.append(f"  ![Image]({item.url}) <!-- {size_info} -->")
            
            sections.append(f"{header}\n\n" + "\n".join(content_parts))
        
        return "\n\n---\n\n".join(sections)

    def to_pretty_string(self, skip_system: bool = False) -> str:
        """Create visually appealing text representation."""
        lines = []
        
        for index, message in enumerate(self.root, 1):
            if skip_system and message.role == MessageRole.SYSTEM:
                continue
                
            lines.append(f"┌─ {message.role.value.upper()} Message #{index}")
            
            for item in message.content:
                if isinstance(item, TextContent):
                    lines.append("│  📝 Text Content:")
                    for text_line in item.text.strip().split('\n'):
                        lines.append(f"│     {text_line}")
                elif isinstance(item, ImageContent):
                    size_info = f"{item.width}×{item.height}" if item.size else "?×?"
                    load_status = "✓" if item.is_loaded else "○"
                    lines.append(f"│  🖼️  Image: {size_info} {load_status}")
                elif isinstance(item, McpPromptContent):
                    lines.append(f"│  🎯 MCP Prompt: {item.prompt_name} {item.prompt_arguments}")
            
            # Add type-specific information
            if isinstance(message, ToolMessage):
                lines.append(f"│  🔧 Tool: {message.tool_name}")
                if message.tool_arguments:
                    args_preview = str(message.tool_arguments)[:50]
                    if len(str(message.tool_arguments)) > 50:
                        args_preview += "..."
                    lines.append(f"│     Arguments: {args_preview}")
            
            elif isinstance(message, AssistantMessage) and message.usage:
                lines.append(f"│  📊 Usage: {message.usage}")
            
            lines.append("└─")
            lines.append("")  # Empty line between messages
        
        return "\n".join(lines).rstrip()

    # ---------------------------------------------------------------------------
    # Statistics and analysis
    # ---------------------------------------------------------------------------

    def count_images(self) -> int:
        """Count total number of images across all messages."""
        return sum(len(message.get_images()) for message in self.root)