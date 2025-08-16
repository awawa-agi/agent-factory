"""
Rollout system configuration and data structures.

This module defines all configuration classes and data structures needed for
executing AI model rollouts, including API settings, generation parameters,
tool calling configs, and request/result structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

from .messages import Messages


# ============================================================================
# API and Generation Configuration
# ============================================================================

class ApiConfig(BaseModel):
    """API provider configuration for LLM services."""
    
    provider: Literal["vllm_openai", "anthropic"] = Field(
        default="vllm_openai",
        description="LLM service provider (vllm_openai or anthropic)"
    )
    api_key: str = Field(
        default="sk-proj-1234567890",
        description="Authentication key for the API service"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for API requests (overrides default)"
    )


class GenerationConfig(BaseModel):
    """Text generation parameters for controlling model output."""
    
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model variant to use (e.g., gpt-4, claude-3)"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0, le=2.0,
        description="Randomness in generation (0.0=deterministic, 2.0=very random)"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Nucleus sampling threshold (0.1=focused, 1.0=full vocab)"
    )
    top_k: int = Field(
        default=-1,
        description="Top-k sampling limit (-1=disabled, >0=limit vocabulary)"
    )
    min_p: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Minimum probability threshold for token selection"
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=0.0, le=2.0,
        description="Penalty for repeated tokens (1.0=none, >1.0=discourage)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation"
    )
    stop_sequences: List[str] = Field(
        default_factory=lambda: [],
        description="Sequences that halt generation when encountered"
    )
    stream_output: bool = Field(
        default=False,
        description="Enable real-time token streaming during generation"
    )


class ToolCallingConfig(BaseModel):
    """Configuration for AI model tool/function calling capabilities."""
    
    max_calls_per_round: int = Field(
        default=1,
        ge=1,
        description="Maximum tool calls allowed in a single conversation round"
    )
    num_parallel_calls: int = Field(
        default=1,
        ge=1,
        description="Number of tools that can be called simultaneously"
    )
    single_call_timeout: float = Field(
        default=20.0,
        gt=0,
        description="Timeout in seconds for individual tool calls"
    )
    total_call_timeout: float = Field(
        default=50.0,
        gt=0,
        description="Total timeout in seconds for all tool calls in a round"
    )
    message_role: str = Field(
        default="tool",
        description="Role identifier for tool response messages"
    )
    output_max_length: int = Field(
        default=6000,
        gt=0,
        description="Maximum character length for tool call outputs"
    )


# ============================================================================
# MCP (Model Control Protocol) Configuration
# ============================================================================

class McpConfig(BaseModel):
    """Configuration for MCP (Model Control Protocol) server connections."""
    
    server_configs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of MCP server connection configurations"
    )
    tool_limits: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-tool usage limits (tool_name -> max_calls)"
    )
    connection_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for failed connections"
    )
    connection_retry_delay: float = Field(
        default=2.0,
        ge=0,
        description="Base delay in seconds between connection retries"
    )
    connection_exponential_backoff: bool = Field(
        default=True,
        description="Use exponential backoff for retry delays"
    )
    tool_name_separator: str = Field(
        default=":",
        description="Separator character for namespaced tool names"
    )


# ============================================================================
# Main Rollout Configuration
# ============================================================================

class RolloutConfig(BaseModel):
    """Main configuration for rollout execution parameters."""
    
    # Core LLM settings
    api_config: ApiConfig = Field(default_factory=ApiConfig, description="API provider configuration")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig, description="Text generation parameters")
    tool_calling_config: ToolCallingConfig = Field(default_factory=ToolCallingConfig, description="Tool calling parameters")
    
    # Execution limits
    seed: int = Field(
        default=42,
        description="Random seed for deterministic random number generation"
    )
    max_rounds: int = Field(
        default=5,
        ge=1,
        description="Maximum conversation rounds before termination"
    )
    max_total_completion_tokens: int = Field(
        default=8_000,
        gt=0,
        description="Total token limit across all rounds"
    )
    max_round_completion_tokens: int = Field(
        default=4_000,
        gt=0,
        description="Token limit per individual round"
    )
    tool_format: str = Field(
        default="agentfactory",
        description="Tool format to use for tool calling"
    )
    allow_tools_in_final_round: bool = Field(
        default=False,
        description="Whether to allow tool execution in the final round. If False, the last round will skip tool processing to ensure conversation ends with assistant response"
    )
    
    # Image processing settings
    image_max_pixels: int = Field(
        default=28 * 28 * 576,
        gt=0,
        description="Maximum image resolution in pixels"
    )
    image_min_pixels: int = Field(
        default=28 * 28 * 16,
        gt=0,
        description="Minimum image resolution in pixels"
    )
    image_format: str = Field(
        default="JPEG",
        description="Image file format for processing"
    )
    image_quality: int = Field(
        default=85,
        ge=1, le=100,
        description="Image compression quality (1-100)"
    )
    
    # Generation control
    assistant_prefix: str = Field(
        default="",
        description="Prefix added to assistant responses"
    )


# ============================================================================
# Request and Result Data Structures
# ============================================================================


class SingleRolloutRequest(BaseModel):
    """Complete request specification for a single rollout execution."""
    
    id: str = Field(description="Unique identifier for this rollout")
    messages: Messages = Field(description="Initial conversation messages")
    rollout_config: RolloutConfig = Field(default_factory=RolloutConfig, description="Execution configuration")
    
    # Optional components
    user_files: List[str] = Field(
        default_factory=list,
        description="Paths to user-provided files for processing"
    )
    image_do_save: bool = Field(
        default=False,
        description="Whether to save processed images to disk"
    )
    mcp_config: McpConfig = Field(
        default_factory=McpConfig,
        description="MCP server configuration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this rollout"
    )
    
    # Reward and evaluation settings
    reward_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Episode reward metric weights (metric_name -> weight)"
    )
    turn_reward_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Turn reward metric weights (metric_name -> weight)"
    )
    
    # Prompt configuration
    add_img_system_prompt: bool = Field(
        default=False,
        description="Include image handling instructions in system prompt"
    )
    call_mcp_prompts: bool = Field(
        default=True,
        description="Enable MCP-based prompt enhancement"
    )

class TokenStats(BaseModel):
    """Token usage statistics for a rollout."""
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens consumed (input + output)"
    )
    num_completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of generated output tokens"
    )
    total_token_until_last_assistant: Optional[int] = Field(
        default=None,
        description="Total tokens count until the last assistant message"
    )

class TaskDuration(BaseModel):
    """Task duration statistics."""
    task_name: str = Field(
        default="",
        description="Task name"
    )
    start_time: datetime = Field(
        default=datetime.now(),
        description="Start time of the task"
    )
    end_time: datetime = Field(
        default=datetime.now(),
        description="End time of the task"
    )
    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

class SingleRolloutResult(BaseModel):
    """Complete result data from a single rollout execution."""
    
    id: str = Field(description="Unique identifier matching the request")
    is_success: bool = Field(
        default=True,
        description="Whether execution completed successfully"
    )
    preprocessed_prompt_messages: Messages | None = Field(
        default=None,
        description="Preprocessed prompt messages",
    )
    messages: Messages = Field(
        default=Messages(root=[]),
        description="Full conversation history",
    )
    
    start_time: datetime = Field(
        default=datetime.now(),
        description="Start time of the rollout"
    )
    end_time: datetime = Field(
        default=datetime.now(),
        description="End time of the rollout"
    )
    @property
    def execution_time(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    error: str | None = Field(
        default=None,
        description="Error message if execution failed"
    )
    metadata: Dict[str, Any] | None = Field(
        default=None,
        description="Additional result metadata"
    )
    token_stats: TokenStats | None = Field(
        default=None,
        description="Token usage statistics"
    )
    task_records: List[TaskDuration] = Field(
        default_factory=list,
        description="Task duration records"
    )

    # Reward and evaluation results
    reward_components: Dict[str, float] | None = Field(
        default=None,
        description="Individual reward component scores"
    )
    weighted_reward: float | None = Field(
        default=None,
        description="Weighted sum of all reward components"
    )
    advantage: float | None = Field(
        default=None,
        description="Advantage value for reinforcement learning"
    )

    # ------- Token-Level Output Data -------
    input_ids: List[int] | None = Field(
        default=None,
        description="Tokenized input IDs for the generated sequence"
    )
    logprobs: List[float] | None = Field(
        default=None,
        description="Log probabilities corresponding to each token"
    )
    entropies: List[float] | None = Field(
        default=None,
        description="Entropy values for each token in the sequence"
    )


# ============================================================================
# Batch Processing
# ============================================================================

class BatchRolloutRequest(BaseModel):
    """Configuration for executing multiple rollouts in batch."""
    
    requests: List[SingleRolloutRequest] = Field(
        description="List of individual rollout requests to execute"
    )
    base_urls: List[str] = Field(
        description="API base URLs for load balancing across requests"
    )
    
    # Execution control
    concurrent_limit: int = Field(
        default=64,
        ge=1,
        description="Maximum number of concurrent rollout executions"
    )
    start_interval: float = Field(
        default=1.0,
        ge=0,
        description="Delay in seconds between starting new rollouts"
    )
    single_rollout_timeout: float = Field(
        default=600.0,
        gt=0,
        description="Timeout in seconds for individual rollouts"
    )
    
    # Persistence settings
    save_dir: Optional[str] = Field(
        default=None,
        description="Directory path for saving batch results"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="File path for execution logs"
    )
    base_seed: int = Field(
        default=59487,
        description="Base seed for deterministic random number generation"
    )


class BatchRolloutResult(BaseModel):
    """Aggregated results from a batch rollout execution."""
    
    results: List[SingleRolloutResult] = Field(
        description="Results from all individual rollouts"
    )
    start_time: datetime = Field(description="Batch execution start timestamp")
    end_time: datetime = Field(description="Batch execution end timestamp")