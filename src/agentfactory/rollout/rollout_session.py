import time
from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime
from typing import Optional
import anyio
from loguru import logger

from .core.config import RolloutConfig, TaskDuration
from .core.messages import Messages

from .mcp_client import MultiServerClientWithLimit
from .llm_provider import LLMProvider, GenerationStopReason, create_llm_provider

from .tool_format import ToolFormat
from .tool_executor import ToolExecutor

class RolloutEndReason(StrEnum):
    """Reasons for ending a rollout session"""
    MAX_ROUNDS = "max_rounds"                   # Reached the maximum number of tool-call rounds
    NO_TOOL_CALLS = "no_tool_calls"             # No tool call detected, session ends naturally
    MAX_TOKENS = "max_tokens"                   # Reached the maximum token limit
    LLM_END_TURN = "llm_end_turn"               # LLM ended the turn naturally
    LLM_GENERATION_ERROR = "llm_generation_error"  # LLM generation error
    TOOL_TIMEOUT = "tool_timeout"               # Tool-execution timeout
    TOOL_EXECUTION_ERROR = "tool_execution_error"  # Tool-execution error
    END_BY_TOOL = "end_by_tool"                     # Tool signalled to finish the rollout

@dataclass
class RolloutSessionResult:
    """Complete result of a rollout session"""
    messages: Messages
    total_time: float
    total_tool_time: float
    total_tokens_generated: int
    total_token_until_last_assistant: int
    total_tokens: Optional[int]
    end_reason: RolloutEndReason
    tool_calls_count: int
    task_records: list[TaskDuration] = field(default_factory=list)

@dataclass
class RolloutState:
    """Mutable state accumulated during a rollout session"""
    start_time: float = field(default_factory=time.time)
    total_tool_run_time: float = 0.0
    num_tokens_generated: int = 0
    num_tool_calls: int = 0
    end_reason: RolloutEndReason | None = None
    task_records: list[TaskDuration] = field(default_factory=list)

class RolloutSession:
    """Unified rollout session that supports different LLM providers"""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        mcp_client: MultiServerClientWithLimit,
        config: RolloutConfig,
        tool_format: ToolFormat,
    ):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client
        self.config = config
        self.tool_format = tool_format
        self.tool_executor = ToolExecutor(mcp_client, config)

    def calc_max_completion_tokens(self, rollout_state: RolloutState) -> int:
        """Compute the remaining tokens that can be generated"""
        remaining_tokens = max(0, self.config.max_total_completion_tokens - rollout_state.num_tokens_generated)
        return min(remaining_tokens, self.config.max_round_completion_tokens)

    def _record_task(self, rollout_state: RolloutState, task_name: str, start_time: datetime, end_time: datetime):
        """Record task duration"""
        task_record = TaskDuration(
            task_name=task_name,
            start_time=start_time,
            end_time=end_time
        )
        rollout_state.task_records.append(task_record)

    async def run(self, prompt_messages: Messages) -> RolloutSessionResult:
        """Run the full rollout loop"""
        logger.info(f"Start rollout session, max rounds: {self.config.max_rounds}")
        
        rollout_state = RolloutState()
        messages = prompt_messages.model_copy(deep=True)
        response_round = 0
        
        while response_round < self.config.max_rounds and rollout_state.end_reason is None:
            response_round += 1
            logger.debug(f"Start round {response_round}")
            
            # Phase 1: LLM Generation
            if not await self._handle_llm_generation(messages, rollout_state):
                break
                
            # Phase 2: Tool Processing  
            # Skip tool processing if this is the last round and tools are not allowed in final round
            if response_round == self.config.max_rounds and not self.config.allow_tools_in_final_round:
                logger.debug(
                    f"Reached max rounds {self.config.max_rounds}, skipping tool processing (allow_tools_in_final_round=False)"
                )
                rollout_state.end_reason = RolloutEndReason.MAX_ROUNDS
                break
                
            if not await self._handle_tool_processing(messages, rollout_state):
                break
            
            # If tools were executed in the final round, set end reason
            if response_round == self.config.max_rounds:
                logger.debug(f"Completed final round {response_round} with tool execution")
                rollout_state.end_reason = RolloutEndReason.MAX_ROUNDS
                break
        
        # Determine final end reason
        if rollout_state.end_reason is None:
            logger.warning("end_reason is None")
            rollout_state.end_reason = RolloutEndReason.MAX_ROUNDS
            
        # Build final result
        total_time = time.time() - rollout_state.start_time
        logger.success(
            f"Rollout finished: {rollout_state.end_reason}, rounds: {response_round}, "
            f"total time: {total_time:.2f}s, tool time: {rollout_state.total_tool_run_time:.2f}s, "
            f"tokens generated: {rollout_state.num_tokens_generated}, tool calls: {rollout_state.num_tool_calls}"
        )

        total_token_until_last_assistant = 0
        last_assistant_message = messages.last_assistant_message()
        if last_assistant_message:
            total_token_until_last_assistant = last_assistant_message.usage.total_tokens if last_assistant_message.usage else 0

        # Try to tokenize, but don't fail the entire rollout if it times out
        try:
            tokenize_result = await self.llm_provider.tokenize(messages)
            total_tokens = tokenize_result.count
        except Exception as e:
            logger.warning(f"Tokenize failed, setting total_tokens to None: {e}")
            total_tokens = None
        
        return RolloutSessionResult(
            messages=messages,
            total_time=total_time,
            total_tool_time=rollout_state.total_tool_run_time,
            total_tokens_generated=rollout_state.num_tokens_generated,
            total_token_until_last_assistant=total_token_until_last_assistant,
            total_tokens=total_tokens,
            end_reason=rollout_state.end_reason,
            tool_calls_count=rollout_state.num_tool_calls,
            task_records=rollout_state.task_records
        )

    async def _handle_llm_generation(self, messages: Messages, rollout_state: RolloutState) -> bool:
        """Handle the LLM-generation phase and return whether to continue"""
        start_time = datetime.now()
        try:
            max_completion_tokens = self.calc_max_completion_tokens(rollout_state)
            logger.debug(
                f"Tokens generated: {rollout_state.num_tokens_generated}, "
                f"remaining tokens: {max_completion_tokens}"
            )
            
            gen_config = self.config.generation_config.model_copy(
                update={
                    "seed": self.config.seed,
                    "stop_sequences": self.tool_format.stop_sequences + self.config.generation_config.stop_sequences,
                }
            )
            gen_result = await self.llm_provider.generate(
                messages,
                config=gen_config,
                max_tokens=max_completion_tokens,
                assistant_prefix=self.config.assistant_prefix,
            )
            
            end_time = datetime.now()
            self._record_task(rollout_state, "llm_generation", start_time, end_time)
            
            # Append assistant response to the conversation
            messages.append({
                "role": "assistant",
                "content": gen_result.content,
                "usage": gen_result.usage,
            })
            
            rollout_state.num_tokens_generated += gen_result.num_completion_tokens
            
            stop_reason_str = gen_result.stop_reason
            if stop_reason_str == "stop_sequences":
                stop_reason_str = f"{stop_reason_str}({gen_result.stop_sequence})"
            logger.debug(
                f"LLM generation finished, tokens: {gen_result.num_completion_tokens}, "
                f"stop reason: {stop_reason_str}"
            )
            
            # Check stopping conditions
            if gen_result.stop_reason == GenerationStopReason.MAX_TOKENS:
                rollout_state.end_reason = RolloutEndReason.MAX_TOKENS
                return False
            elif gen_result.stop_reason == GenerationStopReason.END_TURN:
                rollout_state.end_reason = RolloutEndReason.LLM_END_TURN
                return False
                
            return True
            
        except Exception as e:
            end_time = datetime.now()
            self._record_task(rollout_state, "llm_generation", start_time, end_time)
            logger.error(f"LLM generation failed: {e}")
            rollout_state.end_reason = RolloutEndReason.LLM_GENERATION_ERROR
            return False

    async def _handle_tool_processing(self, messages: Messages, rollout_state: RolloutState) -> bool:
        """Handle the tool-execution phase and return whether to continue"""
        start_time = datetime.now()
        tool_call_start_time = time.time()
        
        # Check overall tool-execution time limit
        if rollout_state.total_tool_run_time > self.config.tool_calling_config.total_call_timeout:
            logger.warning(
                f"Total tool time exceeded ({rollout_state.total_tool_run_time:.2f}s), ending session"
            )
            rollout_state.end_reason = RolloutEndReason.TOOL_TIMEOUT
            return False
        
        # Parse tool calls
        tool_calls = self.tool_format.parse_tool_calls(messages[-1].get_text())
        if not tool_calls:
            logger.debug("No tool call detected, ending session")
            rollout_state.end_reason = RolloutEndReason.NO_TOOL_CALLS
            return False
        
        logger.debug(f"Parsed {len(tool_calls)} tool call(s)")
        
        # Execute tool calls
        try:
            tool_output_messages = await self.tool_executor.execute(tool_calls)
            rollout_state.num_tool_calls += len(tool_output_messages)
            
            # Calculate tool-execution time
            tool_time_used = time.time() - tool_call_start_time
            rollout_state.total_tool_run_time += tool_time_used
            
            end_time = datetime.now()
            self._record_task(rollout_state, "tool_execution", start_time, end_time)
            
            # Post-process tool outputs and add to the conversation
            tool_output_messages = await anyio.to_thread.run_sync( # type: ignore
                self.tool_executor.postprocess_tool_outputs, tool_output_messages
            )
            messages.extend(tool_output_messages)
            
            if self.config.generation_config.stream_output:
                print(tool_output_messages.to_pretty_str())
            
            if tool_output_messages[-1].structured_output.get('done', False):
                logger.trace("Tool indicated to finish rollout")
                rollout_state.end_reason = RolloutEndReason.END_BY_TOOL
                return False
            
            return True
            
        except Exception as e:
            end_time = datetime.now()
            self._record_task(rollout_state, "tool_execution", start_time, end_time)
            logger.exception(f"Tool execution phase failed: {e}")
            tool_time_used = time.time() - tool_call_start_time
            rollout_state.total_tool_run_time += tool_time_used
            rollout_state.end_reason = RolloutEndReason.TOOL_EXECUTION_ERROR
            return False


def create_rollout_session(
    config: RolloutConfig,
    mcp_client: MultiServerClientWithLimit,
    tool_format: ToolFormat,
) -> RolloutSession:
    """Factory function for creating a RolloutSession"""
    
    # Create provider using factory function
    llm_provider = create_llm_provider(config.api_config)
    
    return RolloutSession(llm_provider, mcp_client, config, tool_format)