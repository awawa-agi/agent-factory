"""
Tool execution module for rollout system

Handles execution of tool calls and processing of outputs.
"""

import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import anyio
from loguru import logger

from .core.config import RolloutConfig
from .core.messages import Messages
from .mcp_client import MultiServerClientWithLimit
from .core.utils.message_processor import process_image_in_messages, truncate_overlong_text_in_messages


class ToolExecutor:
    """Execute tool calls with concurrency control and output processing"""
    
    def __init__(self, mcp_client: MultiServerClientWithLimit, config: RolloutConfig):
        self.mcp_client = mcp_client
        self.config = config
    
    # ========== Core Execution Methods ==========
    
    async def execute(self, tool_calls: List[Dict[str, Any]]) -> Messages:
        """
        Execute tool calls with concurrency control
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            Messages object with tool outputs
        """
        # Check tool call count limit
        if len(tool_calls) > self.config.tool_calling_config.max_calls_per_round:
            logger.info(f"Tool call count ({len(tool_calls)}) exceeds limit ({self.config.tool_calling_config.max_calls_per_round})")
            error_msg = (
                f"Tool call limit exceeded. Attempted to invoke {len(tool_calls)} tools, "
                f"but maximum allowed is {self.config.tool_calling_config.max_calls_per_round} tools per call. "
                f"Please reduce the number of tool invocations in your request."
            )
            return Messages.model_validate([
                {
                    "role": "tool",
                    "content": [{"type": "text", "text": error_msg}],
                    "tool_name": "tool_limit_exceeded",
                    "tool_args": {},
                }
            ])
        
        if not tool_calls:
            return Messages([])
        
        logger.debug(f"Executing {len(tool_calls)} tool calls (max parallel: {self.config.tool_calling_config.num_parallel_calls})")
        
        all_tool_outputs: List[Dict[str, Any]] = []
        semaphore = anyio.Semaphore(self.config.tool_calling_config.num_parallel_calls)
        
        async with anyio.create_task_group() as tg:
            async def call_single_tool(tool_name: str, tool_args: Dict[str, Any]):
                async with semaphore:
                    structured_output = {}
                    try:
                        with anyio.fail_after(self.config.tool_calling_config.single_call_timeout):
                            tool_outputs, structured_output = await self.mcp_client.call_tool(tool_name, tool_args)
                            logger.debug(f"Tool call completed: {tool_name}")

                    except TimeoutError:
                        logger.info(f"Tool {tool_name} execution timeout ({self.config.tool_calling_config.single_call_timeout}s)")
                        timeout_msg = (f"Tool '{tool_name}' execution timed out after "
                                     f"{self.config.tool_calling_config.single_call_timeout} seconds")
                        tool_outputs = [{"type": "text", "text": timeout_msg}]

                    except Exception as e:
                        logger.info(f"Tool {tool_name} execution failed: {e}")
                        error_msg = f"Tool {tool_name} failed: {str(e)}"
                        tool_outputs = [{"type": "text", "text": error_msg}]

                    all_tool_outputs.append({
                        "role": "tool",
                        "content": tool_outputs,
                        "tool_name": tool_name if self.mcp_client.is_tool_name_available(tool_name) else "unavailable_tool",
                        "tool_arguments": tool_args,
                        "structured_output": structured_output
                    })
            
            # Start all tool calls
            for tool_call in tool_calls:
                tg.start_soon(call_single_tool, tool_call["tool"], tool_call["arguments"])
        
        return Messages.model_validate(all_tool_outputs)
    
    def postprocess_tool_outputs(self, tool_outputs: Messages) -> Messages:
        """
        Process tool output including image processing and text truncation
        
        Args:
            tool_output: Raw tool output
            
        Returns:
            Processed Content object
        """
        tool_outputs.load_all_images()
    
        messages = process_image_in_messages(
            tool_outputs,
            do_resize=True,
            max_pixels=self.config.image_max_pixels,
            min_pixels=self.config.image_min_pixels,
            add_image_tag=True,
            do_save=False,
            local_save_dir=None,
            display_save_dir=None,
        )
        
        messages = truncate_overlong_text_in_messages(messages, self.config.tool_calling_config.output_max_length)
        
        return messages