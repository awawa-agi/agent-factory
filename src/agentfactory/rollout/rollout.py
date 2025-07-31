import os
import json
import anyio
import random
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
import anyio.to_thread
from loguru import logger
from contextlib import AsyncExitStack, asynccontextmanager

from .core.config import (
    RolloutConfig, 
    SingleRolloutRequest, 
    SingleRolloutResult, 
    BatchRolloutRequest, 
    BatchRolloutResult,
    TokenStats,
    TaskDuration,
)
from .rewards_manager import RewardManager
from .mcp_client import MultiServerClientWithLimit
from .monitor import ExecutionMonitor
from .core.utils.message_processor import process_image_in_messages
from .rollout_session import create_rollout_session
from .core.utils.prompt_formatter import add_function_calling_system
from .tool_format import create_tool_format

@asynccontextmanager
async def async_temp_work_dir(prefix: str):
    """Create a temporary directory with proper permissions for rollout execution."""
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    try:
        os.chmod(temp_dir.name, 0o755)
        logger.debug(f"Created temporary directory: {temp_dir.name}")
        yield temp_dir.name
    except Exception as e:
        logger.error(f"Error with temporary directory {temp_dir.name}: {e}")
        raise
    finally:
        logger.debug(f"Cleaning up temporary directory: {temp_dir.name}")
        temp_dir.cleanup()

def render_config(config, variables):
    """Replace @VAR@ placeholders in a configuration object.
    
    Args:
        config: Configuration object to render
        variables: Dictionary of variable name-value pairs
    Returns:
        Rendered configuration object
    Raises:
        json.JSONDecodeError: If config cannot be serialized/deserialized
    """
    try:
        config_str = json.dumps(config)
        for var_name, value in variables.items():
            placeholder = f"@{var_name}@"
            config_str = config_str.replace(placeholder, str(value))
        return json.loads(config_str)
    except (TypeError, json.JSONDecodeError) as e:
        logger.error(f"Failed to render config with variables {variables}: {e}")
        raise

async def _setup_mcp_client(request: SingleRolloutRequest, tmp_dir: str, a_exit_stack: AsyncExitStack) -> MultiServerClientWithLimit:
    """Setup and configure MCP client with rendered server configs."""
    render_vars = {"MOUNT_DIR": tmp_dir}
    
    mcp_config = request.mcp_config
    mcp_config.server_configs = [render_config(cfg, render_vars) for cfg in mcp_config.server_configs]
    logger.debug(f"MCP server config: {mcp_config.server_configs}, tool limits: {mcp_config.tool_limits}")
    
    return await a_exit_stack.enter_async_context(MultiServerClientWithLimit(mcp_config))

async def _setup_tool_system_message(request: SingleRolloutRequest, mcp_client: MultiServerClientWithLimit, messages: List[Dict]) -> List[Dict]:
    """Setup tool system message and add it to the messages."""
    tool_definitions = await mcp_client.get_tool_definitions()
    tool_format = create_tool_format(request.rollout_config.tool_format)
    
    tool_system_message = tool_format.build_system_message(
        tool_definitions,
        max_calls_per_round=request.rollout_config.tool_calling_config.max_calls_per_round,
        include_image_instructions=request.add_img_system_prompt,
    )

    if messages[0].role == "system":
        messages[0] = messages[0].update_content(
            messages[0].content + [{"type": "text", "text": tool_system_message}]
        )
    else:
        system_msg = {"role": "system", "content": [{"type": "text", "text": tool_system_message}]}
        messages.insert(0, system_msg)
    
    return messages

async def _process_mcp_prompts(request: SingleRolloutRequest, mcp_client: MultiServerClientWithLimit, messages: List[Dict]) -> List[Dict]:
    """Process MCP prompts in user messages."""
    if not request.call_mcp_prompts:
        return messages
        
    for i, msg in enumerate(messages):
        if msg.role != 'user':
            continue

        new_content = []
        for content in msg.content:
            if content.type == 'mcp_prompt':
                mcp_content = await mcp_client.get_prompt(content.prompt_name, content.arguments)
                new_content.extend(mcp_content)
            else:
                new_content.append(content)

        messages[i] = msg.update_content(new_content)
    
    return messages

async def _process_images(request: SingleRolloutRequest, messages: List[Dict], tmp_dir: str) -> List[Dict]:
    """Process images in messages with resizing and saving options."""
    rollout_config = request.rollout_config
    
    def _process_images_sync():
        return process_image_in_messages(
            messages,
            do_resize=True,
            max_pixels=rollout_config.image_max_pixels,
            min_pixels=rollout_config.image_min_pixels,
            add_image_tag=True,
            do_save=request.image_do_save,
            local_save_dir=tmp_dir,
            display_save_dir='/mnt/data',
        )
    
    return await anyio.to_thread.run_sync(_process_images_sync)

async def _calculate_rewards(request: SingleRolloutRequest, rollout_result) -> tuple[float | None, Dict[str, Any], List[TaskDuration]]:
    """Calculate episode and turn rewards using unified reward system."""
    reward_components = {}
    weighted_reward = None
    task_records = []
    
    # Skip if no reward metrics specified
    if not request.reward_metrics and not request.turn_reward_metrics:
        return weighted_reward, reward_components, task_records
        
    logger.debug(f"Calculating reward metrics - episode: {request.reward_metrics}, turn: {request.turn_reward_metrics}")
    reward_start_time = datetime.now()
    
    # Validate reward functions exist
    all_function_names = set()
    if request.reward_metrics:
        all_function_names.update(metric.split('.')[0] for metric in request.reward_metrics.keys())
    if request.turn_reward_metrics:
        all_function_names.update(metric.split('.')[0] for metric in request.turn_reward_metrics.keys())
    
    available_rewards = RewardManager.list_available_rewards()
    for name in all_function_names:
        if name not in available_rewards:
            raise ValueError(f"Reward function '{name}' not found in registry")
    
    # Calculate unified rewards
    unified_result = await anyio.to_thread.run_sync(
        RewardManager.calculate_unified_rewards,
        rollout_result.messages,
        request.metadata,
        request.reward_metrics or {},
        request.turn_reward_metrics or {}
    )
    
    # Extract episode results
    weighted_reward = unified_result['episode']['weighted_reward']
    reward_components = unified_result['episode']['components']
    
    # Assign turn rewards to assistant messages
    turn_result = unified_result['turn']
    if turn_result['components']:
        from .core.messages import AssistantMessage
        assistant_idx = 0
        
        for msg in rollout_result.messages.root:
            if isinstance(msg, AssistantMessage):
                if assistant_idx < len(turn_result['components']):
                    msg.turn_reward_components = turn_result['components'][assistant_idx]
                    msg.weighted_turn_reward = turn_result['weighted_rewards'][assistant_idx]
                    assistant_idx += 1
    
    reward_end_time = datetime.now()
    task_records.append(TaskDuration(
        task_name="reward_calculation", 
        start_time=reward_start_time,
        end_time=reward_end_time
    ))
    
    logger.debug(f"Episode reward components: {reward_components}")
    logger.debug(f"Episode weighted reward: {weighted_reward}")
    if turn_result['components']:
        logger.debug(f"Turn rewards assigned to {len(turn_result['components'])} assistant messages")
    
    return weighted_reward, reward_components, task_records

def _create_rollout_result(request: SingleRolloutRequest, rollout_result, rollout_start_time: datetime, 
                          reward_components: Dict[str, Any], weighted_reward: float | None, 
                          reward_task_records: List[TaskDuration]) -> SingleRolloutResult:
    """Create the final SingleRolloutResult from all computed components."""
    all_task_records = rollout_result.task_records + reward_task_records
    
    return SingleRolloutResult(
        id=request.id,
        is_success=not rollout_result.end_reason in ("llm_generation_error", "tool_execution_error"),
        messages=rollout_result.messages,
        start_time=rollout_start_time,
        end_time=datetime.now(),
        metadata={
            "total_time": rollout_result.total_time,
            "end_reason": rollout_result.end_reason,
        },
        reward_components=reward_components,
        weighted_reward=weighted_reward,
        token_stats=TokenStats(
            total_tokens=rollout_result.total_tokens,
            num_completion_tokens=rollout_result.total_tokens_generated,
            total_token_until_last_assistant=rollout_result.total_token_until_last_assistant,
        ),
        task_records=all_task_records,
    )

# ------- Execute a Single Rollout -------
async def run_single_rollout(
    request: SingleRolloutRequest,
) -> SingleRolloutResult | None:
    """Execute a single rollout request with proper error handling and resource management."""
    logger.bind(ctx=f"rollout_{request.id}")
    rollout_start_time = datetime.now()
    
    async with AsyncExitStack() as a_exit_stack:
        tmp_dir = await a_exit_stack.enter_async_context(
            async_temp_work_dir(prefix=f"rollout_{request.id}_")
        )

        # Setup MCP client
        mcp_client = await _setup_mcp_client(request, tmp_dir, a_exit_stack)
        
        # Process messages
        messages = request.messages.model_copy(deep=True)
        messages = await _setup_tool_system_message(request, mcp_client, messages)
        messages = await _process_mcp_prompts(request, mcp_client, messages)
        messages = await _process_images(request, messages, tmp_dir)
            
        # Execute the session
        rollout_config = request.rollout_config
        tool_format = create_tool_format(rollout_config.tool_format)
        session = create_rollout_session(rollout_config, mcp_client, tool_format)
        
        logger.info(f"Start session: model={rollout_config.generation_config.model_name}, max_rounds={rollout_config.max_rounds}")
        rollout_result = await session.run(messages)

        # Calculate rewards
        weighted_reward, reward_components, reward_task_records = await _calculate_rewards(request, rollout_result)
                    
        return _create_rollout_result(
            request, rollout_result, rollout_start_time, 
            reward_components, weighted_reward, reward_task_records
        )

def _create_error_result(request_id: str, error_message: str) -> SingleRolloutResult:
    """Create a standardized error result for failed rollouts."""
    return SingleRolloutResult(
        id=request_id,
        is_success=False,
        error=error_message,
    )

async def _configure_request_for_execution(request: SingleRolloutRequest, idx: int, base_seed: int, 
                                          base_urls: List[str], monitor: ExecutionMonitor) -> str:
    """Configure request settings for execution including seed and URL assignment."""
    request.rollout_config = request.rollout_config.model_copy(update={"seed": idx + base_seed})
    
    if base_urls:
        url = await monitor.allocate_url(base_urls)
        request.rollout_config.api_config.base_url = url
        logger.debug(f"Request #{request.id} assigned to URL: {url}")
        return url
    else:
        return request.rollout_config.api_config.base_url

async def _execute_single_rollout_with_timeout(request: SingleRolloutRequest, timeout: float) -> SingleRolloutResult:
    """Execute a single rollout with timeout and error handling."""
    try:
        with anyio.fail_after(timeout):
            result = await run_single_rollout(request)
            return result or _create_error_result(request.id, "Rollout returned None")
    except TimeoutError:
        logger.error(f"Rollout #{request.id} timed out after {timeout} seconds")
        return _create_error_result(request.id, f"Rollout execution timeout after {timeout} seconds")
    except Exception as e:
        logger.exception(f"Rollout #{request.id} failed")
        return _create_error_result(request.id, str(e))

async def _save_batch_results(results: List[SingleRolloutResult], save_dir: str) -> None:
    """Save batch rollout results to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for result in results:
        req_id = result.id
        filename = f"rollout_{req_id}_{ts}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved result #{req_id} → {filepath}")

# ------- Execute Batch Rollouts -------
async def run_batch_rollouts_async(
    batch_request: BatchRolloutRequest,
) -> BatchRolloutResult:
    """
    Asynchronous version for executing a batch of rollout requests.

    Args:
        batch_request: BatchRolloutRequest containing all execution parameters
    Returns:
        BatchRolloutResult containing execution results and timing
    """
    start_time = datetime.now()
    
    requests = batch_request.requests
    base_urls = batch_request.base_urls
    concurrent_limit = batch_request.concurrent_limit
    start_interval = batch_request.start_interval
    single_rollout_timeout = batch_request.single_rollout_timeout
    save_dir = batch_request.save_dir
    
    limiter = anyio.CapacityLimiter(concurrent_limit)
    logger.bind(ctx="batch")
    logger.info(f"Starting batch request execution, total {len(requests)} requests, concurrent limit {concurrent_limit}, start interval {start_interval} seconds")
    logger.info(f"Using dynamic load balancing, available URLs: {len(base_urls)}")

    results: List[SingleRolloutResult | None] = [None] * len(requests)

    async def _task_wrapper(idx: int, request: SingleRolloutRequest, monitor: ExecutionMonitor):
        async with limiter:
            url = await _configure_request_for_execution(request, idx, batch_request.base_seed, base_urls, monitor)
            
            # Execute rollout with monitoring
            async with monitor.track(request.id, url):
                result = await _execute_single_rollout_with_timeout(request, single_rollout_timeout)
                results[idx] = result

            # Random sleep for resource release and load balancing
            await anyio.sleep(random.uniform(1.0, 3.0))

    async with ExecutionMonitor(total_tasks=len(requests)) as monitor_ctx:
        try:
            async with anyio.create_task_group() as tg:
                for i, req in enumerate(requests):
                    tg.start_soon(_task_wrapper, i, req, monitor_ctx)
                    if i < len(requests) - 1:
                        await anyio.sleep(start_interval)
        except (Exception, anyio.get_cancelled_exc_class()) as e:
            logger.warning(f"Batch rollout interrupted or error: {e}")
            raise
        
    valid_results: List[SingleRolloutResult] = [r for r in results if r is not None]
    
    # Save results if directory is provided
    if save_dir:
        await _save_batch_results(valid_results, save_dir)
    
    end_time = datetime.now()
    
    return BatchRolloutResult(
        results=valid_results,
        start_time=start_time,
        end_time=end_time
    )

def run_batch_rollouts(
    batch_request: BatchRolloutRequest,
) -> BatchRolloutResult:
    """
    Execute a batch of rollout requests synchronously.
    
    Args:
        batch_request: BatchRolloutRequest containing all execution parameters
    Returns:
        BatchRolloutResult containing execution results and timing
    Raises:
        RuntimeError: If called from within an async context
    """
    try:
        return anyio.run(
            lambda: run_batch_rollouts_async(batch_request)
        )
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "cannot be called from a running event loop" in error_msg or "asyncio.run() cannot be called" in error_msg:
            logger.error("Attempted to call sync version from async context")
            raise RuntimeError(
                "Cannot call sync version from async context. "
                "Use run_batch_rollouts_async() instead."
            ) from e
        else:
            logger.exception("Unexpected RuntimeError in batch rollout execution")
            raise