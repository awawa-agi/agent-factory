import os
import json
import random
from datetime import datetime
from typing import List

import anyio
from loguru import logger

from ..rollout import (
    SingleRolloutRequest, 
    SingleRolloutResult, 
    ExecutionMonitor, 
    BatchRolloutResult,
    BatchRolloutRequest,
    run_single_rollout
)
from ..hparams.rollout_worker_config import RolloutWorkerConfig


class SimpleAsyncRolloutWorker:
    def __init__(self, config: RolloutWorkerConfig):
        self.config = config

    def run(self, batch_request: BatchRolloutRequest) -> BatchRolloutResult:
        """Synchronous interface for batch rollout execution."""
        return anyio.run(self.run_async_rollout, batch_request)

    async def run_async_rollout(self, batch_request: BatchRolloutRequest) -> BatchRolloutResult:
        start_time = datetime.now()
        
        logger.info(
            f"Starting batch request execution, total {len(batch_request.requests)} requests, "
            f"concurrent limit {batch_request.concurrent_limit}, "
            f"start interval {batch_request.start_interval} seconds"
        )
        if batch_request.base_urls:
            logger.info(f"Using dynamic load balancing, available URLs: {len(batch_request.base_urls)}")
        
        # Execute batch requests
        limiter = anyio.CapacityLimiter(batch_request.concurrent_limit)
        results: List[SingleRolloutResult | None] = [None] * len(batch_request.requests)
        
        async with ExecutionMonitor(total_tasks=len(batch_request.requests)) as monitor:
            async with anyio.create_task_group() as tg:
                for i, request in enumerate(batch_request.requests):
                    tg.start_soon(
                        self._execute_single_task, 
                        i, request, batch_request, limiter, monitor, results
                    )
                    await anyio.sleep(batch_request.start_interval)
            
            # Give a moment for any remaining async cleanup (httpx clients, etc.)
            await anyio.sleep(0.05)
        
        valid_results = [r for r in results if r is not None]
        
        # Save results if needed
        if batch_request.save_dir:
            self._save_results(valid_results, batch_request.save_dir)
        
        return BatchRolloutResult(
            results=valid_results,
            start_time=start_time,
            end_time=datetime.now()
        )
    
    async def _execute_single_task(
        self, 
        idx: int, 
        request: SingleRolloutRequest,
        batch_request: BatchRolloutRequest,
        limiter: anyio.CapacityLimiter,
        monitor: ExecutionMonitor,
        results: List[SingleRolloutResult | None]
    ) -> None:
        async with limiter:
            request.rollout_config.seed = idx + batch_request.base_seed
            
            # Assign URL for load balancing
            if batch_request.base_urls:
                url = await monitor.allocate_url(batch_request.base_urls)
                request.rollout_config.api_config.base_url = url
                logger.debug(f"Request #{request.id} assigned to URL: {url}")
            else:
                url = request.rollout_config.api_config.base_url
            
            # Execute with timeout and error handling
            async with monitor.track(request.id, url):
                try:
                    with anyio.fail_after(batch_request.single_rollout_timeout):
                        result = await run_single_rollout(request)
                except TimeoutError:
                    logger.error(f"Rollout #{request.id} timed out after {batch_request.single_rollout_timeout} seconds")
                    result = SingleRolloutResult(
                        id=request.id,
                        is_success=False,
                        error=f"Rollout execution timeout after {batch_request.single_rollout_timeout} seconds",
                    )
                except Exception as e:
                    logger.exception(f"Rollout #{request.id} failed")
                    result = SingleRolloutResult(
                        id=request.id,
                        is_success=False,
                        error=str(e),
                    )
                
                results[idx] = result
            
            await anyio.sleep(random.uniform(1.0, 3.0))  # Load balancing delay
    
    def _save_results(self, results: List[SingleRolloutResult], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for result in results:
            filename = f"rollout_{result.id}_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved result #{result.id} → {filepath}")