# pyright: reportTypeCommentUsage=false
import os
import sys
import json
import numpy as np
import tempfile
import importlib
from tqdm import tqdm   
from typing import Any
from pathlib import Path
from loguru import logger
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from ..hparams import BasicFlowConfig
from ..inference_engine import create_inference_engine
from ..trainer.agent_trainer import AgentTrainer, TrainSample
from ..trainer.cpu_barrier import CpuBarrier
from ..rollout_worker.simple_rollout_worker import SimpleAsyncRolloutWorker
from ..rollout import BatchRolloutResult, SingleRolloutResult
from ..trainer.advantage_estimator import AdvantageEstimator
from ..profiling import ProfilerManager
from ..metrics import MetricsManager
from concurrent.futures import ThreadPoolExecutor
import atexit

class BasicFlow:
    def __init__(
        self,
        config: BasicFlowConfig,
    ):
        self.config = config
        self._iteration = 0

        # Load plugins before any other component initialization
        self._load_plugins(config.plugins)

        self.set_up_logger()
        self.accelerator = Accelerator(
            log_with=self.config.report_to, # type: ignore
            project_dir=self.config.save_dir,
        )

        if self.accelerator.is_main_process:
            config_dict = self.config.model_dump()
            self.accelerator.init_trackers(
                project_name=self.config.project_name,
                config=config_dict,
                init_kwargs={
                    "wandb": {
                        "name": self.config.run_name,
                        "tags": ["agentfactory", self.config.run_name],
                    }
                }
            )
            logger.info(f"AgentFactory config: {self.config.model_dump_json(indent=2)}")

        self._setup_directories()
        if self.accelerator.is_main_process:
            self.inference_engine = create_inference_engine(
                config=self.config.inference_engine,
            )

        self._load_data()

        self.accelerator.wait_for_everyone()
        self.trainer = AgentTrainer(
            config=self.config.trainer,
            accelerator=self.accelerator,
        )
        if self.accelerator.is_main_process:
            logger.info(f"Model initialized. Model structure:\n{self.trainer.model}")

        self.rollout_worker = SimpleAsyncRolloutWorker(
            config=self.config.rollout_worker,
        )
        self.advantage_estimator = AdvantageEstimator(
            config=self.config.algorithm,
        )
        
        # Initialize metrics manager
        self.metrics_manager = MetricsManager()
        
        # Initialize profiler manager if configured (main process only)
        self.profiler_manager = None
        if self.accelerator.is_main_process and self.config.profiler:
            self.profiler_manager = ProfilerManager(self.config.profiler)
            logger.info(f"Initialized {self.config.profiler} profiler manager")
        
        # Initialize async file saver (main process only)
        self._file_saver = None
        if self.accelerator.is_main_process:
            self._file_saver = ThreadPoolExecutor(max_workers=2, thread_name_prefix="file-saver")
            atexit.register(self._cleanup_file_saver)

    def run(self):
        pbar = tqdm(range(self.config.num_iterations), desc="Training", disable=not self.accelerator.is_main_process)
        while self._iteration < self.config.num_iterations:
            pbar.n = self._iteration
            pbar.set_description(f"Training iteration {self._iteration}")
            pbar.refresh()

            # Start profiling for this iteration (main process only)
            profiling_started = False
            if self.profiler_manager:
                profiling_started = self.profiler_manager.start_profiling()

            # Run evaluation if configured and due (before training)
            with CpuBarrier(self.accelerator):
                if self.accelerator.is_main_process and self._should_run_evaluation(self._iteration):
                    self.run_evaluation(self._iteration)

            self._sync_model_weights()

            with CpuBarrier(self.accelerator):
                if self.accelerator.is_main_process:
                    batch_request = self.get_batch_requests(self._iteration)
                    batch_rollout_result = self.rollout_worker.run(batch_request)
                    self._estimate_advantages(batch_rollout_result)
                    self.save_and_log_rollout_results(batch_rollout_result)
                    train_batch = self._prepare_train_batch(batch_rollout_result)
                    logger.info(f"Iteration {self._iteration} num filtered train samples: {len(train_batch)}")
                    train_batch = [train_batch]

                    self.inference_engine.sleep()
                else:
                    train_batch = [None]

            train_batch = broadcast_object_list(train_batch, from_process=0)
            train_batch = train_batch[0]

            metrics, per_token_logs = self.trainer.train_on_batch(train_batch)
            
            # Add assistant token statistics using new metrics manager
            if per_token_logs:
                assistant_token_stats = self.metrics_manager.compute_assistant_token_metrics(per_token_logs)
                metrics.update(assistant_token_stats)
            
            self.accelerator.log(metrics, step=self._iteration)

            if self.accelerator.is_main_process:
                # Async save per token logs using numpy format for efficiency
                if per_token_logs:
                    self._async_save(self._save_per_token_logs, per_token_logs)
                
                # Generate and upload token visualizer HTML
                if "wandb" in self.config.report_to and per_token_logs:
                    self._upload_token_visualizer(per_token_logs)
                
                # Stop profiling and async save/upload results
                if profiling_started and self.profiler_manager:
                    # Get HTML and cleanup profiler instance immediately
                    html_content = self.profiler_manager.stop_and_get_results()
                    if html_content:
                        profiler_name = self.profiler_manager.get_profiler_name()
                        profile_file = self.rollout_save_dir / f"profile_{profiler_name}_iter{self._iteration}.html"
                        current_step = self._iteration
                        self._async_save(self._save_profile_data, profile_file, html_content, profiler_name, current_step)
            
            self._iteration += 1

    def _load_data(self):
        train_dataset_info = self.config.data.train
        
        if train_dataset_info.hf_hub_url is not None:
            self.train_data = load_dataset(train_dataset_info.hf_hub_url, split=train_dataset_info.split)
            # Shuffle the training data for better training dynamics
            self.train_data = self.train_data.shuffle(seed=self.config.seed)
            logger.info(f"Shuffled training data with seed {self.config.seed}")
        elif train_dataset_info.file_name is not None:
            with open(train_dataset_info.file_name, "r") as f:
                self.train_data = json.load(f)
            # Shuffle list-based data
            import random
            random.seed(self.config.seed)
            random.shuffle(self.train_data)
            logger.info(f"Shuffled training data (list) with seed {self.config.seed}")
        else:
            raise ValueError("No data source provided")

        self.eval_data = {}
        for eval_dataset_info in self.config.data.eval:
            if eval_dataset_info.hf_hub_url is not None:
                self.eval_data[eval_dataset_info.name] = load_dataset(eval_dataset_info.hf_hub_url, split=eval_dataset_info.split)
            elif eval_dataset_info.file_name is not None:
                with open(eval_dataset_info.file_name, "r") as f:
                    self.eval_data[eval_dataset_info.name] = json.load(f)
            else:
                raise ValueError("No data source provided")
        

    def get_batch_requests(self, iteration: int):
        from agentfactory.rollout import SingleRolloutRequest, BatchRolloutRequest

        batch_size = self.config.data.n_prompts_per_iteration
        if isinstance(self.train_data, Dataset):
            batch_prompts = self.train_data.select(range(iteration * batch_size, (iteration + 1) * batch_size))
        else:
            batch_prompts = self.train_data[iteration * batch_size: (iteration + 1) * batch_size]

        batch_requests = []
        for i_prompt, prompt_d in enumerate(batch_prompts):
            for i_gen in range(self.config.data.n_rollouts_per_prompt):
                rollout_config = self.config.default_rollout_config.model_copy()
                if 'rollout_config' in prompt_d:
                    rollout_config = rollout_config.model_copy(update=prompt_d['rollout_config'], deep=True)

                prompt_id = prompt_d.get('id', f"prompt{i_prompt}")
                req = SingleRolloutRequest(
                    id=f"{prompt_id}_gen{i_gen}",
                    messages=prompt_d['messages'],
                    mcp_config=prompt_d['mcp_config'],
                    metadata=prompt_d.get('metadata', {}),
                    reward_metrics=prompt_d.get('reward_metrics', {}),
                    turn_reward_metrics=prompt_d.get('turn_reward_metrics', {}),
                    rollout_config=rollout_config,
                )
                req.rollout_config.generation_config = req.rollout_config.generation_config.model_copy(update={"model_name": "train_lora"})
                batch_requests.append(req)

        logger.info(f"Start rollout with {len(batch_requests)} requests")

        rollout_save_dir = self.rollout_save_dir / f"step_{self._iteration}"
        trace_file_path = rollout_save_dir / "rollout_trace.log"
        rollout_save_dir.mkdir(parents=True, exist_ok=True)

        infer_api_urls = [
            url.rstrip("/") + "/v1" if not url.rstrip("/").endswith("/v1") else url.rstrip("/")
            for url in self.config.inference_engine.urls
        ]

        batch_request = BatchRolloutRequest(
            requests=batch_requests,
            base_urls=infer_api_urls,
            concurrent_limit=self.config.rollout_worker.num_concurrent_rollouts,
            start_interval=self.config.rollout_worker.rollout_start_interval,
            single_rollout_timeout=self.config.rollout_worker.single_rollout_timeout, # type: ignore
            save_dir=str(rollout_save_dir),
            log_file=str(trace_file_path),
        )

        return batch_request

    def _should_run_evaluation(self, iteration: int) -> bool:
        """Check if evaluation should run at current iteration"""
        if self.config.evaluation.steps is None:
            return False
        
        # Run on first step if configured
        if iteration == 0 and self.config.evaluation.on_first_step:
            return True
            
        # Run based on evaluation frequency
        if self.config.evaluation.steps > 0 and iteration % self.config.evaluation.steps == 0:
            return True
            
        return False

    def _get_eval_batch_requests(self, eval_name: str, eval_data, iteration: int):
        """Generate batch requests for evaluation dataset"""
        from agentfactory.rollout import SingleRolloutRequest, BatchRolloutRequest
        
        # Use evaluation-specific rollout worker config
        eval_rollout_worker_config = self.config.evaluation.rollout_worker
        
        # Generate requests for all evaluation data (no batching for eval)
        batch_requests = []
        for i_prompt, prompt_d in enumerate(eval_data):
            # Single rollout per prompt for evaluation (can be made configurable)
            # Use evaluation-specific rollout config if provided, otherwise use default
            if self.config.evaluation.rollout_config is not None:
                rollout_config = self.config.evaluation.rollout_config.model_copy()
            else:
                rollout_config = self.config.default_rollout_config.model_copy()
                
            # Allow per-prompt rollout config overrides
            if 'rollout_config' in prompt_d:
                rollout_config = rollout_config.model_copy(update=prompt_d['rollout_config'], deep=True)

            prompt_id = prompt_d.get('id', f"eval_{eval_name}_prompt{i_prompt}")
            req = SingleRolloutRequest(
                id=f"{prompt_id}_eval",
                messages=prompt_d['messages'],
                mcp_config=prompt_d['mcp_config'],
                metadata=prompt_d.get('metadata', {}),
                reward_metrics=prompt_d.get('reward_metrics', {}),
                turn_reward_metrics=prompt_d.get('turn_reward_metrics', {}),
                rollout_config=rollout_config,
            )
            # Use train_lora model for evaluation
            req.rollout_config.generation_config = req.rollout_config.generation_config.model_copy(
                update={"model_name": "train_lora"}
            )
            batch_requests.append(req)

        logger.info(f"Start evaluation rollout for {eval_name} with {len(batch_requests)} requests")

        # Setup evaluation save directory
        eval_save_dir = self.rollout_save_dir / f"step_{iteration}" / f"eval_{eval_name}"
        trace_file_path = eval_save_dir / "rollout_trace.log"
        eval_save_dir.mkdir(parents=True, exist_ok=True)

        infer_api_urls = [
            url.rstrip("/") + "/v1" if not url.rstrip("/").endswith("/v1") else url.rstrip("/")
            for url in self.config.inference_engine.urls
        ]

        batch_request = BatchRolloutRequest(
            requests=batch_requests,
            base_urls=infer_api_urls,
            concurrent_limit=eval_rollout_worker_config.num_concurrent_rollouts,
            start_interval=eval_rollout_worker_config.rollout_start_interval,
            single_rollout_timeout=eval_rollout_worker_config.single_rollout_timeout,
            save_dir=str(eval_save_dir),
            log_file=str(trace_file_path),
        )

        return batch_request

    def _save_and_log_eval_results(self, batch_rollout_result: BatchRolloutResult, eval_name: str, iteration: int):
        """Save and log evaluation results with eval-specific prefixes"""
        # Save evaluation results to file
        eval_save_dir = self.rollout_save_dir / f"step_{iteration}" / f"eval_{eval_name}"
        save_path = eval_save_dir / f"eval_results_{eval_name}_iter{iteration}.json"
        with open(save_path, "w") as file:
            file.write(batch_rollout_result.model_dump_json())
        logger.info(f"Evaluation results saved to {save_path}")

        rollout_results = batch_rollout_result.results

        # Compute evaluation metrics with eval prefix
        eval_metrics = self.metrics_manager.compute_rollout_metrics(rollout_results)
        # Add eval prefix to all metrics
        prefixed_metrics = {f"eval_{eval_name}/{k}": v for k, v in eval_metrics.items()}
        self.accelerator.log(prefixed_metrics, step=iteration)

        # Log to wandb if configured
        if "wandb" in self.config.report_to:
            import wandb
            import random
            from ..visualizer import multiple_results_to_html, grouped_results_to_html
            
            if wandb.run is not None:
                # Log evaluation reward histogram
                histogram_data = self.metrics_manager.prepare_prompt_reward_histogram_data(rollout_results)
                if histogram_data:
                    wandb.log({f"eval_{eval_name}/rewards_histogram": wandb.Histogram(histogram_data)}, step=iteration)
                
                # Log evaluation visualization (sample some results)
                num_samples = min(self.config.num_rollouts_to_log, len(rollout_results))
                log_results = random.sample(rollout_results, num_samples) if len(rollout_results) > num_samples else rollout_results
                log_results = sorted(log_results, key=lambda x: x.id)
                
                if self.config.html_visualization_mode == "individual":
                    eval_html = multiple_results_to_html(
                        results=log_results,
                        title=f"Evaluation {eval_name} - Step {iteration}",
                    )
                    wandb.log({f"eval_{eval_name}/visualization": wandb.Html(eval_html, inject=False)}, step=iteration)
                elif self.config.html_visualization_mode == "grouped":
                    grouped_results = self._group_sampled_results(log_results)
                    eval_html = grouped_results_to_html(
                        grouped_results=grouped_results,
                        title=f"Evaluation {eval_name} - Step {iteration}",
                    )
                    wandb.log({f"eval_{eval_name}/visualization": wandb.Html(eval_html, inject=False)}, step=iteration)

    def run_evaluation(self, iteration: int):
        """Run evaluation on all configured eval datasets"""
        if not self.eval_data:
            logger.warning("No evaluation datasets configured, skipping evaluation")
            return
            
        logger.info(f"Running evaluation at iteration {iteration}")
        
        # Wake up inference engine for evaluation (it was put to sleep after training rollout)
        try:
            self.inference_engine.wake_up()
        except Exception as e:
            raise RuntimeError(f"Failed to wake up inference engine for evaluation at iteration {iteration}: {e}") from e
        
        for eval_name, eval_dataset in self.eval_data.items():
            logger.info(f"Evaluating on dataset: {eval_name}")
            
            # Generate evaluation batch requests
            batch_request = self._get_eval_batch_requests(eval_name, eval_dataset, iteration)
            
            # Run evaluation rollout
            batch_rollout_result = self.rollout_worker.run(batch_request)
            
            # Estimate advantages (reuse existing logic)
            self._estimate_advantages(batch_rollout_result)
            
            # Save and log evaluation results
            self._save_and_log_eval_results(batch_rollout_result, eval_name, iteration)
        
        # Put inference engine back to sleep after evaluation
        self.inference_engine.sleep()
        
        logger.info(f"Evaluation completed at iteration {iteration}")

    def _sync_model_weights(self):
        self.trainer.save_model(self.sync_weight_dir)

        if self.accelerator.is_main_process:
            # Wake up inference engine before loading weights if it was sleeping
            self.inference_engine.wake_up()
            self.inference_engine.load_lora_weights_from_disk(self.sync_weight_dir, name="train_lora")

    def _estimate_advantages(self, batch_rollout_result: BatchRolloutResult):
        result_groups: dict[str, list[SingleRolloutResult]] = {}

        for result in batch_rollout_result.results:
            prompt_id = result.id.rsplit("_", 1)[0]
            if prompt_id not in result_groups:
                result_groups[prompt_id] = []
            result_groups[prompt_id].append(result)

        self.advantage_estimator.add_advantages_to_results(result_groups)

    def save_and_log_rollout_results(self, batch_rollout_result: BatchRolloutResult):
        save_path = Path(self.rollout_save_dir) / f"rollout_results_iter{self._iteration}.json"
        with open(save_path, "w") as file:
            file.write(batch_rollout_result.model_dump_json())
        logger.debug(f"Rollout results saved to {save_path}!")

        rollout_results = batch_rollout_result.results

        # Log rollout results using new metrics manager
        rollout_metrics = self.metrics_manager.compute_rollout_metrics(rollout_results)
        self.accelerator.log(rollout_metrics, step=self._iteration)

        if "wandb" in self.config.report_to:
            import wandb
            import random
            from ..visualizer import multiple_results_to_html, grouped_results_to_html
            from ..visualizer.token_visualizer import generate_token_visualizer_html
            if wandb.run is not None:
                # Log per-prompt average reward histogram using new metrics manager
                histogram_data = self.metrics_manager.prepare_prompt_reward_histogram_data(rollout_results)
                if histogram_data:
                    wandb.log({"rewards/per_prompt_avg_histogram": wandb.Histogram(histogram_data)}, step=self._iteration)
                logger.debug(f"Logging rollout visualization to wandb")
                
                # Shared sampling strategy: sample first, then group if needed
                num_samples = min(self.config.num_rollouts_to_log, len(rollout_results))
                log_results = random.sample(rollout_results, num_samples)
                log_results = sorted(log_results, key=lambda x: x.id)
                
                # Choose visualization mode based on config
                if self.config.html_visualization_mode == "individual":
                    # Use existing individual mode
                    big_html = multiple_results_to_html(
                        results=log_results,
                        title=f"Step {self._iteration}",
                    )
                    wandb.log({"Rollout Visualization": wandb.Html(big_html, inject=False)}, step=self._iteration)
                    
                elif self.config.html_visualization_mode == "grouped":
                    # Group the sampled results by prompt_id
                    grouped_results = self._group_sampled_results(log_results)
                    grouped_html = grouped_results_to_html(
                        grouped_results=grouped_results,
                        title=f"Grouped Rollouts - Step {self._iteration}",
                    )
                    wandb.log({"Rollout Visualization": wandb.Html(grouped_html, inject=False)}, step=self._iteration)

    def _prepare_train_batch(self, batch_rollout_result: BatchRolloutResult) -> list[TrainSample]:
        train_samples = []
        for result in batch_rollout_result.results:
            if result.advantage is None or result.advantage == 0:
                continue

            messages = result.messages.to_openai_format()
            
            # Only keep messages up to and including the last assistant message,
            # as any messages after the last assistant response are irrelevant for training.
            # This ensures the training sample reflects the actual completion generated by the model.
            assistant_indices = [i for i, msg in enumerate(messages) if msg['role'] == 'assistant']
            if assistant_indices:
                messages = messages[:assistant_indices[-1] + 1]

            # Create base train sample
            train_sample = TrainSample(
                id=result.id,
                messages=messages,
                advantage=result.advantage,
                num_loss_tokens=result.token_stats.num_completion_tokens,
                num_tokens=result.token_stats.total_token_until_last_assistant,
            )
            
            # Add turn-level advantages for EMT-GRPO mode
            if (self.config.algorithm.adv_estimator == "emt_grpo" and 
                result.metadata and 'emt_grpo_advantages' in result.metadata):
                train_sample['turn_advantages'] = result.metadata['emt_grpo_advantages']
                # logger.trace(f"Added turn advantages to sample {result.id}: {len(train_sample['turn_advantages'])} turns")
            
            train_samples.append(train_sample)
            
        return train_samples
            
    def _setup_directories(self):
        """Setup all necessary directories for training workflow."""
        
        if self.accelerator.is_main_process:
            # ------- Main directories setup -------
            # Create base save directory
            save_path = Path(self.config.save_dir) / self.config.run_name
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f'Main save directory: {save_path}')
            
            # Create rollout logs directory
            self.rollout_save_dir = save_path / "rollout_logs"
            self.rollout_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Rollout logs directory: {self.rollout_save_dir}')
            
            # Setup weight temporary directory
            if self.config.tmp_weight_dir is None:
                self.sync_weight_dir = Path(tempfile.mkdtemp())
            else:
                self.sync_weight_dir = Path(self.config.tmp_weight_dir)
                self.sync_weight_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Weight temporary directory: {self.sync_weight_dir}')
            
        else:
            self.sync_weight_dir = None

    def _load_plugins(self, plugins: list[str]) -> None:
        """Load plugin modules from file paths to register custom components.
        
        Args:
            plugins: List of file paths to import (e.g., ['examples/alfworld/rewards.py'])
        """
        if not plugins:
            logger.debug("No plugins specified, skipping plugin loading")
            return
            
        logger.info(f"Loading {len(plugins)} plugin(s): {plugins}")
        
        import importlib.util
        from pathlib import Path
        
        # Get current working directory (where the training is running from)
        project_root = Path.cwd()
        
        for plugin_path in plugins:
            try:
                logger.debug(f"Loading plugin from path: {plugin_path}")
                
                # Convert relative path to absolute path
                if not plugin_path.endswith('.py'):
                    plugin_path += '.py'
                
                file_path = project_root / plugin_path
                
                if not file_path.exists():
                    logger.error(f"❌ Plugin file not found: {file_path}")
                    continue
                
                # Create a module name from the path
                module_name = plugin_path.replace('/', '.').replace('.py', '')
                
                # Load the module using spec
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    logger.error(f"❌ Failed to create spec for {plugin_path}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                logger.success(f"✅ Successfully loaded plugin: {plugin_path}")
                
            except Exception as e:
                logger.error(f"❌ Error loading plugin '{plugin_path}': {e}")
                # Continue loading other plugins even if one fails
        
        logger.info("Plugin loading completed")

    def set_up_logger(self):
        level_per_module: dict[str | None, str | int | bool] = {
            "": "WARNING",
            "agentfactory.training_flows": "TRACE",
            "agentfactory.trainer": "TRACE",
            "agentfactory.rollout": "WARNING",
            "agentfactory.rollout.monitor": "WARNING",
        }
        logger.remove()
        logger.add(sys.stderr, filter=level_per_module, level="TRACE")
        

    def _group_sampled_results(self, sampled_results: list[SingleRolloutResult]) -> dict[str, list[SingleRolloutResult]]:
        """Group sampled results by prompt_id"""
        groups = {}
        for result in sampled_results:
            # Extract prompt_id using the same logic as in _estimate_advantages
            prompt_id = result.id.rsplit("_", 1)[0] if "_" in result.id else result.id
            if prompt_id not in groups:
                groups[prompt_id] = []
            groups[prompt_id].append(result)
        
        # Sort results within each group by id for consistent ordering
        for group_results in groups.values():
            group_results.sort(key=lambda r: r.id)
        
        return groups
        
    def _save_per_token_logs(self, per_token_logs: list) -> None:
        """Save per token logs using efficient numpy format"""
        try:
            save_path = self.rollout_save_dir / f"per_token_logs_iter{self._iteration}.npz"
            
            # Convert to numpy arrays for efficient storage
            data_to_save = {}
            for i, log_entry in enumerate(per_token_logs):
                prefix = f"sample_{i}"
                data_to_save[f"{prefix}_train_sample_id"] = log_entry['train_sample_id']
                data_to_save[f"{prefix}_input_ids"] = log_entry['input_ids']
                data_to_save[f"{prefix}_logps"] = log_entry['logps']
                data_to_save[f"{prefix}_entropies"] = log_entry['entropies'] 
                data_to_save[f"{prefix}_assistant_masks"] = log_entry['assistant_masks']
                data_to_save[f"{prefix}_advantages"] = log_entry['advantages']
            
            np.savez_compressed(save_path, **data_to_save)
            logger.info(f"[Iter {self._iteration}] Saved token logs: {save_path.name} ({len(per_token_logs)} samples)")
            
            # Also save a JSON version for easy inspection (with reduced precision)
            json_path = self.rollout_save_dir / f"per_token_logs_iter{self._iteration}.json"
            json_data = []
            for log_entry in per_token_logs:
                json_entry = {
                    'train_sample_id': log_entry['train_sample_id'],
                    'input_ids': log_entry['input_ids'].tolist(),
                    'logps': np.round(log_entry['logps'], 4).tolist(),
                    'entropies': np.round(log_entry['entropies'], 4).tolist(),
                    'assistant_masks': log_entry['assistant_masks'].tolist(),
                    'advantages': np.round(log_entry['advantages'], 4).tolist(),
                }
                json_data.append(json_entry)
            
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"[Iter {self._iteration}] Failed to save token logs: {e}")
        
    
    def _upload_token_visualizer(self, per_token_logs: list) -> None:
        """Generate token visualizer HTML and upload to wandb using numpy arrays"""
        try:
            import wandb
            from ..visualizer.token_visualizer import generate_token_visualizer_html
            from ..visualizer_new import create_token_app
            
            if not wandb.run or not per_token_logs:
                return
                
            # Convert per token logs to visualizer format
            sequences_data = []
            processing_class = self.trainer.processing_class
            
            for log_entry in per_token_logs:
                input_ids = log_entry['input_ids']
                logps = log_entry['logps']
                entropies = log_entry['entropies']
                advantages = log_entry['advantages']
                assistant_masks = log_entry['assistant_masks']
                train_sample_id = log_entry['train_sample_id']
                
                # Convert token IDs to tokens
                if hasattr(processing_class, 'tokenizer'):
                    tokens = processing_class.tokenizer.batch_decode(input_ids.tolist())
                elif hasattr(processing_class, 'batch_decode'):
                    tokens = processing_class.batch_decode(input_ids.tolist())
                else:
                    tokens = [str(token_id) for token_id in input_ids.tolist()]
                
                if tokens and len(logps) > 0:
                    sequences_data.append({
                        'tokens': tokens,
                        'logprobs': logps.tolist(),
                        'entropies': entropies.tolist(),
                        'advantages': advantages.tolist(),
                        'assistant_masks': assistant_masks.tolist(),
                        'display_id': train_sample_id,
                    })
            
            if sequences_data:
                # Limit to avoid too large HTML using the config parameter
                max_sequences = min(self.config.num_token_logs_to_upload, len(sequences_data))
                sequences_data = sequences_data[:max_sequences]
                
                # Try new optimized visualizer first, fallback to legacy
                try:
                    html_content = create_token_app(
                        token_data=sequences_data,
                        title=f"🌸 Token Analysis - Step {self._iteration} 🌸"
                    )
                    logger.debug(f"Using optimized token visualizer for {len(sequences_data)} sequences")
                except Exception as e:
                    logger.warning(f"Optimized visualizer failed, using legacy: {e}")
                    html_content = generate_token_visualizer_html(
                        sequences_data,
                        title=f"Token Analysis - Step {self._iteration}",
                        collapse_min_length=3
                    )
                
                wandb.log({"Token Visualizer": wandb.Html(html_content, inject=False)}, step=self._iteration)
                logger.debug(f"Uploaded token visualizer for {len(sequences_data)} sequences to wandb")
                
        except Exception as e:
            logger.warning(f"Failed to upload token visualizer: {e}")
    
    def _save_profile_data(self, profile_file: Path, html_content: str, profiler_name: str, step: int) -> None:
        """Save profile HTML data to file and upload to W&B (called in background thread)."""
        try:
            # Save HTML profile to file
            profile_file.parent.mkdir(parents=True, exist_ok=True)
            with open(profile_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size_mb = profile_file.stat().st_size / (1024 * 1024)
            logger.info(f"[Iter {step}] Saved {profiler_name} profile: {profile_file.name} ({file_size_mb:.2f}MB)")
            
            # Upload to W&B if configured
            if "wandb" in self.config.report_to:
                import wandb
                if wandb.run is not None:
                    # Upload as file with proper base_path to preserve directory structure
                    wandb.save(str(profile_file), base_path=str(self.rollout_save_dir.parent))
                    logger.info(f"[Iter {step}] Uploaded {profiler_name} profile file to W&B: {profile_file.name}")
                    
        except Exception as e:
            logger.error(f"[Iter {step}] Failed to save/upload {profiler_name} profile: {e}")
    
    def _async_save(self, save_func, *args, **kwargs):
        """Submit file save task to background thread pool."""
        if self._file_saver:
            def log_result(future):
                try:
                    future.result()  # Check for exceptions
                except Exception as e:
                    logger.warning(f"Async file save failed: {e}")
            
            future = self._file_saver.submit(save_func, *args, **kwargs)
            future.add_done_callback(log_result)
        else:
            # Fallback to sync save
            save_func(*args, **kwargs)
    
    def _cleanup_file_saver(self):
        """Cleanup file saver thread pool."""
        if self._file_saver:
            self._file_saver.shutdown(wait=True)
    
