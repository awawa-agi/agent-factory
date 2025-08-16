"""
Heavily referenced from: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L293
"""

import os
import gc
import stat
import time
from pathlib import Path
from typing import Any, Union, TypedDict, NotRequired
from loguru import logger
import numpy as np

import torch
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.trainer import Trainer
from transformers.optimization import get_scheduler
from transformers.modeling_utils import PreTrainedModel
from torch.distributed.tensor import DTensor

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
from peft.optimizers import create_loraplus_optimizer

from ..hparams import TrainerConfig
from ..data import RLDataset, DataCollatorWithPacking, BinPackBatchSampler, wrap_sentinel_collate, SentinelWrapper
from ..data.packing_utils import unpack_batch
from .loss_calculator import LossCalculator

class TrainSample(TypedDict):
    id: str
    messages: list[dict[str, Any]]
    advantage: float
    num_loss_tokens: int
    num_tokens: int
    turn_advantages: NotRequired[list[float]]  # Per-turn advantages for EMT-GRPO

def get_optimizer_cls(name: str):
    """
    A hacking way to get the optimizer class from the name
    """
    from dataclasses import dataclass
    @dataclass
    class DummyArgs:
        output_dir: str
        optim: str
        learning_rate: float = 1e-4
        adam_beta1: float = 0.9
        adam_beta2: float = 0.999
        adam_epsilon: float = 1e-8
        optim_args: str = ""
    args = DummyArgs(output_dir="tmp", optim=name)
    cls, _ = Trainer.get_optimizer_cls_and_kwargs(args) # type: ignore
    return cls

class AgentTrainer:
    def __init__(
        self,
        config: TrainerConfig,
        accelerator: Accelerator,
    ):
        self.config = config
        self.accelerator = accelerator
        self._global_step = 0

        self.model = self._load_model(config)
        self.processing_class = self._load_processing_class(config)

        self.optimizer = self._create_optimizer(config)
        self.lr_scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.optimizer.warmup_steps,
            num_training_steps=100_0000,
        )

        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler,
        )
        self.loss_calculator = LossCalculator(
            config=self.config.loss,
        )
        self._metric_logs = {}

    def _load_model(self, config: TrainerConfig):
        """
        Ref: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/loader.py
        """
        if config.model.name is None:
            raise ValueError("Model name cannot be None")
            
        model_config = AutoConfig.from_pretrained(
            config.model.name,
        )

        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
            load_class = AutoModelForVision2Seq
        elif type(model_config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
            load_class = AutoModelForImageTextToText
        elif type(model_config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
            load_class = AutoModelForSeq2SeqLM
        elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
            load_class = AutoModelForTextToWaveform
        else:
            load_class = AutoModelForCausalLM

        model_init_kwargs = config.model.model_init_kwargs or {}
        model = load_class.from_pretrained(
            config.model.name,
            **model_init_kwargs,
        )

        if config.model.lora_config is not None:
            lora_config = LoraConfig(**config.model.lora_config.model_dump())
            
            model = get_peft_model(
                model,
                lora_config,
                autocast_adapter_dtype=config.model.autocast_adapter_dtype,
            )
        else:
            raise ValueError("lora_config is None. While currently only support LoRA training")

        return model

    def _create_optimizer(self, config: TrainerConfig):
        optimizer_cls = get_optimizer_cls(config.optimizer.name)
        optimizer = create_loraplus_optimizer(
            model=self.model,  # type: ignore
            optimizer_cls=optimizer_cls,
            lr=config.optimizer.lr,
            loraplus_lr_ratio=config.optimizer.loraplus_lr_ratio,
        )

        return optimizer

    def _load_processing_class(self, config: TrainerConfig):
        if config.model.chat_template is not None and config.model.chat_template_path is not None:
            raise ValueError("chat_template and chat_template_path cannot be set at the same time")

        if config.model.chat_template_path is not None and not os.path.exists(config.model.chat_template_path):
            raise ValueError(f"chat_template_path {config.model.chat_template_path} does not exist")
        
        init_kwargs = {}
        if config.model.chat_template is not None:
            init_kwargs['chat_template'] = config.model.chat_template
        elif config.model.chat_template_path is not None:
            with open(config.model.chat_template_path, 'r') as f:
                init_kwargs['chat_template'] = f.read()

        # --------- Processing class loading ---------
        init_kwargs['pretrained_model_name_or_path'] = (
            config.model.processing_class
            if config.model.processing_class is not None
            else config.model.name
        )
        processing_class = AutoProcessor.from_pretrained(
            **init_kwargs
        )
            
        if type(processing_class).__name__ == "Qwen2_5_VLProcessor":
            from ..data import Qwen2_5_VLProcessorWithAssistantMask
            processing_class = Qwen2_5_VLProcessorWithAssistantMask.from_pretrained(
                **init_kwargs,
            )

        if 'chat_template' in init_kwargs:
            processing_class.chat_template = init_kwargs['chat_template']

        return processing_class

    def _save_fsdp_lora(self, model: PreTrainedModel, save_dir: Path | None, adapter_name: str="default"):
        """
        reference: https://github.com/huggingface/accelerate/issues/3487
        """
        state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
        for key, value in state_dict.items():
            local_full_value = value
            if isinstance(value, DTensor):
                local_full_value = local_full_value.full_tensor()
            if self.accelerator.is_main_process:
                state_dict[key] = local_full_value.cpu()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            assert save_dir is not None
            torch.save(state_dict, save_dir / "adapter_model.bin")
            model.peft_config[adapter_name].save_pretrained(save_dir)  # type: ignore
            
    def save_model(self, save_dir: str | Path | None):
        if self.accelerator.is_main_process:
            assert save_dir is not None
            save_dir = Path(save_dir).resolve()
            save_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

        self._save_fsdp_lora(self.model, save_dir)  # type: ignore

        if self.accelerator.is_main_process:
            self.processing_class.save_pretrained(save_dir)

    def _get_policy_dataloader(self, train_samples: list[TrainSample], num_big_batches: int=1):
        # Auto-detect or use configured turn advantage mode
        if self.config.use_turn_advantage == "auto":
            use_turn_advantage = any('turn_advantages' in sample for sample in train_samples)
            logger.info(f"Auto-detected use_turn_advantage = {use_turn_advantage}")
        else:
            use_turn_advantage = bool(self.config.use_turn_advantage)
        
        train_dataset = RLDataset(
            train_samples,
            self.processing_class,
            use_turn_advantage=use_turn_advantage
        )
        train_dataset = SentinelWrapper(train_dataset)

        data_collator = DataCollatorWithPacking(self.processing_class)
        collate_fn = wrap_sentinel_collate(data_collator)

        max_capacity = self.config.ppo_max_token_len_per_gpu
        logger.info(f"Data packing max_capacity: {max_capacity}")
        batch_sampler = BinPackBatchSampler(
            dataset=train_dataset,
            max_capacity=max_capacity,
            num_big_batches=num_big_batches,
            world_size=self.accelerator.num_processes,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            pin_memory=True,
            num_workers=self.config.dataloader_num_workers,
        )
        self.accelerator.even_batches = False
        train_dataloader = self.accelerator.prepare(train_dataloader)

        return train_dataloader
        
    def _extract_per_token_logs(self, batch, loss_dict, batch_indices):
        """Extract per-token logs from batch results for visualization"""
        # Unpack batch results and map back to TrainSample IDs
        packed_batch_data = {
            'input_ids': batch['input_ids'].squeeze(0),
            'logps': loss_dict['logps'].squeeze(0),
            'entropies': loss_dict['entropies'].squeeze(0),
            'assistant_masks': batch['assistant_masks'].squeeze(0),
            'advantages': batch['advantages'].squeeze(0),
        }
        
        # Unpack sequences and map to TrainSample IDs
        with torch.no_grad():
            unpacked_sequences = unpack_batch(
                packed_batch_data,
                cu_seq_lens=batch['cu_seq_lens_q'],
                indices=batch_indices,
                return_dict=True
            )
        
        # Convert to format expected by logging, using dataset indices as temporary IDs
        per_token_logs = []
        for dataset_idx, train_sample_data in unpacked_sequences.items():
            per_token_logs.append({
                'dataset_idx': dataset_idx,  # Temporary: will be mapped to actual train_sample_id later
                'input_ids': train_sample_data['input_ids'].cpu().numpy(),
                'logps': train_sample_data['logps'].detach().cpu().float().numpy(),
                'entropies': train_sample_data['entropies'].detach().cpu().float().numpy(),
                'assistant_masks': train_sample_data['assistant_masks'].cpu().numpy(),
                'advantages': train_sample_data['advantages'].cpu().float().numpy(),
            })
        
        return per_token_logs

    def _update_model_parameters(self) -> dict[str, float]:
        """Update model parameters with gradient clipping and return training metrics."""
        metrics = {}

        # Apply gradient clipping if configured
        if self.config.optimizer.max_grad_norm > 0:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.config.optimizer.max_grad_norm
            )
            metrics['grad_norm'] = grad_norm.item()
        
        # Record learning rates for all parameter groups
        learning_rates = [group['lr'] for group in self.optimizer.param_groups]
        if len(learning_rates) == 1:
            metrics['learning_rate'] = learning_rates[0]
        else:
            for i, lr in enumerate(learning_rates):
                metrics[f'learning_rate_group_{i}'] = lr

        # Perform parameter update
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        # Log parameter update details with formatted learning rates
        if len(learning_rates) == 1:
            lr_info = f"learning_rate: {learning_rates[0]:.2e}"
        else:
            formatted_lrs = [f"{lr:.2e}" for lr in learning_rates]
            lr_info = f"learning_rates: [{', '.join(formatted_lrs)}]"
        logger.trace(
            f"Process {self.accelerator.process_index}: "
            f"Parameters updated - "
            f"grad_norm: {metrics.get('grad_norm', 'N/A')}, "
            f"{lr_info}"
        )

        return metrics

    def train_on_batch(self, batch_data: list[TrainSample]):
        """
        batch_data: list[TrainSample]
        batch_data is a list of rollout results
        each rollout result is a TrainSample with the following keys:
        - messages: list[dict[str, Any]]
        - advantage: float
        - num_loss_tokens: int, the number of loss tokens in the messages
        - num_tokens: int, the number of tokens in the messages
        """
        log_metrics = {}
        per_token_logs = []

        policy_dataloader = self._get_policy_dataloader(batch_data, num_big_batches=1)

        self.optimizer.zero_grad()

        for i_ppo_epoch in range(self.config.ppo_epochs):
            policy_dataloader.set_epoch(i_ppo_epoch)
            if hasattr(policy_dataloader.batch_sampler, 'batch_sampler'):
                policy_dataloader.batch_sampler.batch_sampler.set_epoch(i_ppo_epoch)
            else:
                policy_dataloader.batch_sampler.set_epoch(i_ppo_epoch)

            num_loss_tokens_in_batch = sum([d['num_loss_tokens'] for d in batch_data])
            logger.info(f"Num loss tokens in batch: {num_loss_tokens_in_batch}")

            log_metrics['trainer/num_loss_tokens'] = num_loss_tokens_in_batch
            log_metrics['trainer/num_sequences'] = len(batch_data)

            self.model.train()
            for i_batch, batch in enumerate(policy_dataloader):
                # Log batch information for debugging
                batch_indices = batch.pop('indices').cpu().tolist()
                logger.trace(
                    f"Process {self.accelerator.process_index} "
                    f"input_ids shape: {batch['input_ids'].shape}, "
                    f"num loss tokens: {batch['assistant_masks'].sum()}, "
                    f"num images: {len(batch['image_grid_thw']) if batch.get('image_grid_thw', None) is not None else 0}, "
                    f"indices: {batch_indices}"
                )

                loss_dict = self.loss_calculator.compute_loss(
                    self.model, 
                    batch, 
                    self.accelerator,
                    batch_num_loss_tokens=num_loss_tokens_in_batch,
                    batch_num_sequences=len(batch_data),
                )
                loss = loss_dict['loss']
                loss *= self.accelerator.num_processes

                # Extract per-token logs for visualization
                batch_logs = self._extract_per_token_logs(batch, loss_dict, batch_indices)
                per_token_logs.extend(batch_logs)

                logger.trace(
                    f"Process {self.accelerator.process_index} "
                    f"loss: {loss.item()}, "
                    f"logps: {loss_dict['logps'].shape}, "
                    f"entropies: {loss_dict['entropies'].shape}"
                )
                log_metrics['trainer/loss'] = loss.item()

                if not batch['is_last']:
                    with self.accelerator.no_sync(self.model):
                        self.accelerator.backward(loss)

                    logger.trace(
                        f"Process {self.accelerator.process_index} "
                        "backward done, "
                        f"batch {i_batch}/{len(policy_dataloader)} loss: {loss.item()}"
                    )
                    continue
                    
                # Last batch - perform parameter update
                self.accelerator.backward(loss)
                update_metrics = self._update_model_parameters()
                log_metrics.update({f"trainer/{k}": v for k, v in update_metrics.items()})
                self._global_step += 1

        # Map dataset indices back to TrainSample IDs
        for log_entry in per_token_logs:
            dataset_idx = log_entry.pop('dataset_idx')  # Remove temporary key
            log_entry['train_sample_id'] = batch_data[dataset_idx]['id']  # Set actual TrainSample ID

        if hasattr(self.accelerator, '_dataloaders'):
            self.accelerator._dataloaders.clear()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        return log_metrics, per_token_logs