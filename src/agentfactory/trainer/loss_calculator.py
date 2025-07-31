"""Ref: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py"""
import torch
from torch import nn
from typing import Union, List, Dict, Tuple, Optional
from contextlib import contextmanager
from accelerate.utils import is_peft_model
from trl.models.utils import _ForwardRedirection
from trl.trainer.utils import selective_log_softmax
from .losses.liger_grpo_loss import LigerFusedLinearGRPOLoss
from .losses.verl_fused_linear_for_ppo import FusedLinearForPPO
from .losses.verl_linear_cross_entropy import linear_cross_entropy
from ..hparams.trainer_config import LossConfig

from transformers import Qwen2ForCausalLM

# ──────────────── LOSS CALCULATION MODULE ────────────────

class LossCalculator:
    """Utility class for computing model losses."""
    
    # Constants
    PAD_TOKEN_ID = 888
    
    def __init__(self, config: LossConfig):
        self.config = config
        self._forward_redirection = _ForwardRedirection()

    def _get_last_hidden_state(self, unwrapped_model, inputs):
        """Get the model's last hidden state."""
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        
        valid_keys = [
            'input_ids', 'attention_mask', 'position_ids',
            'pixel_values', 'image_grid_thw',
            'cu_seq_lens_q', 'cu_seq_lens_k',
            'max_length_k', 'max_length_q',
        ]

        model_inputs = {
            key: inputs[key] for key in valid_keys if key in inputs
        }

        return unwrapped_model.model(**model_inputs).last_hidden_state

    def compute_loss(
        self,
        model,
        inputs,
        accelerator,
        batch_num_loss_tokens,
        batch_num_sequences,
    ):
        if self.config.use_fused_kernels:
            unwrapped_model = accelerator.unwrap_model(model)
            return self._forward_redirection(
                model, unwrapped_model, self._compute_loss_triton,
                unwrapped_model, inputs, batch_num_loss_tokens, batch_num_sequences,
            )

        else:
            raise ValueError("Currently only support fused kernels")

    def _compute_loss_triton(self, unwrapped_model, inputs, batch_num_loss_tokens, batch_num_sequences):

        last_hidden_state = self._get_last_hidden_state(unwrapped_model, inputs)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            unwrapped_model, inputs, last_hidden_state, backend="triton"
        )

        rolled_assistant_masks = torch.roll(inputs['assistant_masks'], shifts=-1, dims=-1)
        rolled_advantages = torch.roll(inputs['advantages'], shifts=-1, dims=-1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps

        log_importance_weights = log_ratio

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.config.epsilon_low, 1 + self.config.epsilon_high)

        per_token_loss1 = coef_1 * rolled_advantages
        per_token_loss2 = coef_2 * rolled_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        loss_flat = per_token_loss.reshape(-1)
        mask_flat = rolled_assistant_masks.reshape(-1).bool()
        masked_loss = loss_flat * mask_flat

        seglen = inputs['cu_seq_lens_q'][1:] - inputs['cu_seq_lens_q'][:-1]

        if self.config.loss_agg_mode == "token-mean":
            loss = masked_loss.sum() / batch_num_loss_tokens
        
        elif self.config.loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-mean"]:

            seq_loss_sum = torch.segment_reduce(masked_loss, lengths=seglen, reduce="sum")

            if self.config.loss_agg_mode == "seq-mean-token-sum":
                loss = seq_loss_sum.sum() / batch_num_sequences

            elif self.config.loss_agg_mode == "seq-mean-token-mean":
                seq_token_cnt = torch.segment_reduce(
                    mask_flat.float(), lengths=seglen, reduce="sum"
                )
                seq_loss_mean = seq_loss_sum / seq_token_cnt.clamp(min=1)
                loss = seq_loss_mean.sum() / batch_num_sequences

        else:
            raise ValueError(f"Unknown mode {self.config.loss_agg_mode}")

        out_dict = {
            'loss': loss,
            'logps': per_token_logps,
            'entropies': entropies,
        }

        return out_dict
   

    def _get_per_token_logps_and_entropies(
        self, model, inputs, last_hidden_state, backend="triton"
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Compute log‐probs and (optionally) entropies for each token.
        Ref: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
        Ref: https://github.com/volcengine/verl/blob/main/verl/models/transformers/dense_common.py
        """

        if 'labels' in inputs:
            rolled_labels = torch.roll(inputs['labels'], shifts=-1, dims=-1)
        elif 'input_ids' in inputs:
            rolled_labels = torch.roll(inputs['input_ids'], shifts=-1, dims=-1)
        else:
            raise RuntimeError(
                "To use _get_per_token_logps_and_entropies, either labels or input_ids must be provided."
            )
        
        if backend == "torch":
            fused_linear_for_ppo = FusedLinearForPPO()
            log_probs, entropy = fused_linear_for_ppo.forward(
                hidden_states=last_hidden_state,
                vocab_weights=model.lm_head.weight,
                input_ids=rolled_labels.long(),  # type: ignore
                temperature=self.config.temperature,
            )

        elif backend == "triton":
            log_probs, entropy = linear_cross_entropy(
                last_hidden_state,
                model.lm_head.weight,
                rolled_labels,
                self.config.temperature,
                "none",
            )
            
        
        return log_probs, entropy