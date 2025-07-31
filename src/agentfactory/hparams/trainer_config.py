from typing import Any, Literal, Union
from pydantic import BaseModel, Field

class OptimizerConfig(BaseModel):
    """Optimizer configuration class"""
    name: str = Field(default="adamw_torch", description="Optimizer type")
    lr: float = Field(default=1e-5, description="Learning rate")
    weight_decay: float = Field(default=0.001, description="Weight decay coefficient")
    warmup_steps: int = Field(default=20, description="Number of warmup steps")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm for clipping")
    loraplus_lr_ratio: float = Field(default=1.0, description="LoRA+ learning rate ratio")

class LossConfig(BaseModel):
    """Loss configuration class"""
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"] = Field(default="seq-mean-token-mean", description="Loss aggregation mode")
    temperature: float = Field(default=1.0, description="Temperature for softmax")
    beta: float = Field(default=0.0, description="Beta parameter for loss function (related to KL divergence)")
    epsilon_low: float = Field(default=0.2, description="Lower epsilon for clipping")
    epsilon_high: float = Field(default=0.28, description="Upper epsilon for clipping")
    
    use_fused_kernels: bool = Field(default=True, description="Use fused loss")
    use_liger_loss: bool = Field(default=False, description="Enable Liger loss optimization")
    liger_loss_num_chunks: int = Field(default=16, description="Number of chunks for Liger loss")

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.use_fused_kernels and self.use_liger_loss:
            raise ValueError("use_fused_kernels and use_liger_loss cannot both be True")

class LoraConfig(BaseModel):
    """Lora configuration class"""
    r: int = Field(default=8, description="LoRA rank")
    target_modules: list[str] | str | None = Field(default=None, description="Target modules for LoRA")
    exclude_modules: list[str] | str | None = Field(default=None, description="Modules to exclude from LoRA")
    lora_alpha: int = Field(default=8, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.0, description="Dropout rate for LoRA layers")
    bias: Literal["none", "all", "lora_only"] = Field(default="none", description="Bias handling mode")
    use_rslora: bool = Field(default=False, description="Enable rank-stabilized LoRA")
    modules_to_save: list[str] | None = Field(default=None, description="Additional modules to save")
    init_lora_weights: bool | str = Field(default=True, description="Initialize LoRA weights")

class ModelConfig(BaseModel):
    """Model configuration class"""
    name: str | None = Field(default=None, description="Model name or path")
    model_init_kwargs: dict[str, Any] | None = Field(default=None, description="Model initialization arguments")
    processing_class: str | None = Field(default=None, description="Processing class for tokenization")
    
    chat_template: str | None = Field(default=None, description="Custom chat template")
    chat_template_path: str | None = Field(default=None, description="Path to chat template file")

    autocast_adapter_dtype: bool = Field(default=False, description="Auto-cast adapter to specific dtype")
    lora_config: LoraConfig | None = Field(default=None, description="LoRA configuration")

class TrainerConfig(BaseModel):
    """Trainer configuration."""

    ppo_epochs: int = Field(default=1, description="Number of PPO epochs per update")
    num_update_per_ppo_epoch: int = Field(default=1, description="Number of updates per PPO epoch")
    
    use_turn_advantage: Union[bool, Literal["auto"]] = Field(default="auto", description="Enable turn-level advantages for EMT-GRPO. Set to 'auto' to automatically detect based on data")
    padding_free_packing: bool = Field(default=True, description="Enable padding-free sequence packing")
    ppo_max_token_len_per_gpu: int = Field(default=8192, description="Max tokens per GPU for PPO training")
    forward_max_token_len_per_gpu: int = Field(default=8192, description="Max tokens per GPU for forward pass")

    dataloader_num_workers: int = Field(default=0, description="Number of dataloader workers")

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer configuration")
    loss: LossConfig = Field(default_factory=LossConfig, description="Loss function configuration")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")