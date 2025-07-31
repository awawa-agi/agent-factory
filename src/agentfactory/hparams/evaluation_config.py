from pydantic import BaseModel, Field
from .rollout_worker_config import RolloutWorkerConfig
from ..rollout.core.config import RolloutConfig

class EvaluationConfig(BaseModel):
    """Evaluation configuration class"""
    steps: int | None = Field(default=None, description="Evaluation frequency in steps")
    on_first_step: bool = Field(default=False, description="Run evaluation on first step")
    rollout_worker: RolloutWorkerConfig = Field(default_factory=RolloutWorkerConfig, description="Rollout worker configuration for evaluation")
    rollout_config: RolloutConfig | None = Field(default=None, description="Evaluation rollout config (None = use default_rollout_config)")