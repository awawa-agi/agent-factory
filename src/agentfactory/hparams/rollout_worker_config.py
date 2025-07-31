from pydantic import BaseModel, Field

class RolloutWorkerConfig(BaseModel):
    """Rollout configuration class"""
    num_concurrent_rollouts: int = Field(default=60, description="Number of concurrent rollout processes")
    rollout_start_interval: float = Field(default=0.3, description="Interval between starting rollouts (seconds)")
    single_rollout_timeout: float = Field(default=300.0, description="Timeout for single rollout execution (seconds)")