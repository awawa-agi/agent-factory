from typing import Any
from pydantic import Field, model_validator
from .dynamic_yaml_setting import DynamicYamlSetting
from .trainer_config import TrainerConfig
from .rollout_worker_config import RolloutWorkerConfig
from .inference_engine_config import InferenceEngineConfig
from .evaluation_config import EvaluationConfig
from .data_config import DataConfig
from .algorithm_config import AlgorithmConfig
from ..rollout import RolloutConfig

# Main Application Configuration
class BasicFlowConfig(DynamicYamlSetting):
    save_dir: str = Field(default="agentfactory_save", description="Directory to save checkpoints and outputs")
    run_name: str = Field(default="agentfactory_run", description="Name of this training run")
    project_name: str = Field(default="agentfactory", description="Project name for experiment tracking")
    tmp_weight_dir: str | None = Field(default=None, description="Temporary directory for model weights")
    
    plugins: list[str] = Field(default=[], description="Plugin file paths to load (e.g., 'examples/alfworld/rewards.py')")
    
    report_to: list[str] = Field(default=['wandb'], description="Experiment tracking services")
    
    # Visualization logging configuration
    num_rollouts_to_log: int = Field(default=32, description="Number of rollouts to log for message visualization")
    num_token_logs_to_upload: int = Field(default=10, description="Number of token logs to upload for token visualization")
    html_visualization_mode: str = Field(default="grouped", description="HTML visualization mode: 'individual' or 'grouped'")
    
    seed: int = Field(default=211, description="Random seed for reproducibility")
    
    # Profiling configuration
    profiler: str | None = Field(default=None, description="Profiler backend to use (e.g., 'pyinstrument'). None to disable profiling.")

    num_iterations: int = Field(default=10_000, description="Total number of training iterations")
    save_steps: int = Field(default=100, description="Save checkpoint every N steps")

    trainer: TrainerConfig = Field(default_factory=TrainerConfig, description="Trainer configuration")
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig, description="Algorithm configuration")
    rollout_worker: RolloutWorkerConfig = Field(default_factory=RolloutWorkerConfig, description="Rollout worker configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    inference_engine: InferenceEngineConfig = Field(default_factory=InferenceEngineConfig, description="Inference engine configuration")
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    default_rollout_config: RolloutConfig = Field(default_factory=RolloutConfig, description="Default rollout configuration")

if __name__ == "__main__":
    config = BasicFlowConfig()
    print(config)