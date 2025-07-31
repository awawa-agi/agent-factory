from pydantic import BaseModel, Field

class SingleDatasetConfig(BaseModel):
    """Single dataset configuration class"""
    name: str = Field(description="Dataset name")
    hf_hub_url: str | None = Field(default=None, description="HuggingFace Hub URL")
    split: str | None = Field(default=None, description="Dataset split")
    file_name: str | None = Field(default=None, description="File name for local datasets")

class DataConfig(BaseModel):
    """Data configuration class"""
    train: SingleDatasetConfig | None = Field(default=None, description="Training dataset configuration")
    eval: SingleDatasetConfig | list[SingleDatasetConfig] | None = Field(default=None, description="Evaluation dataset configuration")
    n_prompts_per_iteration: int = Field(default=32, description="Number of prompts per training iteration")
    n_rollouts_per_prompt: int = Field(default=8, description="Number of rollouts per prompt")