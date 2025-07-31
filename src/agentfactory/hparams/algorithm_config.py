from pydantic import BaseModel, Field

class AlgorithmConfig(BaseModel):
    adv_estimator: str = Field(default="grpo", description="Advantage estimator type: 'grpo' or 'emt_grpo'")
    norm_adv_by_std_in_grpo: bool = Field(default=True, description="Normalize advantage by standard deviation in GRPO")
    min_group_size: int = Field(default=5, description="Minimum group size for advantage calculation")
    
    # EMT-GRPO dual-coefficient parameters
    emt_grpo_turn_weight: float = Field(
        default=1.0,
        description="Weight coefficient for turn-level advantages"
    )
    emt_grpo_episode_weight: float = Field(
        default=1.0,
        description="Weight coefficient for episode-level advantages"
    )
    emt_grpo_gamma: float = Field(
        default=1.0, 
        gt=0.0, le=1.0,
        description="Time discount factor for episode influence (1.0=no decay, <1.0=decay over time)"
    )
    
    # Fine-grained normalization control
    norm_turn_adv_by_std: bool = Field(
        default=True,
        description="Normalize turn-level advantages by standard deviation"
    )
    norm_episode_adv_by_std: bool = Field(
        default=True,
        description="Normalize episode-level advantages by standard deviation"
    )