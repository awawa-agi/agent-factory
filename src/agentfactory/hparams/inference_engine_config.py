from typing import Literal
from pydantic import BaseModel, Field

class InferenceEngineConfig(BaseModel):
    """Inference engine configuration class"""
    type: str = Field(default="vllm", description="Inference engine type")
    mode: Literal["server", "colocate"] = Field(default="server", description="Engine mode: server or colocate")
    urls: list[str] = Field(default=["http://localhost:8000"], description="Inference server URLs")
    server_startup_timeout: float = Field(default=300.0, description="Server startup timeout (seconds)")
    api_request_timeout: float = Field(default=30.0, description="API request timeout (seconds)")