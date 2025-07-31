from .base_inference_engine import BaseInferenceEngine
from .vllm_api_engine import VLLMApiEngine
from ..hparams.inference_engine_config import InferenceEngineConfig

def create_inference_engine(config: InferenceEngineConfig) -> BaseInferenceEngine:
    if config.type == "vllm" and config.mode == "server":
        return VLLMApiEngine(config)
    else:
        raise ValueError(f"Invalid inference engine type: {config.type}")

__all__ = [
    "BaseInferenceEngine",
    "VLLMApiEngine",
    "create_inference_engine",
]