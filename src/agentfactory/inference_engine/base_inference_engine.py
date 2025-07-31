from abc import abstractmethod
from pathlib import Path
from ..hparams import InferenceEngineConfig

class BaseInferenceEngine:
    def __init__(self, config: InferenceEngineConfig):
        self.config = config

    @abstractmethod
    def sleep(self):
        pass

    @abstractmethod
    def wake_up(self):
        pass

    @abstractmethod
    def load_weights_from_disk(self, weights_path: str | Path):
        pass

    @abstractmethod
    def load_lora_weights_from_disk(self, weights_path: str | Path, name: str = "default_lora"):
        pass