from pathlib import Path
from typing import Any
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
)

# Custom YAML Configuration Source
class DynamicYamlSource(YamlConfigSettingsSource):
    """
    Custom YAML configuration source that dynamically reads YAML file
    specified via CLI --config argument.
    """
    
    def __init__(self, settings_cls):
        # Initialize with dummy yaml_file, will be overridden later
        super().__init__(settings_cls, yaml_file=None)
    
    def __call__(self):
        config_path = self.current_state.get("config")
        if config_path is None:
            return {}  # No config file specified
        
        # Reinitialize with actual YAML file path
        super().__init__(self.settings_cls, yaml_file=Path(config_path))
        return super().__call__()


# Main Application Configuration
class DynamicYamlSetting(BaseSettings, cli_parse_args=True, cli_avoid_json=True):
    """
    Main application configuration supporting CLI arguments and YAML files.
    Priority: CLI args > YAML config > defaults
    """
    
    config: Path | None = Field(
        default=None,
        description="Path to YAML configuration file",
        cli_only=True,  # Only available via CLI
        exclude=True    # Exclude from model_dump()
    )

    model_config = SettingsConfigDict(
        extra="forbid",  # Forbid extra fields
        env_prefix="",   # Disable environment variable reading
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["DynamicYamlSetting"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        """
        Customize configuration sources priority:
        1. CLI arguments and init values (highest priority)
        2. YAML configuration file (if --config specified)
        3. Environment variables
        4. .env files
        5. Secret files (lowest priority)
        """
        return (
            init_settings,
            DynamicYamlSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )