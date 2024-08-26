from dataclasses import asdict, dataclass, field
from typing import Optional

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    API_BASE_CONFIG_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
    API_TYPE_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
    REQUEST_TIMEOUT_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.providers._configs.utils import (
    resolve_aliases,
    validate_required_keys,
    raise_deprecation_warnings,
    validate_forbidden_keys,
)

structlogger = structlog.get_logger()

OPENAI_API_TYPE = "openai"

DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    # Model name aliases
    MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
    # API type aliases
    OPENAI_API_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    # API base aliases
    OPENAI_API_BASE_CONFIG_KEY: API_BASE_CONFIG_KEY,
    # API version aliases
    OPENAI_API_VERSION_CONFIG_KEY: API_VERSION_CONFIG_KEY,
    # Timeout aliases
    REQUEST_TIMEOUT_CONFIG_KEY: TIMEOUT_CONFIG_KEY,
}

REQUIRED_KEYS = [MODEL_CONFIG_KEY, API_TYPE_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


@dataclass
class OpenAIClientConfig:
    """Parses configuration for Azure OpenAI client, resolves aliases and
    raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
            - If `api_type` has a value different from `openai`.
    """

    model: str

    # API Type is not actually used by LiteLLM backend, but we define
    # it here for:
    # 1. Backward compatibility.
    # 2. Because it's used as a switch denominator for Azure OpenAI clients.
    api_type: str

    api_base: Optional[str]
    api_version: Optional[str]
    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_type != OPENAI_API_TYPE:
            message = f"API type must be set to '{OPENAI_API_TYPE}'."
            structlogger.error(
                "openai_client_config.validation_error",
                message=message,
                api_type=self.api_type,
            )
            raise ValueError(message)
        if self.model is None:
            message = "Model cannot be set to None."
            structlogger.error(
                "openai_client_config.validation_error",
                message=message,
                model=self.model,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> "OpenAIClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            AzureOpenAIClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Validate that the required keys are present
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        this = OpenAIClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            api_type=config.pop(API_TYPE_CONFIG_KEY),
            # Optional parameters
            api_base=config.pop(API_BASE_CONFIG_KEY, None),
            api_version=config.pop(API_VERSION_CONFIG_KEY, None),
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters (this also includes timeout).
            extra_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)


def is_openai_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    an OpenAI client.
    """

    from litellm.utils import get_llm_provider

    # Process the config to handle all the aliases
    config = resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)

    # Case: Configuration contains `api_type: openai`
    if config.get(API_TYPE_CONFIG_KEY) == OPENAI_API_TYPE:
        return True

    # Case: Configuration contains `model: openai/gpt-4` (litellm approach)
    #
    # This case would bypass the Rasa's Azure OpenAI client and
    # instantiate the client through the default litellm clients.
    # This expression will recognize this attempt and return
    # `true` if this is the case. However, this config is not
    # valid config to be used within Rasa. We want to avoid having
    # multiple ways to do the same thing. This configuration will
    # result in an error.
    if (model := config.get(MODEL_CONFIG_KEY)) is not None:
        if model.startswith(f"{OPENAI_API_TYPE}/"):
            return True

    # Case: Configuration contains "known" models of openai (litellm approach)
    #
    # Similar to the case above.
    try:
        _, provider, _, _ = get_llm_provider(config.get(MODEL_CONFIG_KEY))
        if provider == OPENAI_API_TYPE:
            return True
    except Exception:
        pass

    return False
