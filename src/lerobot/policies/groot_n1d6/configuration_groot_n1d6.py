# configuration_my_custom_policy.py
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import OptimizerConfig, AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig

@PreTrainedConfig.register_subclass("groot_n1d6")
@dataclass
class GrootN1D6Config(PreTrainedConfig):
    """Configuration class for GrootN1D6.
    """
    # Huggingface model path
    base_model_path: str = "nvidia/GR00T-N1.6-3B"

    # Tuning properties
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True # tuning layer normalization
    
    # Model parameters
    state_dropout_prob = 0.0
    backbone_trainable_params_fp32 = True

    # Build transformers loading kwargs from training config
    transformers_loading_kwargs = {
        "trust_remote_code": True,
        "local_files_only": False
    }

    # Training parameters (matching groot_finetune_script.py)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Add any validation logic here

    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        # TODO: implement
        pass

    @property
    def observation_delta_indices(self) -> None:
        """Return indices for delta observations (None for Groot)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(16))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for Groot)."""
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        # TODO: implement
        pass