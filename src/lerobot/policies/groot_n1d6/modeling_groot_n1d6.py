import builtins
import os
from collections import deque
from typing import TypeVar
from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any

from transformers import AutoModel

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_groot_n1d6 import GrootN1D6Config

T = TypeVar("T", bound="GrootPolicy")

class GrootN1D6Policy(PreTrainedPolicy):
    config_class = GrootN1D6Config
    name = "groot_n1d6"

    def __init__(self, config: GrootN1D6Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        self._groot_model = self._create_groot_model()

    def _create_groot_model(self):
        model, loading_info = AutoModel.from_pretrained(
            self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            tune_vlln=self.config.tune_vlln,
            state_dropout_prob=self.config.state_dropout_prob,
            backbone_trainable_params_fp32=self.config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=self.config.transformers_loading_kwargs,
            output_loading_info=True,
            **self.config.transformers_loading_kwargs,
        )

        # TODO: log model parameters

        return model


    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: GrootN1D6Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load Groot policy from pretrained model.

        Handles two cases:
        1. Base GR00T models (e.g., 'nvidia/GR00T-N1.5-3B') - loads the raw model
        2. Fine-tuned LeRobot checkpoints - loads config and weights from safetensors

        Args:
            pretrained_name_or_path: Path to the GR00T model or fine-tuned checkpoint
            config: Optional GrootConfig. If None, loads from checkpoint or creates default
            force_download: Force download even if cached
            resume_download: Resume interrupted download
            proxies: Proxy settings
            token: HuggingFace authentication token
            cache_dir: Cache directory path
            local_files_only: Only use local files
            revision: Specific model revision
            strict: Strict state dict loading
            **kwargs: Additional arguments (passed to config)

        Returns:
            Initialized GrootPolicy instance with loaded model
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from huggingface_hub.errors import HfHubHTTPError

        print(
            "The Groot policy is a wrapper around Nvidia's GR00T N1.6 model.\n"
            f"Loading pretrained model from: {pretrained_name_or_path}"
        )

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False

        import ipdb; ipdb.set_trace()
        # Check if this is a fine-tuned LeRobot checkpoint (has model.safetensors)
        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(os.path.join(model_id, SAFETENSORS_SINGLE_FILE))
            else:
                # Try to download the safetensors file to check if it exists
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,  # Just check, don't force download
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            # This is a fine-tuned LeRobot checkpoint - use parent class loading
            print("Detected fine-tuned LeRobot checkpoint, loading with state dict...")
            return super().from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                config=config,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            )

        # This is a base GR00T model - load it fresh
        print("Detected base GR00T model, loading from HuggingFace...")


        # Override the base_model_path with the provided path
        config.base_model_path = str(pretrained_name_or_path)

        # Pass through any additional config overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create a fresh policy instance - this will automatically load the GR00T model
        # in __init__ via _create_groot_model()
        policy = cls(config)

        policy.eval()
        return policy
        

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        pass

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        pass


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # TODO: implement forward pass
        pass