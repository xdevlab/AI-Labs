import json
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from omegaconf import OmegaConf
from pydantic import Field

from ..config import CheckpointConfig, MainCheckpointConfig, MainDiffusersConfig
from .base import (
    BaseModelType,
    DiffusersModel,
    InvalidModelException,
    ModelConfigBase,
    ModelType,
    ModelVariantType,
    classproperty,
    read_checkpoint_meta,
)
from .stable_diffusion import StableDiffusionModelBase


class StableDiffusionXLModelFormat(str, Enum):
    Checkpoint = "checkpoint"
    Diffusers = "diffusers"


class StableDiffusionXLModel(StableDiffusionModelBase):
    # TODO: check that configs overwriten properly
    class DiffusersConfig(MainDiffusersConfig):
        model_format: Literal[StableDiffusionXLModelFormat.Diffusers]

    class CheckpointConfig(ModelConfigBase):
        model_format: Literal[StableDiffusionXLModelFormat.Checkpoint]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model in {BaseModelType.StableDiffusionXL, BaseModelType.StableDiffusionXLRefiner}
        assert model_type == ModelType.Main
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusionXL,
            model_type=ModelType.Main,
        )

    @classmethod
    def probe_config(cls, path: str, **kwargs):
        model_format = cls.detect_format(path)
        ckpt_config_path = kwargs.get("config", None)
        if model_format == StableDiffusionXLModelFormat.Checkpoint:
            if ckpt_config_path:
                ckpt_config = OmegaConf.load(ckpt_config_path)
                in_channels = ckpt_config["model"]["params"]["unet_config"]["params"]["in_channels"]

            else:
                checkpoint = read_checkpoint_meta(path)
                checkpoint = checkpoint.get("state_dict", checkpoint)
                in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        elif model_format == StableDiffusionXLModelFormat.Diffusers:
            unet_config_path = os.path.join(path, "unet", "config.json")
            if os.path.exists(unet_config_path):
                with open(unet_config_path, "r") as f:
                    unet_config = json.loads(f.read())
                in_channels = unet_config["in_channels"]

            else:
                raise InvalidModelException(f"{path} is not a recognized Stable Diffusion diffusers model")

        else:
            raise NotImplementedError(f"Unknown stable diffusion 2.* format: {model_format}")

        if in_channels == 9:
            variant = ModelVariantType.Inpaint
        elif in_channels == 5:
            variant = ModelVariantType.Depth
        elif in_channels == 4:
            variant = ModelVariantType.Normal
        else:
            raise Exception("Unkown stable diffusion 2.* model format")

        if ckpt_config_path is None:
            # avoid circular import
            from .stable_diffusion import _select_ckpt_config

            ckpt_config_path = _select_ckpt_config(kwargs.get("model_base", BaseModelType.StableDiffusionXL), variant)

        return cls.create_config(
            path=path,
            model_format=model_format,
            config=ckpt_config_path,
            variant=variant,
        )

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    @classmethod
    def detect_format(cls, model_path: str):
        if os.path.isdir(model_path):
            return StableDiffusionXLModelFormat.Diffusers
        else:
            return StableDiffusionXLModelFormat.Checkpoint
