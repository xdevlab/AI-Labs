import time

import pytest

from invokeai.backend.model_management.models.base import BaseModelType, ModelType, SubModelType
from invokeai.backend.util.test_utils import install_and_load_model


@pytest.mark.slow
def test_model_load_time(model_installer, torch_device):
    model_name = "juggernautXL_version2"
    base_model = BaseModelType.StableDiffusionXL
    model_type = ModelType.Main
    submodel_type = SubModelType.TextEncoder

    start = time.time()
    model = install_and_load_model(
        model_installer=model_installer,
        model_path_id_or_url="TODO",
        model_name=model_name,
        base_model=base_model,
        model_type=model_type,
        submodel_type=submodel_type,
    )
    done = time.time()
    print(f"Loaded model from disk in {done - start}s")
