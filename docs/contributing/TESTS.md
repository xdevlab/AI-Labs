# InvokeAI Tests

## Running Tests

We use `pytest` to run the backend python tests. (See [pyproject.toml](/pyproject.toml) for the default `pytest` options.)

All tests are either 'fast' (no test annotation) or 'slow' (annotated with the `@slow` decorator).

Fast tests are run to validate every PR, and are fast enough that they can be run routinely during development.

Slow tests usually depend on downloading a model, running model inference, or some other slow operation. These tests are currently only run manually on an ad-hoc basis (in the future, they be automated to run nightly). Most developers are only expected to run the 'slow' tests that directly relate to the feature(s) that they are working on.

Below are some common test commands:
```bash
# Run the fast tests. (This implicitly uses the configured default option: `-m "not slow"`.)
pytest tests/

# Equivalent command to run the fast tests.
pytest tests/ -m "not slow"

# Run the slow tests.
pytest tests/ -m "slow"

# Run the slow tests from a specific file.
pytest tests/path/to/slow_test.py -m "slow"

# Run all tests (fast and slow).
pytest tests -m ""
```

## Test Organization

All tests are in the `tests/` directory. This directory mirrors the organization of the `invokeai/` directory. For example, tests for `invokeai/model_management/model_manager.py` would be found in `tests/model_management/test_model_manager.py`.

TODO: The above statement is aspirational. A re-organization of existing tests is required to make it true.

## Fast vs. Slow Tests

Every Python test must be categorized as either 'fast' or 'slow'. 'Fast' tests do no require any annotation, but 'slow' tests must be marked with the `@slow` decorator.

As a rule of thumb, tests should be marked as 'slow' if there is a chance that they take >1s (e.g. on a CPU-only machine with slow internet connection).

Based on this definition, any test that downloads a model or runs model inference should be marked as 'slow'.

## Tests that depend on models

There are a few things to keep in mind when adding tests that depend on models.

1. If a required model is not already present, it should automatically be downloaded as part of the test setup.
2. If a model is already downloaded, it should not be re-downloaded unnecessarily.
3. Take reasonable care to keep the total number of models required for the tests low. Whenever possible, re-use models that are already required for other tests. If you are adding a new model, consider including a comment to describe why it is required/unique.

There are several utilities to help with model setup for tests. Here is a sample test that depends on a model:
```python
import torch

from invokeai.backend.model_management.models.base import BaseModelType, ModelType
from invokeai.backend.util.test_utils import install_and_load_model, model_installer, slow, torch_device

@slow
def test_model(model_installer, torch_device):
    model_info = install_and_load_model(
        model_installer=model_installer,
        model_path_id_or_url="HF/dummy_model_id",
        model_name="dummy_model",
        base_model=BaseModelType.StableDiffusion1,
        model_type=ModelType.Dummy,
    )

    dummy_input = build_dummy_input(torch_device)

    with torch.no_grad(), model_info as model:
        model.to(torch_device, dtype=torch.float32)
        output = model(dummy_input)

    # Validate output...

```