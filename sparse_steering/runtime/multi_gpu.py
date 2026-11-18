"""Load the 32B model across GPUs on a single node."""

from unittest.mock import patch


def device_map(two_gpus):
    if not two_gpus:
        return {"": 0}
    # Keep the intervention and output projection on the input device.
    # Only layers after the intervention cross to the second GPU.
    return {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
        **{f"model.layers.{i}": 0 if i <= 43 else 1 for i in range(64)},
    }


def bind(spec, *, two_gpus):
    import torch
    import transformers

    import sparse_steering.runtime.model_loading as parent
    from sparse_steering.runtime.model_loading import bind as shared_bind

    count = 2 if two_gpus else 1
    if torch.cuda.device_count() < count:
        raise RuntimeError(f"The 32B model requires {count} visible GPU(s) for this phase")
    original = getattr(transformers, "AutoModelFor" + "CausalLM").from_pretrained

    class Loaded:

        def __init__(self, model):
            self.model = model

        def cuda(self):
            return self.model

    class Loader:

        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.update(device_map=device_map(two_gpus))
            return Loaded(original(*args, **kwargs))

    with patch.object(parent, "CausalModel", Loader):
        result = shared_bind(spec)
    if next(result[0].layer.parameters()).device != torch.device("cuda:0"):
        raise ValueError("The intervention layer must be on GPU 0.")
    return result
