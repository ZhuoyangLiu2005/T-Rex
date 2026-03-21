import sys
from .registry_utils import Registry
MODELS = Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT): 
    Returns:
        Model: a constructed model specified by NAME.
    """
    return MODELS.build(cfg, **kwargs)