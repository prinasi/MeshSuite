from .trainers.VanillaTS_trainer import VanillaTSTrainer

from .datasets.Colmap_dataset import ColmapDatasetFactory, BaseDatasetFactory
from .datasets.NerfSynthetic_dataset import NerfSyntheticDatasetFactory
from .datasets.MatrixCity_dataset import MatrixCityDatasetFactory

from .models.VanillaTS_model import VanillaTSModel
from .models.animated_triangle import AnimatedTriangle

from .models.raw_gaussian import RawGaussian
from .models.raw_triangle import RawTriangle

from .utils.config import loadConfig, Config
from .utils.pipeline_utils import run_exp_with_args, run_exp
from .utils.logger import stdout_logger

_EXPERIMENTAL_SYMBOLS = {
    "VanillaGSTrainer": ".trainers.VanillaGS_trainer",
    "VanillaGSModel": ".models.VanillaGS_model",
    "HybridGTModel": ".models.HybridGT_model",
}

__all__ = [
    "VanillaTSTrainer", "ColmapDatasetFactory", "BaseDatasetFactory",
    "NerfSyntheticDatasetFactory", "MatrixCityDatasetFactory", "VanillaTSModel",
    "AnimatedTriangle", "RawGaussian", "RawTriangle", "loadConfig", "Config",
    "run_exp_with_args", "run_exp", "stdout_logger", *_EXPERIMENTAL_SYMBOLS,
]


def __getattr__(name):
    if name not in _EXPERIMENTAL_SYMBOLS:
        raise AttributeError(name)
    from ._optional import optional_symbol

    value = optional_symbol(__name__ + _EXPERIMENTAL_SYMBOLS[name], name)
    globals()[name] = value
    return value
