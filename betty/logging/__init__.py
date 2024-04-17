from logging import Logger
from .logger_tensorboard import TensorBoardLogger
from .logger_wandb import WandBLogger
from .logger_base import get_logger, LoggerBase


logger_mapping = {
    "tensorboard": TensorBoardLogger,
    "wandb": WandBLogger,
    "none": LoggerBase,
}


def type_check(logger_type):
    assert logger_type in logger_mapping

    if logger_type == "wandb":
        try:
            import wandb
        except ImportError:
            get_logger().warning(
                "[!] WandB is not installed. The default logger will be instead used."
            )
            logger_type = "none"
    elif logger_type == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            get_logger().warning(
                "[!] Tensorboard is not installed. The default logger will be instead used."
            )
            logger_type = "none"

    return logger_type


def logger(logger_type="none", name = None):
    logger_type = type_check(logger_type)
    if logger_type == "wandb" or logger_type == "tensorboard":
        return logger_mapping[logger_type](name)
    return logger_mapping[logger_type]()
