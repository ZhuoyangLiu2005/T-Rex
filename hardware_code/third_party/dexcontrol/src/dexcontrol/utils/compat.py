"""Robot model compatibility utilities.

This module provides tools for declaring and checking robot model compatibility
at both the method level (via decorators) and script level (via helpers).
"""

import functools
import sys
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from loguru import logger

from dexcontrol.exceptions import ModelNotSupportedError

P = ParamSpec("P")
T = TypeVar("T")


def requires_model(*models: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that restricts a method to specific robot models.

    The decorated method must be on a class with a ``robot_model`` property
    (e.g., ``Robot``). At call time, if ``self.robot_model`` is not in the
    allowed set, ``ModelNotSupportedError`` is raised.

    The allowed models are stored as ``func.__supported_models__`` for
    introspection and documentation generation.

    Args:
        *models: Allowed robot model names (e.g., ``"vega_1"``, ``"vega_1p"``).

    Raises:
        ValueError: If no model names are provided.

    Example::

        class Robot:
            @requires_model("vega_1", "vega_1p")
            def full_body_calibration(self):
                ...
    """
    if not models:
        raise ValueError("requires_model() requires at least one model name")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            if self.robot_model not in models:
                raise ModelNotSupportedError(
                    method=func.__name__,
                    robot_model=self.robot_model,
                    supported_models=models,
                )
            return func(self, *args, **kwargs)

        wrapper.__supported_models__ = models  # type: ignore[attr-defined]
        return wrapper

    return decorator


def resolve_robot_model() -> str:
    """Resolve the current robot model from environment variables.

    Uses the same resolution logic as ``RobotInfo`` to determine the robot
    model without creating a full ``RobotInfo`` instance (avoids URDF loading).

    Resolution order:
        1. ``ROBOT_CONFIG`` env var -> variant -> ``config.robot_model``
        2. ``ROBOT_NAME`` env var -> derived variant -> ``config.robot_model``

    Returns:
        The robot model string (e.g., ``"vega_1"``, ``"vega_1p"``).

    Raises:
        ValueError: If environment variables are not set or invalid.
    """
    from dexbot_utils.robot_info import RobotInfo

    return RobotInfo.get_default_config().robot_model


def supported_models(*models: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that restricts a script function to specific robot models.

    Resolves the robot model from environment variables (``ROBOT_CONFIG`` or
    ``ROBOT_NAME``) *before* the decorated function runs. If the model is not
    in the allowed set, logs a clear error and exits with code 1.

    The allowed models are stored as ``func.__supported_models__`` for
    introspection and documentation generation (same convention as
    :func:`requires_model`).

    This decorator preserves the original function signature, making it
    compatible with ``tyro.cli()``.

    Args:
        *models: Allowed robot model names (e.g., ``"vega_1"``, ``"vega_1p"``).

    Raises:
        ValueError: If no model names are provided.
    """
    if not models:
        raise ValueError("supported_models() requires at least one model name")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                robot_model = resolve_robot_model()
            except ValueError as e:
                logger.error(f"Cannot determine robot model: {e}")
                sys.exit(1)

            if robot_model not in models:
                logger.error(
                    f"This script is only supported on ({', '.join(models)}). "
                    f"Detected robot model from environment: {robot_model}"
                )
                sys.exit(1)

            return func(*args, **kwargs)

        wrapper.__supported_models__ = models  # type: ignore[attr-defined]
        return wrapper

    return decorator
