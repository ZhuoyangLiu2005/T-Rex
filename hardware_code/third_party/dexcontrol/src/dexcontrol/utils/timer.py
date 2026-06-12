# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Utility for measuring code execution time."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from loguru import logger

# Type variable for preserving function return type in decorator
F = TypeVar("F", bound=Callable[..., Any])


class ExecutionTimer:
    """A utility class for measuring execution time of code blocks.

    This class can be used either as a context manager or decorator.

    Examples:
        As context manager:
            with ExecutionTimer("operation name"):
                # code to time

        As decorator:
            @ExecutionTimer("operation name")
            def function_to_time():
                # code to time
    """

    def __init__(self, name: str | None = "Operation") -> None:
        """Initialize the timer.

        Args:
            name: Optional name to identify this timing operation in logs.
                 Defaults to "Operation" if not provided.
        """
        self.name = name or "Operation"
        self.start_time: float = 0.0
        self.end_time: float | None = None

    def __enter__(self) -> "ExecutionTimer":
        """Start timing when entering context.

        Returns:
            Self reference for context manager usage.
        """
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing when exiting context and log the duration.

        Args:
            *args: Exception information passed by context manager.
        """
        self.end_time = time.perf_counter()
        duration = self.elapsed
        logger.info(f"{self.name} took {duration:.4f} seconds")

    def __call__(self, func: F) -> F:
        """Allow the ExecutionTimer to be used as a decorator.

        Args:
            func: The function to be timed.

        Returns:
            Wrapped function that measures and logs execution time.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return cast(F, wrapper)

    def start(self) -> "ExecutionTimer":
        """Start timing manually.

        Returns:
            Self reference for method chaining.
        """
        return self.__enter__()

    def stop(self) -> float:
        """Stop timing manually and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        self.__exit__(None, None, None)
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            Time elapsed since start in seconds. If timer is still running,
            returns current elapsed time.
        """
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time


if __name__ == "__main__":
    # Test as a context manager
    logger.info("Testing ExecutionTimer as context manager:")
    with ExecutionTimer("Sleep operation"):
        time.sleep(1.0)

    # Test as a decorator
    logger.info("\nTesting ExecutionTimer as decorator:")

    @ExecutionTimer("Decorated function")
    def slow_function() -> str:
        time.sleep(0.5)
        return "Done!"

    result = slow_function()
    logger.info(f"Function returned: {result}")

    # Test with manual timing
    logger.info("\nTesting ExecutionTimer with manual timing:")
    timer = ExecutionTimer("Manual timing")
    timer.start()
    time.sleep(3.0)
    timer.stop()
    logger.info(f"Elapsed time property: {timer.elapsed:.4f} seconds")
