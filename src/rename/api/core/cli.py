import asyncio
import inspect
from functools import partial, wraps
from typing import Any, Callable

from typer import Typer


class AsyncTyper(Typer):
    """
    A subclass of Typer that allows for async command functions.

    Notes:
        - https://github.com/fastapi/typer/issues/950
        - https://github.com/pallets/click/issues/85#issuecomment-503464628
        - https://github.com/fastapi/typer/issues/88#issuecomment-1794013640
    """

    @staticmethod
    def maybe_run_async(decorator: Callable, func: Callable) -> Any:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            def runner(*args: Any, **kwargs: Any) -> Any:
                return asyncio.run(func(*args, **kwargs))

            decorator(runner)
        else:
            decorator(func)
        return func

    def callback(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)
