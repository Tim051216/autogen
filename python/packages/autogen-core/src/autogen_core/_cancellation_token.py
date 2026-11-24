import asyncio
import threading
from asyncio import Future
from typing import Any, Callable, List


def _cancel_future(future: Future[Any]) -> None:
    loop = future.get_loop()
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is loop:
        future.cancel()
        return

    if loop.is_closed():
        return

    try:
        loop.call_soon_threadsafe(future.cancel)
    except RuntimeError:
        if not loop.is_closed():
            raise


class CancellationToken:
    """A token used to cancel pending async calls"""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        """Cancel pending async calls linked to this cancellation token."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                for callback in self._callbacks:
                    callback()

    def is_cancelled(self) -> bool:
        """Check if the CancellationToken has been used"""
        with self._lock:
            return self._cancelled

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Attach a callback that will be called when cancel is invoked"""
        with self._lock:
            if self._cancelled:
                callback()
            else:
                self._callbacks.append(callback)

    def link_future(self, future: Future[Any]) -> Future[Any]:
        """Link a pending async call to a token to allow its cancellation"""
        with self._lock:
            if self._cancelled:
                _cancel_future(future)
            else:

                def _cancel() -> None:
                    _cancel_future(future)

                self._callbacks.append(_cancel)
        return future
