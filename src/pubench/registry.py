"""A tiny name-keyed registry used for the featurizer and model extension points.

The registry realises the *registry pattern* required by the pipeline: a concrete featurizer
or PU model self-registers under a string name via a decorator, and the runner instantiates it
by name from config — so adding an implementation touches exactly one file.
"""

from __future__ import annotations

from collections.abc import Callable


class Registry[T]:
    """A name -> class registry for a family of pluggable implementations.

    Parameters
    ----------
    kind:
        Human-readable name of the family (e.g. ``"featurizer"``), used in error messages.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers a class under ``name``.

        Parameters
        ----------
        name:
            The unique lookup key for the decorated class.

        Returns:
        -------
        Callable[[type[T]], type[T]]
            A class decorator that records the class and returns it unchanged.

        Raises:
        ------
        ValueError
            If ``name`` is already registered.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise ValueError(f"{self._kind} {name!r} is already registered.")
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``.

        Parameters
        ----------
        name:
            The lookup key.

        Returns:
        -------
        type[T]
            The registered class.

        Raises:
        ------
        KeyError
            If ``name`` is not registered.
        """
        if name not in self._entries:
            raise KeyError(f"Unknown {self._kind} {name!r}; registered: {sorted(self._entries)}.")
        return self._entries[name]

    def names(self) -> list[str]:
        """Return the sorted list of registered names."""
        return sorted(self._entries)
