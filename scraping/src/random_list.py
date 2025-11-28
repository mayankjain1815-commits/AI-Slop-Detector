import random
from collections.abc import Iterable, Iterator


class RandomList[T]:
    def __init__(self, _iter: Iterable[T], /) -> None:
        self._list = list(_iter)

    def append(self, item: T) -> None:
        self._list.append(item)

    def pop(self) -> T:
        if not self._list:
            raise IndexError("pop from empty list")

        idx = random.randint(0, len(self._list) - 1)
        result = self._list[idx]

        self._list[idx] = self._list[-1]
        self._list.pop()

        return result

    def __bool__(self) -> bool:
        return bool(self._list)

    def __iter__(self) -> Iterator[T]:
        return iter(self._list)
