#!/usr/bin/env python3
import argparse
import collections

import numpy as np
from typing import Generic, TypeVar

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size to sample from the buffer")
parser.add_argument("--max_length", default=128, type=int, help="Maximum length of the replay buffer")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.

NamedTuple = TypeVar("NamedTuple")  #A generic type fulfilling the NamedTuple protocol.


class PrioritizedReplayBuffer(Generic[NamedTuple]):
    """A prioritized replay buffer with a limited capacity."""

    def __init__(self, max_length: int) -> None:
        self._len: int = 0
        self._max_length: int = max_length
        self._offset: int = 0
        self._data: NamedTuple | None = None

        #Sum tree for efficient sampling and updates
        self._tree = np.zeros(2 * max_length, dtype=np.float64)
        self._max_priority = 1.0  #Default max priority when none is provided

    def __len__(self) -> int:
        return self._len

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def data(self) -> NamedTuple | None:
        return self._data

    def __getitem__(self, index: int | np.ndarray) -> NamedTuple:
        return self._data._make(value[index] for value in self._data)

    def append(self, item: NamedTuple, priority: float | None = None) -> int:
        if not self._len:
            values = [np.empty((self._max_length, *value.shape), dtype=value.dtype)
                      for value in map(np.asarray, item)]
            self._data = item._make(values)

        if self._len < self._max_length:
            index, self._len = self._len, self._len + 1
        else:
            index, self._offset = self._offset, (self._offset + 1) % self._max_length

        for i, value in enumerate(item):
            self._data[i][index] = value

        self.update_priority(index, priority)
        return index

    def update_priority(self, index: int, priority: float | None = None) -> None:
        assert 0 <= index < self._len
        if priority is None:
            priority = self._max_priority
        else:
            assert priority >= 0
            self._max_priority = max(self._max_priority, priority)

        tree_index = index + self._max_length
        diff = priority - self._tree[tree_index]
        self._tree[tree_index] = priority

        # Propagate the difference up the tree
        while tree_index > 1:
            tree_index //= 2
            self._tree[tree_index] = self._tree[2 * tree_index] + self._tree[2 * tree_index + 1]

    def sample(self, size: int, generator=np.random) -> tuple[NamedTuple, np.ndarray, np.ndarray]:
        samples = (generator.uniform(size=size) + np.arange(size)) / size
        total = self._tree[1]
        sample_values = samples * total

        indices = np.empty(size, dtype=np.int32)
        for i, value in enumerate(sample_values):
            idx = 1
            while idx < self._max_length:
                left = 2 * idx
                if value <= self._tree[left]:
                    idx = left
                else:
                    value -= self._tree[left]
                    idx = left + 1
            indices[i] = idx - self._max_length

        priorities = self._tree[indices + self._max_length] / total
        return self[indices], indices, priorities





def main(args: argparse.Namespace) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(args.max_length)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    if main_args.max_length & (main_args.max_length - 1):
        raise ValueError("For the tests to work, max_length must be a power of two.")

    buffer = main(main_args)

    # Run a random test that verifies the required behavior of the buffer.
    Element = collections.namedtuple("Element", ["value"])
    values = np.zeros(main_args.max_length, dtype=np.int64)
    priorities = np.zeros(main_args.max_length + 1, dtype=np.float64)

    for i in range(3 * main_args.max_length):
        # Either append a new character, of update priority of an existing one.
        if i > main_args.max_length and i % 3 == 0:
            buffer.update_priority(j := (i % main_args.max_length), i if i % 2 else None)
        else:
            j = buffer.append(Element(i), i if i % 2 else None)
            values[j] = i
        priorities[j + 1] = i if i % 2 else max(i - 1, 1)

        # Sample from the replay buffer if it is large enough.
        if i >= main_args.batch_size:
            items, indices, probs = buffer.sample(main_args.batch_size, np.random.RandomState(main_args.seed + i))

            # Generate the same samples and compute the buffer probabilities.
            samples = (np.random.RandomState(main_args.seed + i).uniform(size=main_args.batch_size)
                       + np.arange(main_args.batch_size)) / main_args.batch_size

            priorities_cumsum = np.cumsum(priorities)

            # Final verification.
            assert np.all(priorities_cumsum[indices] <= samples * priorities_cumsum[-1])
            assert np.all(priorities_cumsum[indices + 1] > samples * priorities_cumsum[-1])
            np.testing.assert_allclose(probs * priorities_cumsum[-1], priorities[indices + 1], atol=1e-5)
            np.testing.assert_equal(items.value, values[indices])

    print("All checks passed.")
