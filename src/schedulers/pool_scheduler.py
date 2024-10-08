"""Implements a processing scheduler that uses
:py:class:`multiprocessing.Pool`.

:py:class:`multiprocessing.Pool` is a built-in Python class that facilitates
parallel processing on a single machine. Note that it requires compatibility
with `pickle`.
"""

import multiprocessing
from typing import Callable, List, Optional, Sequence
import typing


class ProgressCounter(object):
    def __init__(self) -> None:
        self.val = multiprocessing.Value("i", 0)

    def increment(self, n: int = 1) -> None:
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self) -> int:
        return self.val.value


PROGRESS_COUNTER = ProgressCounter()


class PoolProcessingScheduler:
    """Processing scheduler that uses Python's
    :py:class:`multiprocessing.Pool`.

    This scheduler implements parallel processing (on a single machine) using
    Python's built-in module :py:mod:`multiprocessing`, specifically, the class
    :py:class:`.Pool`.

    Note:
        :py:class:`.Pool` uses, internally, `pickle` as the serialization
        method. This might be a source of errors due to incompatibility. Make
        sure you read the docs carefully before using this scheduler.

    Note:
        When the processing of individual items isn't a very resource demanding
        task (e.g., learning the 2 variable XOR), using this scheduler might
        yield significantly better performance than using
        :class:`.RayProcessingScheduler` (due to `ray's` greater overhead).
        However, in most situations, the performance difference is negligible
        and using :class:`.RayProcessingScheduler` as the processing scheduler
        is preferable to using this class, since `ray` is safer, scales better
        and allows clustering.

    Args:
        num_processes (Optional[int]): Number of worker processes to use. If
            `None`, then the number returned by :py:func:`os.cpu_count()` is
            used.
        chunksize (Optional[int]): :py:meth:`.Pool.map`, used internally by the
            scheduler, chops the input iterable into a number of chunks which it
            submits to the process pool as separate tasks. This parameter
            specifies the (approximate) size of these chunks.
    """

    def __init__(
        self, num_processes: Optional[int] = None, chunksize: Optional[int] = None
    ) -> None:
        self._num_processes = num_processes
        self._chunksize = chunksize or 1
        self._pool = multiprocessing.Pool(processes=num_processes)

    def run(
        self, items: Sequence[typing.Any], func: Callable[[typing.Any], typing.Any]
    ) -> List[typing.Any]:
        """Processes the given items and returns a result.

        Main function of the scheduler. Call it to make the scheduler manage the
        parallel processing of a batch of items using
        :py:class:`multiprocessing.Pool`.

        Note:
            Make sure that both `items` and `func` are serializable with pickle.

        Args:
            items (Sequence[TProcItem]): Iterable containing the items to be
                processed.
            func (Callable[[TProcItem], TProcResult]): Callable (usually a
                function) that takes one item :attr:`.TProcItem` as input and
                returns a result :attr:`.TProcResult` as output. Generally,
                :attr:`.TProcItem` is an individual in the population and
                :attr:`.TProcResult` is the individual's fitness.

        Returns:
            A list containing the results of the processing of each item. It is
            guaranteed that the ordering of the items in the returned list
            follows the order in which the items are yielded by the iterable
            passed as argument.
        """
        return self._pool.map(func, items, chunksize=self._chunksize)

    def run_with_progress(
        self,
        items: Sequence[typing.Any],
        func: Callable[[typing.Any], typing.Any],
        progress_callback: Callable[[int], typing.Any],
    ) -> List[typing.Any]:
        """Processes the given items and returns a result.

        Main function of the scheduler. Call it to make the scheduler manage the
        parallel processing of a batch of items using
        :py:class:`multiprocessing.Pool`.

        Note:
            Make sure that both `items` and `func` are serializable with pickle.

        Args:
            items (Sequence[TProcItem]): Iterable containing the items to be
                processed.
            func (Callable[[TProcItem], TProcResult]): Callable (usually a
                function) that takes one item :attr:`.TProcItem` as input and
                returns a result :attr:`.TProcResult` as output. Generally,
                :attr:`.TProcItem` is an individual in the population and
                :attr:`.TProcResult` is the individual's fitness.

        Returns:
            A list containing the results of the processing of each item. It is
            guaranteed that the ordering of the items in the returned list
            follows the order in which the items are yielded by the iterable
            passed as argument.
        """

        result = []
        with multiprocessing.Pool(processes=self._num_processes) as pool:
            result_iterator = pool.imap(func, items, chunksize=self._chunksize)
            for item in result_iterator:
                result.append(item)
                progress_rate = (len(result) * 100) // len(items)
                progress_callback(progress_rate)

        return result

    def close(self) -> None:
        """Calls the equivalent method on the scheduler's pool object."""
        self._pool.close()

    def join(self) -> None:
        """Calls the equivalent method on the scheduler's pool object."""
        self._pool.join()

    def terminate(self) -> None:
        """Calls the equivalent method on the scheduler's pool object."""
        self._pool.terminate()
