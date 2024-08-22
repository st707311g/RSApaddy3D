import os
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Callable

import numpy as np


class Vacancy_ID:
    lock = threading.Lock()

    def __init__(self, max_num: int = 1):
        self.__vacancy = Queue()
        for i in range(max_num):
            self.__vacancy.put(i)

    @property
    def next_id(self):
        with self.lock:
            return self.__vacancy.get() if self.__vacancy.not_empty else -1

    def put(self, id: int):
        with self.lock:
            return self.__vacancy.put(id)


class CallbackPrint:
    lock = threading.Lock()

    def __init__(self, callback: Callable, i: int = 0, total: int = 0):
        self.__callback = callback
        self.__i = i
        self.__total = total

    def forward(self):
        with self.lock:
            self.__i += 1
            percent = round((self.__i) * 100 / self.__total, 1)
            self.__callback(self.__i, self.__total, percent)


class ForEach(object):
    def __init__(self, data) -> None:
        self.data = data


class MultiThreading(object):
    def __init__(self, thread_number: int = os.cpu_count(), task_name: str = "processing"):
        self.thread_number = thread_number
        self.task_name = task_name
        self.vacancy_id = Vacancy_ID(self.thread_number)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def print_callback(self, i: int, total: int, percent: float):
        print(f"\r{self.task_name}: {percent}%", end="")
        if i == total:
            print("")

    def run(self, args: list, func, callback: Callable = None):
        self._func = func

        total: int = None
        for it in args:
            if isinstance(it, ForEach):
                if total is None:
                    total = len(it.data)
                else:
                    assert total == len(it.data)

        assert total is not None

        if callback:
            self.__callback = CallbackPrint(callback, 0, total)
        else:
            self.__callback = None

        # // If you let numpy or cupy handle it,
        # // multi-threading is almost multi-processing.
        with ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            futures = []
            for i in range(total):
                job_args = []
                for it in args:
                    if isinstance(it, ForEach):
                        job_args.append(it.data[i])
                    else:
                        job_args.append(it)

                futures.append(executor.submit(self.job, *job_args))

        return [f.result() for f in futures]

    def run_function(self, *args, thread_id: int):
        return self._func(*args)

    def job(self, *args):
        while True:
            next_id = self.vacancy_id.next_id
            if next_id != -1:
                break

        ret = self.run_function(thread_id=next_id, *args)

        self.vacancy_id.put(next_id)

        if self.__callback:
            self.__callback.forward()

        return ret


class MultiThreadingGPU(MultiThreading):
    def __init__(self, gpu_ids: list[int], task_name: str = "processing"):
        self.gpu_ids = gpu_ids.copy()
        super().__init__(thread_number=len(self.gpu_ids), task_name=task_name)

    def run_function(self, *args, thread_id: int):
        import cupy as cp

        with cp.cuda.Device(self.gpu_ids[thread_id]):
            args = [cp.asarray(it) if isinstance(it, np.ndarray) else it for it in args]
            ret = self._func(*args)
            if isinstance(ret, cp.ndarray):
                ret = ret.get()

        return ret
