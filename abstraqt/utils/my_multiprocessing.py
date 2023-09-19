from typing import Callable, Tuple, Optional, Any, List
from multiprocessing import Process, Manager
import traceback
import time
from collections import OrderedDict


FUNC = Callable[..., Any]
ARGS = Tuple[Any, ...]


class MyProcessOutcome:
    
    def __init__(self, f: FUNC, args: ARGS, elapsed_time: float):
        self.f = f
        self.args = args
        self.elapsed_time = elapsed_time
    
    def get_value(self):
        if isinstance(self, MyProcessResult):
            return self.result
        else:
            raise ValueError(f'{self.__class__.__name__} has no result')


class MyProcessTimeout(MyProcessOutcome):
    pass


class MyProcessException(MyProcessOutcome):

    def __init__(self, f: FUNC, args: ARGS, elapsed_time: float, exception: Exception, traceback):
        super().__init__(f, args, elapsed_time)
        self.exception = exception
        self.traceback = traceback
    
    def __str__(self) -> str:
        return f'Exception:\n{self.exception}\n\nOriginal traceback:\n{self.traceback}'


class MyProcessResult(MyProcessOutcome):

    def __init__(self, f: FUNC, args: ARGS, elapsed_time: float, result: Any):
        super().__init__(f, args, elapsed_time)
        self.result = result


def wrap_with_sync_manager(f: FUNC):
    def wrapped(manager_dict, *args):
        try:
            manager_dict['ret'] = f(*args)
        except Exception as e:
            manager_dict['traceback'] = traceback.format_exc()
            manager_dict['exception'] = e
    return wrapped


class MyProcess:

    def __init__(self, f: FUNC, args: ARGS):
        self.f = f
        self.args = args

        # communicate result
        self._manager = Manager()
        self._manager_dict = self._manager.dict()

        # wrap function to communicate result
        wrapped = wrap_with_sync_manager(f)

        # prepare process
        self._process = Process(
            target=wrapped,
            args=(self._manager_dict,) + args
        )

        self.started_time = -1.0

    ############
    # WRAPPERS #
    ############

    def start(self):
        self.started_time = time.time()
        return self._process.start()

    def terminate(self):
        self._process.terminate()

    def is_alive(self):
        return self._process.is_alive()

    def join(self, timeout: Optional[float] = None):
        return self._process.join(timeout)
    
    @property
    def pid(self):
        return self._process.pid

    @property
    def exitcode(self):
        return self._process.exitcode

    ###########
    # HELPERS #
    ###########

    def get_result(
            self,
            timeout: Optional[int] = None
    ) -> Optional[MyProcessOutcome]:
        if not self.is_alive():
            if 'exception' in self._manager_dict:
                exception = self._manager_dict['exception']
                traceback = self._manager_dict['traceback']
                return MyProcessException(
                    self.f,
                    self.args,
                    self.elapsed_time(),
                    exception,
                    traceback
                )
            else:
                return MyProcessResult(
                    self.f,
                    self.args,
                    self.elapsed_time(),
                    self._manager_dict['ret']
                )
        elif timeout is not None and self.elapsed_time() >= timeout:
            self.terminate()
            self.join()
            return MyProcessTimeout(
                self.f,
                self.args,
                self.elapsed_time()
            )
        else:
            return None

    def elapsed_time(self):
        return time.time() - self.started_time


class MyProcessMap:

    def __init__(
            self,
            n_processes: int = 1,
            timeout: Optional[float] = None
        ):
        self.n_processes = n_processes
        self.timeout = timeout
        self.processes: dict[int, MyProcess] = {}
        self._outcomes: dict[int, MyProcessOutcome] = OrderedDict()
    
    def map_join(self, f: FUNC, args_list: List[ARGS], sleep: float=1):
        for _ in self.map_iterator(f, args_list, sleep):
            pass
        return self.outcomes

    def map_iterator(self, f: FUNC, args_list: List[ARGS], sleep: float=1):
        n = len(args_list)
        
        i_args = 0
        while i_args < n or len(self.processes) > 0:
            while i_args < n and len(self.processes) < self.n_processes:
                process = MyProcess(f, args_list[i_args])
                process.start()
                self.processes[i_args] = process
                i_args += 1
            
            i_procs_finished: list[int] = []
            for i_procs, process in self.processes.items():
                result = process.get_result(timeout=self.timeout)
                self._outcomes[i_procs] = result
                if result is not None:
                    i_procs_finished.append(i_procs)
            for i_procs in i_procs_finished:
                del self.processes[i_procs]
            for i_procs in i_procs_finished:
                yield self._outcomes[i_procs]
            
            if len(self.processes) == self.n_processes:
                time.sleep(sleep)
    
    @property
    def outcomes(self):
        return list(self._outcomes.values())

    @property
    def results(self):
        return [o.get_value() for o in self._outcomes.values()]
