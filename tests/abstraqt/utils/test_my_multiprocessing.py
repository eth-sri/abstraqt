from unittest import TestCase
import time
import random

from abstraqt.utils.my_multiprocessing import MyProcess, MyProcessMap, MyProcessResult, MyProcessTimeout


###########
# HELPERS #
###########


def f(i: int):
    return i * i


def f_random(i: int):
    random.randint(0, 1)
    return i * i


def f_slow(i: int):
    time.sleep(10)
    return i * i


n = 10
args = [(i,) for i in range(n)]
expected = [i*i for i in range(n)]


#########
# TESTS #
#########


class MyMultiprocessingTest(TestCase):

    def test_my_process(self):
        p = MyProcess(f, (2, ))
        p.start()
        p.join()
        result = p.get_result()
        self.assertIsInstance(result, MyProcessResult)
        assert isinstance(result, MyProcessResult)
        self.assertEqual(result.result, 4)

    def test_my_process_map(self):
        mapper = MyProcessMap(1)
        mapper.map_join(f, args)
        self.assertEqual(mapper.results, expected)
    
    def test_my_process_order(self):
        mapper = MyProcessMap(1)
        mapper.map_join(f, args)
        self.assertEqual(mapper.results, expected)

    def test_timeout(self):
        timeout = 0.1
        mapper = MyProcessMap(1, timeout=timeout)
        start = time.time()
        outcomes = mapper.map_join(f_slow, args, sleep=0)
        elapsed = time.time() - start
        expected = n * timeout
        self.assertLess(elapsed, expected * 1.1 + 2)

        for outcome in outcomes:
            self.assertIsInstance(outcome, MyProcessTimeout)
