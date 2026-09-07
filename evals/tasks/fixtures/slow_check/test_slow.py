import time
import unittest


class SlowTest(unittest.TestCase):
    def test_slow(self):
        time.sleep(60)
