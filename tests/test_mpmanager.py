from unittest import TestCase
from paperlab.core.utils import MultiProcessManager


def square(x):
    return x ** 2


class TestMPManager(TestCase):
    def test_run(self):
        manager = MultiProcessManager(4)
        results = manager.map(square, [1, 2, 3, 4, 5, 6, -1])
        self.assertEqual(results, [1, 4, 9, 16, 25, 36, 1])


if __name__ == '__main__':
    import unittest
    unittest.main()