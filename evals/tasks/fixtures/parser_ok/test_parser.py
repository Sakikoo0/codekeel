import unittest

from parser import parse


class ParserTest(unittest.TestCase):
    def test_empty_input(self):
        with self.assertRaises(ValueError):
            parse("")

    def test_nonempty_input(self):
        self.assertEqual(parse(" hello "), "hello")
