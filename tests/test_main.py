import unittest
from util.math import add

class TestMain(unittest.TestCase):

    def test_add(self):
        result = add(2,2)
        self.assertEqual(result, 4)

if __name__ == '__main__':
    unittest.main()
