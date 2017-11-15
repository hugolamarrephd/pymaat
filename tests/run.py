#!/usr/bin/env python
import os
import unittest

if __name__ == '__main__':
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(TEST_DIR)
    unittest.TextTestRunner().run(suite)
