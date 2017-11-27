#!/usr/bin/env python
import os
import unittest

import numpy as np

if __name__ == '__main__':
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(TEST_DIR)
    # Numpy set up
    # TODO: get number of column in current bash
    np.set_printoptions(linewidth=175, precision=2)
    # Run tests
    unittest.TextTestRunner().run(suite)
