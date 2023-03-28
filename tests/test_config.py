import os
import unittest

# number of repetitions for tests with random data
default_random_repetitions = 50

# shapes to investigate when testing
default_shapes_list = ((1,), (2,))


skip_expensive_tests_environment_variable = os.environ.get('SKIP_EXPENSIVE_TESTS')

if skip_expensive_tests_environment_variable is None:
    # default
    skip_expensive = False
elif skip_expensive_tests_environment_variable in ['true', '1', 't', 'y', 'yes']:
    skip_expensive = True
else:
    skip_expensive = False


may_skip_expensive = unittest.skipIf(skip_expensive, "Skipping expensive tests")
