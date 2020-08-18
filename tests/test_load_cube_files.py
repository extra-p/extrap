import unittest
import warnings

from fileio.cube_file_reader2 import read_cube_file
from util.exceptions import FileFormatError


class TestCubeFileLoader(unittest.TestCase):

    def test_multi_parameter(self):
        read_cube_file('data/cubeset/multi_parameter', 'weak')
        read_cube_file('data/cubeset/multi_parameter', 'strong')

    def test_single_parameter(self):
        read_cube_file('data/cubeset/single_parameter', 'weak')
        read_cube_file('data/cubeset/single_parameter', 'strong')

    def test_extra_files_folders(self):
        read_cube_file('data/cubeset/extra_folder', 'weak')
        read_cube_file('data/cubeset/extra_file', 'weak')

    def test_allowed_formats(self):
        read_cube_file('data/cubeset/allowed_formats', 'weak')
        read_cube_file('data/cubeset/allowed_formats', 'strong')

    def test_wrong_parameters(self):
        self.assertRaises(FileFormatError, read_cube_file, 'data/cubeset/negative_example', 'weak')
        self.assertRaises(FileFormatError, read_cube_file, 'data/cubeset/negative_example2', 'weak')
        self.assertRaises(FileFormatError, read_cube_file, 'data/cubeset/negative_example', 'strong')
        self.assertRaises(FileFormatError, read_cube_file, 'data/cubeset/negative_example2', 'strong')

    def test_no_cube_files(self):
        self.assertRaises(FileFormatError, read_cube_file, 'data/cubeset', 'weak')
        self.assertRaises(FileFormatError, read_cube_file, 'data/cubeset', 'strong')

    def test_strong_scaling_warning(self):
        self.assertWarnsRegex(UserWarning, 'Strong scaling', read_cube_file, 'data/cubeset/multi_parameter', 'strong')

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.filterwarnings('ignore', r'^((?!Strong scaling).)*$')
            read_cube_file('data/cubeset/single_parameter', 'strong')
        self.assertFalse(record)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.filterwarnings('ignore', r'^((?!Strong scaling).)*$')
            read_cube_file('data/cubeset/single_parameter', 'weak')
        self.assertFalse(record)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.filterwarnings('ignore', r'^((?!Strong scaling).)*$')
            read_cube_file('data/cubeset/multi_parameter', 'weak')
        self.assertFalse(record)


if __name__ == '__main__':
    unittest.main()
