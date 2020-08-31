import unittest
import warnings

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio.cube_file_reader2 import read_cube_file
from extrap.util.exceptions import FileFormatError


class TestCubeFileLoader(unittest.TestCase):

    def test_multi_parameter(self):
        experiment = read_cube_file('data/cubeset/multi_parameter', 'weak')
        self.assertListEqual([Parameter('x'), Parameter('y'), Parameter('z')], experiment.parameters)
        self.assertSetEqual({Coordinate(1, 1, 1), Coordinate(1, 1, 10), Coordinate(1, 1, 25),
                             Coordinate(1, 10, 1), Coordinate(1, 10, 10), Coordinate(1, 10, 25),
                             Coordinate(1, 25, 1), Coordinate(1, 25, 10), Coordinate(1, 25, 25),
                             Coordinate(10, 1, 1), Coordinate(10, 1, 10), Coordinate(10, 1, 25),
                             Coordinate(10, 10, 1), Coordinate(10, 10, 10), Coordinate(10, 10, 25),
                             Coordinate(10, 25, 1), Coordinate(10, 25, 10), Coordinate(10, 25, 25),
                             Coordinate(25, 1, 1), Coordinate(25, 1, 10), Coordinate(25, 1, 25),
                             Coordinate(25, 10, 1), Coordinate(25, 10, 10), Coordinate(25, 10, 25),
                             Coordinate(25, 25, 1), Coordinate(25, 25, 10), Coordinate(25, 25, 25)
                             }, set(experiment.coordinates))
        self.assertSetEqual({Callpath('main'), Callpath('main->init_mat'), Callpath('main->zero_mat'),
                             Callpath('main->mat_mul')}, set(experiment.callpaths))
        call_tree = CallTree()
        main = Node('main', Callpath('main'))
        call_tree.add_child_node(main)
        init_mat = Node('init_mat', Callpath('main->init_mat'))
        main.add_child_node(init_mat)
        zero_mat = Node('zero_mat', Callpath('main->zero_mat'))
        main.add_child_node(zero_mat)
        mat_mul = Node('mat_mul', Callpath('main->mat_mul'))
        main.add_child_node(mat_mul)
        self.assertEqual(call_tree, experiment.call_tree)
        self.assertSetEqual({Metric('visits'), Metric('time'), Metric('min_time'), Metric('max_time'),
                             Metric('PAPI_FP_OPS'), Metric('PAPI_L3_TCM'), Metric('PAPI_L2_TCM')},
                            set(experiment.metrics))
        read_cube_file('data/cubeset/multi_parameter', 'strong')

    def test_single_parameter(self):
        experiment = read_cube_file('data/cubeset/single_parameter', 'weak')
        self.assertListEqual([Parameter('x')], experiment.parameters)
        self.assertSetEqual({Coordinate(1), Coordinate(10), Coordinate(25), Coordinate(50), Coordinate(100),
                             Coordinate(250), Coordinate(500), Coordinate(1000), Coordinate(2000)
                             }, set(experiment.coordinates))
        self.assertSetEqual({Callpath('main'), Callpath('main->init_mat'), Callpath('main->zero_mat'),
                             Callpath('main->mat_mul')}, set(experiment.callpaths))
        self.assertSetEqual({Metric('visits'), Metric('time'), Metric('min_time'), Metric('max_time'),
                             Metric('PAPI_FP_OPS'), Metric('PAPI_L3_TCM'), Metric('PAPI_L2_TCM')},
                            set(experiment.metrics))
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
