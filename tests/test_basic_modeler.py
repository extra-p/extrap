import unittest

from entities.coordinate import Coordinate
from entities.measurement import Measurement
from entities.terms import CompoundTerm
from modelers.single_parameter.basic import SingleParameterModeler


class TestBasicModeler(unittest.TestCase):
    def test_default_functions(self):
        modeler = SingleParameterModeler()
        for bb in modeler.create_default_building_blocks(False):
            self.assertEqual(len(bb.simple_terms), 1)
            self.assertNotEqual(bb.simple_terms[0].term_type, 'logarithm')

    def test_get_matching_hypotheses(self):
        modeler = SingleParameterModeler()
        modeler.hypotheses_building_blocks.append(CompoundTerm.create(1, 1, 1))
        for bb in modeler.get_matching_hypotheses(
                [Measurement(Coordinate(15), None, None, 15),
                 Measurement(Coordinate(0.1), None, None, 0.1)]):
            self.assertEqual(len(bb.simple_terms), 1)
            self.assertNotEqual(bb.simple_terms[0].term_type, 'logarithm')

        hbb = modeler.get_matching_hypotheses(
            [Measurement(Coordinate(31), None, None, 31),
             Measurement(Coordinate(1), None, None, 1)])
        self.assertIn(2, (len(bb.simple_terms) for bb in hbb))
        self.assertIn('logarithm', (bb.simple_terms[0].term_type for bb in hbb))

    def test_generate_building_blocks(self):
        modeler = SingleParameterModeler()
        hbb = modeler.generate_building_blocks([], [])
        self.assertListEqual(hbb, [])

        hbb = modeler.generate_building_blocks([2], [3])
        self.assertEqual(len(hbb), 3)
        self.assertIn(CompoundTerm.create(2, 3), hbb)
        self.assertIn(CompoundTerm.create(2, 0), hbb)
        self.assertIn(CompoundTerm.create(0, 3), hbb)

        hbb = modeler.generate_building_blocks([2, 4], [3, 5])
        self.assertEqual(len(hbb), 8)
        self.assertIn(CompoundTerm.create(2, 0), hbb)
        self.assertIn(CompoundTerm.create(4, 0), hbb)
        self.assertIn(CompoundTerm.create(0, 3), hbb)
        self.assertIn(CompoundTerm.create(0, 5), hbb)

        self.assertIn(CompoundTerm.create(2, 3), hbb)
        self.assertIn(CompoundTerm.create(2, 5), hbb)
        self.assertIn(CompoundTerm.create(4, 3), hbb)
        self.assertIn(CompoundTerm.create(4, 5), hbb)

        hbb = modeler.generate_building_blocks([2], [3], True)
        self.assertEqual(len(hbb), 1)
        self.assertIn(CompoundTerm.create(2, 3), hbb)

        hbb = modeler.generate_building_blocks([2, 4], [3, 5], True)
        self.assertEqual(len(hbb), 4)
        self.assertIn(CompoundTerm.create(2, 3), hbb)
        self.assertIn(CompoundTerm.create(2, 5), hbb)
        self.assertIn(CompoundTerm.create(4, 3), hbb)
        self.assertIn(CompoundTerm.create(4, 5), hbb)
