# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric


class TestTagInit(unittest.TestCase):
    def execute_test_entity(self, entity):
        cp_tag = entity('test_cp')
        self.assertDictEqual({}, cp_tag.tags)

        cp_tag = entity('test_cp', test__tag='test_value')
        self.assertDictEqual({'test__tag': 'test_value'}, cp_tag.tags)

        cp_tag = entity('test_cp', test__tag='test_value', test__tag2='test_value2')
        self.assertDictEqual({'test__tag': 'test_value', 'test__tag2': 'test_value2'}, cp_tag.tags)

        cp_tag = entity('test_cp', test__tag='test_value', test__tag_bool=True, test__tag__int=123,
                        test__tag__list=['Test', 15])
        self.assertDictEqual({'test__tag': 'test_value', 'test__tag_bool': True,
                              'test__tag__list': ['Test', 15], 'test__tag__int': 123},
                             cp_tag.tags)

    def test_callpath(self):
        self.execute_test_entity(Callpath)

    def test_metric(self):
        self.execute_test_entity(Metric)


class TestTagLookup(unittest.TestCase):
    def test_tag_equal(self):
        cp_tag = Callpath('test_cp', test__tag='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag'))
        cp_tag = Callpath('test_cp', test='test')
        self.assertEqual(None, cp_tag.lookup_tag('test__tag'))
        self.assertEqual(None, cp_tag.lookup_tag('test__tag', None))
        self.assertEqual('def_test', cp_tag.lookup_tag('test__tag', 'def_test'))
        cp_tag = Callpath('test_cp', test__tag='test_value', test='fail')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag', 'def_test'))

    def test_tag_more_precise(self):
        cp_tag = Callpath('test_cp', test__tag='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test'))
        cp_tag = Callpath('test_cp', test='test')
        self.assertEqual(None, cp_tag.lookup_tag('test__tag__test'))
        self.assertEqual(None, cp_tag.lookup_tag('test__tag__test', None))
        self.assertEqual('def_test', cp_tag.lookup_tag('test__tag__test', 'def_test'))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__test='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test'))

    def test_tag_more_precise_with_custom_prefix_length(self):
        cp_tag = Callpath('test_cp', test__tag='test_value')
        self.assertEqual(None, cp_tag.lookup_tag('test__tag__test', prefix_len=2))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__test='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test', prefix_len=2))

        cp_tag = Callpath('test_cp', test='test')
        self.assertEqual('test', cp_tag.lookup_tag('test__tag__test', prefix_len=0))
        self.assertEqual('test', cp_tag.lookup_tag('test__tag__test', 'def_test', prefix_len=0))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__test='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test'))

    def test_tag_equal_with_suffix(self):
        cp_tag = Callpath('test_cp', test__tag='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', suffix='suffix'))
        cp_tag = Callpath('test_cp', test='test')
        self.assertEqual(None, cp_tag.lookup_tag('test__tag__test', suffix='suffix'))
        self.assertEqual(None, cp_tag.lookup_tag('test__tag__test', None, suffix='suffix'))
        self.assertEqual('def_test', cp_tag.lookup_tag('test__tag__test', 'def_test', suffix='suffix'))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__test='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test', suffix='suffix'))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__test='fail',
                          test__tag__test__suffix='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test', suffix='suffix'))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__test='fail',
                          test__tag__suffix='fail', test__tag__test__suffix='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test', suffix='suffix'))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__tag__suffix='test_value',
                          test__tag__test='fail')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test', suffix='suffix'))
        cp_tag = Callpath('test_cp', test__tag='fail', test='fail', test__suffix='fail',
                          test__tag__test='test_value')
        self.assertEqual('test_value', cp_tag.lookup_tag('test__tag__test', 'def_test', suffix='suffix'))
