# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.util.exceptions import FileFormatError, InvalidExperimentError


# noinspection DuplicatedCode
class TestOneParameterFiles(unittest.TestCase):
    def test_read_1(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.parameters, [Parameter('x')])

    def test_read_2(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_2.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_3(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_3.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_4(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_4.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_5(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_5.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_6(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_6.txt")
        self.assertEqual(len(experiment.metrics), 1)
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.callpaths, [
            Callpath('met1'), Callpath('met2'), Callpath('met3'), Callpath('met4')])
        p = Parameter('p')
        self.assertListEqual(experiment.parameters, [p])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(1000),
            Coordinate(2000),
            Coordinate(4000),
            Coordinate(8000),
            Coordinate(16000)
        ])

    def test_read_7(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_7.txt")
        self.assertEqual(len(experiment.metrics), 1)
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.callpaths, [Callpath('met1')])
        p = Parameter('p')
        self.assertListEqual(experiment.parameters, [p])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(1000),
            Coordinate(2000),
            Coordinate(4000),
            Coordinate(8000),
            Coordinate(16000)
        ])

    def test_wrong_file(self):
        self.assertRaises(FileFormatError, TextFileReader().read_experiment, "data/json/input_1.JSON")
        self.assertRaises(FileFormatError, TextFileReader().read_experiment, "data/talpas/talpas_1.txt")

    def testOpenTextIntegrity(self):
        # create experiment object to check against
        callpath = Callpath("reg")

        metric = Metric("metr")

        parameters = [Parameter("x")]

        coordinates = [Coordinate(20.0),
                       Coordinate(30.0),
                       Coordinate(40.0),
                       Coordinate(50.0),
                       Coordinate(60.0)]

        data_points = [
            [82.00848518948744, 81.42023198759563, 81.98984501217201, 80.76019835580321, 82.54252892405795],
            [184.5140079299443, 177.46069577925178, 179.12260901809867, 178.9221229310546, 177.93461023001083],
            [315.26845992523744, 314.8769218843138, 315.7165660477195, 324.22329944880516, 323.2719163782126],
            [509.81064352081336, 495.74457198852593, 501.0059487282256, 502.5272204028191, 509.6665018646357],
            [725.3886983183301, 727.8628493044997, 727.0517005284136, 729.5284355624206, 721.0616186790007]
        ]
        # load experiment from openText()
        experiment = TextFileReader().read_experiment("data/input/input_data_1p.txt")
        self.assertListEqual(parameters, experiment.parameters)
        self.assertListEqual(coordinates, experiment.coordinates)
        self.assertListEqual([callpath], experiment.callpaths)
        self.assertListEqual([metric], experiment.metrics)
        self.assertIn((callpath, metric), experiment.measurements)
        self.assertListEqual([(callpath, metric)], list(experiment.measurements.keys()))
        self.assertListEqual([Measurement(c, callpath, metric, v) for c, v in zip(coordinates, data_points)],
                             experiment.measurements[callpath, metric])

    def test_errors(self):
        self.assertRaises(FileFormatError, TextFileReader().read_experiment, "data/input/experiments_SP/experiment_neg.txt")
        TextFileReader().read_experiment("data/input/experiments_SP/experiment2.txt")
        self.assertRaises(InvalidExperimentError, TextFileReader().read_experiment, "data/input/experiments_SP/experiment3_neg.txt")
        TextFileReader().read_experiment("data/input/experiments_SP/experiment4.txt")


# noinspection DuplicatedCode
class TestTwoParameterFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_1.txt")
        self.assertEqual(len(experiment.parameters), 2)

    def test_read_2(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_2.txt")
        self.assertEqual(len(experiment.parameters), 2)

    def test_read_3(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_3.txt")
        self.assertEqual(len(experiment.parameters), 2)
        self.assertListEqual(experiment.parameters, [
            Parameter('x'), Parameter('y')])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(20.0, 1.0),
            Coordinate(20.0, 2.0),
            Coordinate(20.0, 3.0),
            Coordinate(20.0, 4.0),
            Coordinate(20.0, 5.0),
            #
            Coordinate(30.0, 1.0),
            Coordinate(30.0, 2.0),
            Coordinate(30.0, 3.0),
            Coordinate(30.0, 4.0),
            Coordinate(30.0, 5.0),

            Coordinate(40.0, 1.0),
            Coordinate(40.0, 2.0),
            Coordinate(40.0, 3.0),
            Coordinate(40.0, 4.0),
            Coordinate(40.0, 5.0),

            Coordinate(50.0, 1.0),
            Coordinate(50.0, 2.0),
            Coordinate(50.0, 3.0),
            Coordinate(50.0, 4.0),
            Coordinate(50.0, 5.0),

            Coordinate(60.0, 1.0),
            Coordinate(60.0, 2.0),
            Coordinate(60.0, 3.0),
            Coordinate(60.0, 4.0),
            Coordinate(60.0, 5.0)
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('merge'), Callpath('sort')])

    def test_read_4(self):
        experiment = TextFileReader().read_experiment("data/text/two_parameter_4.txt")
        self.assertEqual(len(experiment.metrics), 1)
        self.assertEqual(len(experiment.parameters), 2)
        self.assertListEqual(experiment.callpaths, [Callpath(
            'met1'), Callpath('met2'), Callpath('met3'), Callpath('met4')])
        self.assertListEqual(experiment.parameters, [
            Parameter('p'), Parameter('q')])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(1000, 10),
            Coordinate(2000, 20),
            Coordinate(4000, 40),
            Coordinate(8000, 80),
            Coordinate(16000, 160)])

    def test_errors(self):
        TextFileReader().read_experiment("data/input/experiments_MP/experiment_MP.txt")
        self.assertRaises(InvalidExperimentError, TextFileReader().read_experiment, "data/input/experiments_MP/experiment_MP_neg1.txt")
        self.assertRaises(FileFormatError, TextFileReader().read_experiment, "data/input/experiments_MP/experiment_MP_neg2.txt")

    def testOpenTextIntegrityMultiParam(self):
        callpath = Callpath("reg")

        metric = Metric("metr")

        parameters = [Parameter("x"), Parameter("y")]

        coordinates = [
            Coordinate(20.0, 1.0),
            Coordinate(20.0, 2.0),
            Coordinate(20.0, 3.0),
            Coordinate(20.0, 4.0),
            Coordinate(20.0, 5.0),
            #
            Coordinate(30.0, 1.0),
            Coordinate(30.0, 2.0),
            Coordinate(30.0, 3.0),
            Coordinate(30.0, 4.0),
            Coordinate(30.0, 5.0),

            Coordinate(40.0, 1.0),
            Coordinate(40.0, 2.0),
            Coordinate(40.0, 3.0),
            Coordinate(40.0, 4.0),
            Coordinate(40.0, 5.0),

            Coordinate(50.0, 1.0),
            Coordinate(50.0, 2.0),
            Coordinate(50.0, 3.0),
            Coordinate(50.0, 4.0),
            Coordinate(50.0, 5.0),

            Coordinate(60.0, 1.0),
            Coordinate(60.0, 2.0),
            Coordinate(60.0, 3.0),
            Coordinate(60.0, 4.0),
            Coordinate(60.0, 5.0)
        ]

        data_points = [
            [1.007912824061828, 1.0003031166379763, 1.0146314163801136, 0.9888371709656842, 0.9814784961141532],
            [82.31483486850328, 80.77055829825096, 81.65028667288966, 79.84894914331908, 80.69133265056031],
            [127.0982588521472, 129.97720256124214, 126.37233644578495, 129.95328527792617, 128.79143884701386],
            [163.761116922403, 160.64424490477055, 163.15711078980416, 159.9466809220868, 161.16513023758068],
            [186.84836895674908, 186.88261474927117, 189.29212868834847, 187.00694904185926, 183.26480034190746],
            [1.0195866920777528, 0.9965626548891021, 1.0181840262519737, 0.9863362890535387, 0.9893290137007527],
            [179.55463831422327, 180.74821882024628, 183.8728518616761, 182.54443671884627, 182.24933579588313],
            [280.6908234409108, 281.788263615464, 283.863933677854, 283.8298464032916, 288.8773066881149],
            [366.23299595860806, 361.4313754663896, 364.474908755492, 367.28360995699853, 359.8692941474022],
            [424.1140021695517, 416.5453483049455, 425.49863768871654, 419.9658709243348, 418.9048409420776],
            [0.9832332399984872, 0.9815834670845136, 1.0003806927451324, 0.9968558578424265, 0.9820664393235233],
            [317.0754172371989, 317.36884825095814, 315.9495390075145, 320.4995994903179, 318.6762830447799],
            [504.63767808259195, 516.0708239589679, 507.81689745670286, 511.8768307930591, 506.2807943653508],
            [644.8143141699127, 637.3099257201826, 649.9174626712269, 647.3368427699369, 633.2245986830611],
            [744.9220213789664, 743.3804480439476, 741.3314847916288, 749.8014199318746, 739.8083987550706],
            [1.0072335327043695, 1.005198595503928, 1.0076803056588366, 1.003963391133719, 0.9931552405725348],
            [495.8101061967628, 501.64162008360864, 507.00077184697335, 498.83459734803967, 507.88731718138513],
            [793.4292405511786, 808.4271169186493, 794.4488419240894, 792.3783818373039, 786.1581110613066],
            [986.8778797337357, 1007.0264271966056, 1012.1774356734794, 1012.8234104247143, 994.8558593224009],
            [1168.925970055696, 1152.305843765793, 1163.3320257986136, 1181.721114617381, 1174.9194990604099],
            [1.0123953645414927, 0.9808925858487914, 0.9883957051154446, 1.0097027886766448, 1.0160649467948064],
            [732.9718870349534, 713.5887652750822, 731.5598226968582, 727.8160069274114, 721.843240633473],
            [1139.5493742034544, 1151.1355124340896, 1136.4341443870837, 1123.1009473444153, 1139.5682409322808],
            [1461.1663792149404, 1456.927809881706, 1468.5789635626325, 1425.7927730830347, 1412.4260646792911],
            [1644.9469847409239, 1687.8899020378108, 1659.733621973487, 1646.4234125252788, 1641.8554614544396]
        ]

        experiment = TextFileReader().read_experiment("data/input/input_data_2p.txt")
        self.assertListEqual(parameters, experiment.parameters)
        self.assertListEqual(coordinates, experiment.coordinates)
        self.assertListEqual([callpath], experiment.callpaths)
        self.assertListEqual([metric], experiment.metrics)
        self.assertIn((callpath, metric), experiment.measurements)
        self.assertListEqual([(callpath, metric)], list(experiment.measurements.keys()))
        self.assertListEqual([Measurement(c, callpath, metric, v) for c, v in zip(coordinates, data_points)],
                             experiment.measurements[callpath, metric])


class TestThreeParameterFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = TextFileReader().read_experiment("data/text/three_parameter_1.txt")
        self.assertEqual(len(experiment.parameters), 3)

    def test_read_2(self):
        experiment = TextFileReader().read_experiment("data/text/three_parameter_2.txt")
        self.assertEqual(len(experiment.parameters), 3)

    def test_read_3(self):
        experiment = TextFileReader().read_experiment("data/text/three_parameter_3.txt")
        self.assertEqual(len(experiment.parameters), 3)
        self.assertListEqual(experiment.parameters, [
            Parameter('x'), Parameter('y'), Parameter('z')])

    def testOpenTextMultiParameter2(self):
        experiment = TextFileReader().read_experiment("data/input/input_data_3p.txt")
