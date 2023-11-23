# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np

from extrap.entities.coordinate import Coordinate
from extrap.entities.fraction import Fraction
from extrap.entities.functions import SingleParameterFunction
from extrap.entities.measurement import Measurement
from extrap.entities.terms import CompoundTerm
from extrap.modelers.single_parameter.basic import SingleParameterModeler
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestBasicModeler(TestCaseWithFunctionAssertions):
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

    def test_modeling(self):
        for exponents in [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0), (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2),
                          (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1), (3, 4, 1), (4, 5, 0),
                          (2, 3, 2), (3, 4, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
                          (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0), (7, 4, 0), (2, 1, 0), (2, 1, 1),
                          (2, 1, 2), (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0), (11, 4, 0),
                          (3, 1, 0), (3, 1, 1)]:
            term = CompoundTerm.create(*exponents)
            term.coefficient = 10
            function = SingleParameterFunction(term)
            function.constant_coefficient = 200
            points = [2, 4, 8, 16, 32]

            values = function.evaluate(np.array(points))
            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
            modeler = SingleParameterModeler()

            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_modeling2(self):
        for exponents in [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (1, 1, 0),
                          (1, 1, 1), (1, 1, 2), (5, 4, 0), (4, 3, 0), (3, 2, 0), (5, 3, 0), (7, 4, 0), (2, 1, 0),
                          (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 2), (8, 3, 0), (11, 4, 0), (3, 1, 0), (3, 1, 1)]:
            for coeff in [200, 3000, 40000, 500000]:
                term = CompoundTerm.create(*exponents)
                term.coefficient = coeff
                function = SingleParameterFunction(term)
                function.constant_coefficient = 200
                points = [2, 4, 8, 16, 32]

                values = function.evaluate(np.array(points))
                measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
                modeler = SingleParameterModeler()

                models = modeler.model([measurements])
                self.assertEqual(1, len(models))
                self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_modeling3(self):
        for exponents in [(0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 3, 0), (1, 1, 0),
                          (1, 1, 1), (1, 1, 2), (5, 4, 0), (4, 3, 0), (3, 2, 0), (5, 3, 0), (7, 4, 0), (2, 1, 0),
                          (9, 4, 0), (7, 3, 0), (5, 2, 0), (5, 2, 2), (8, 3, 0), (11, 4, 0), (3, 1, 0), (3, 1, 1)]:
            for coeff in [200, 3000, 40000, 500000]:
                term = CompoundTerm.create(*exponents)
                term.coefficient = 1000
                function = SingleParameterFunction(term)
                function.constant_coefficient = coeff
                points = [2, 4, 8, 16, 32]

                values = function.evaluate(np.array(points))
                measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
                modeler = SingleParameterModeler()

                models = modeler.model([measurements])
                self.assertEqual(1, len(models))
                self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_modeling_negative_exponents(self):
        for exponents in [(0, 1, -1), (0, 1, -2), (-1, 4, 0), (-1, 2, 0), (-1, 2, -1), (-1, 2, -2), (-2, 3, 0),
                          (-1, 1, 0),
                          (-1, 1, -1), (-1, 1, -2), (-5, 4, 0), (-4, 3, 0), (-3, 2, 0), (-5, 3, 0), (-7, 4, 0),
                          (-2, 1, 0),
                          (-9, 4, 0), (-7, 3, 0), (-5, 2, 0), (-5, 2, -2), (-8, 3, 0), (-11, 4, 0), (-3, 1, 0),
                          (-3, 1, -1)]:
            for coeff in [2, 30, 400, 5000]:
                term = CompoundTerm.create(*exponents)
                term.coefficient = 1000
                function = SingleParameterFunction(term)
                function.constant_coefficient = coeff
                points = [2, 4, 8, 16, 32]

                values = function.evaluate(np.array(points))
                measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
                modeler = SingleParameterModeler()
                modeler.allow_negative_exponents = True
                models = modeler.model([measurements])
                self.assertEqual(1, len(models))
                self.assertApproxFunction(function, models[0].hypothesis.function)

    def test_compare(self):
        points = [4, 8, 16, 32, 64, 128]
        data = [((None, (12.279235119728051 + 112.3997486813747, 0)),
                 [124.67898380110276, 124.67898380110276, 124.67898380110276, 124.67898380110276, 124.67898380110276,
                  124.67898380110276], (None, (124.679, 0.0))), (
                    ((0, Fraction(1, 1)), (392.837968713381, 683.8645895889935)),
                    [1760.5671478913678, 2444.4317374803613, 3128.296327069355, 3812.1609166583485, 4496.025506247342,
                     5179.890095836336], ((0.0, 1.0), (392.838, 683.865))), (
                    ((0, Fraction(2, 1)), (138.69179452369758, 112.44445041582443)),
                    [588.4695961869953, 1150.6918482661176, 1937.8030011768885, 2949.803054919308, 4186.692009493378,
                     5648.469864899094], ((0.0, 2.0), (138.692, 112.444))), (
                    ((Fraction(1, 4), 0), (231.8031252715932, 757.5927278025262)),
                    [1303.2010356851542, 1505.917143334448, 1746.9885808766455, 2033.672449625761, 2374.5989460987153,
                     2780.0311613973026], ((0.25, 0.0), (231.803, 757.593))), (
                    ((Fraction(1, 3), 0), (147.40207355905747, 740.6554582848072)),
                    [1323.1193271863492, 1628.712990128672, 2013.7368787841829, 2498.836580813641, 3110.023906698286,
                     3880.071684009308], ((0.333333, 0.0), (147.402, 740.655))), (
                    ((Fraction(1, 4), Fraction(1, 1)), (662.1669933486077, 136.57938776640577)),
                    [1048.4718383883378, 1351.2616987705137, 1754.802095479854, 2286.378790293861, 2979.9960635869884,
                     3877.9422853175015], ((0.25, 1.0), (662.167, 136.579))), (
                    ((Fraction(1, 3), Fraction(1, 1)), (535.6622118860412, 447.75148218635366)),
                    [1957.1845595719178, 3222.171105004163, 5048.714352111772, 7643.273950315424, 11281.697784358528,
                     16331.344702676097], ((0.333333, 1.0), (535.662, 447.751))), (
                    ((Fraction(1, 4), Fraction(2, 1)), (412.5706706079675, 996.343695251814)),
                    [6048.741737048133, 15493.363821169984, 32295.568918666017, 59655.521239685964, 101863.64986653095,
                     164625.65164339435], ((0.25, 2.0), (412.571, 996.344))), (
                    ((Fraction(1, 3), Fraction(2, 1)), (93.11229615417925, 20.367438006670188)),
                    [222.43746622492063, 459.72618027424267, 914.2759402192239, 1709.6769220384463, 3026.0233691146855,
                     5122.739616052577], ((0.333333, 2.0), (93.1123, 20.3674))), (
                    ((Fraction(1, 2), 0), (939.8019758412179, 402.94640866510485)),
                    [1745.6947931714276, 2079.5065279286637, 2551.5876105016373, 3219.2110800161095, 4163.373245162056,
                     5498.620184191001], ((0.5, 0.0), (939.802, 402.946))), (
                    ((Fraction(1, 2), Fraction(1, 1)), (198.49843369241415, 330.31007853365884)),
                    [1519.7387478270496, 3001.272390797349, 5483.459690230956, 9541.078290708863, 16053.382203308038,
                     26357.722033338472], ((0.5, 1.0), (198.498, 330.31))), (
                    ((Fraction(1, 2), Fraction(2, 1)), (364.8953574839538, 955.112891429775)),
                    [8005.798488922153, 24678.100241316602, 61492.120408989555, 135438.25582322088, 275437.4080892591,
                     529852.4683831728], ((0.5, 2.0), (364.895, 955.113))), (
                    ((Fraction(2, 3), 0), (210.3330694987003, 216.92681699057178)),
                    [756.9543955249287, 1078.0403374609873, 1587.732499462487, 2396.8183736036135, 3681.1621413478483,
                     5719.930789353846], ((0.666667, 0.0), (210.333, 216.927))), (
                    ((Fraction(3, 4), 0), (584.9013580111865, 547.3819137326248)),
                    [2133.1312104080216, 3188.7032237497583, 4963.956667872185, 7949.565182530901, 12970.740177185866,
                     21415.31628391976], ((0.75, 0.0), (584.901, 547.382))), (
                    ((Fraction(2, 3), Fraction(1, 1)), (953.7431095545323, 838.6830078923111)),
                    [5180.440612885216, 11017.939204262264, 22254.963733492266, 43220.718142861355, 81467.31186721638,
                     150062.28747711863], ((0.666667, 1.0), (953.743, 838.683))), (
                    ((Fraction(3, 4), Fraction(1, 1)), (355.50475595529707, 203.8586065472728)),
                    [1508.7031806978325, 3264.666020281983, 6878.980165468027, 14069.422473092825, 28032.266949776153,
                     54659.84835672009], ((0.75, 1.0), (355.505, 203.859))), (
                    ((Fraction(4, 5), 0), (836.4136945625079, 988.9778707606993)),
                    [3834.433979810851, 6056.270190754812, 9924.711720732794, 16660.0596267337, 28386.981453862616,
                     48804.738258536], ((0.8, 0.0), (836.414, 988.978))), (
                    ((Fraction(2, 3), Fraction(2, 1)), (30.370684174349353, 735.0460670350777)),
                    [7439.17078417381, 26492.029097437142, 74706.39628779484, 185250.37318416082, 423416.90529637906,
                     914811.6843285251], ((0.666667, 2.0), (30.3707, 735.046))), (
                    ((Fraction(3, 4), Fraction(2, 1)), (868.3557741916711, 651.1510359543278)),
                    [8235.288783790882, 28745.079790529722, 84215.68837634563, 219888.5845432864, 531287.5324653349,
                     1215054.5573746935], ((0.75, 2.0), (868.356, 651.151))), (
                    ((Fraction(1, 1), 0), (218.35982887307853, 796.5944762009765)),
                    [3404.7377336769846, 6591.11563848089, 12963.871448088703, 25709.383067304327, 51200.406305735574,
                     102182.45278259806], ((1.0, 0.0), (218.36, 796.594))), (
                    ((Fraction(1, 1), Fraction(1, 1)), (729.8185276288646, 193.81268721358396)),
                    [2280.320025337536, 5381.323020754879, 13133.830509298237, 31739.8484818023, 75153.8904176451,
                     174385.9862710001], ((1.0, 1.0), (729.819, 193.813))), (
                    ((Fraction(1, 1), Fraction(2, 1)), (640.8857481060144, 219.18401331861853)),
                    [4147.829961203911, 16422.13470704655, 56751.993157672354, 175988.09640300085, 505640.8524342031,
                     1375363.0172824815], ((1.0, 2.0), (640.886, 219.184))), (
                    ((Fraction(5, 4), 0), (41.41439439883205, 336.0107050284518)),
                    [1942.1779790139603, 4562.217551923606, 10793.75695530929, 25614.93894716142, 60865.84910208294,
                     144707.1154351916], ((1.25, 0.0), (41.4144, 336.011))), (
                    ((Fraction(5, 4), Fraction(1, 1)), (334.34344019665406, 396.1666489172391)),
                    [4816.457423065935, 16324.82895624065, 51043.67450160326, 151094.0866783297, 430617.2857956476,
                     1194290.5953048149], ((1.25, 1.0), (334.343, 396.167))), (
                    ((Fraction(4, 3), 0), (646.7639733950962, 836.1733802023176)),
                    [5956.133986838953, 14025.538056632176, 34359.162151911856, 85596.68418849679, 214707.14930518836,
                     540045.1348296632], ((1.33333, 0.0), (646.764, 836.173))), (
                    ((Fraction(4, 3), Fraction(1, 1)), (961.3235324936308, 976.4308028867101)),
                    [13361.221801905767, 47830.002071055715, 158430.21598980148, 496957.254308979, 1500759.03676648,
                     4410090.312337113], ((1.33333, 1.0), (961.324, 976.431))), (
                    ((Fraction(3, 2), 0), (993.5060588040174, 789.8477910359313)),
                    [7312.288387091468, 18865.72139149913, 51543.76468510362, 143971.22872036492, 405395.57506920083,
                     1144815.2873512912], ((1.5, 0.0), (993.506, 789.848))), (
                    ((Fraction(3, 2), Fraction(1, 1)), (306.3138276450713, 176.2989470011036)),
                    [3127.096979662729, 12273.883197935773, 45438.844259927595, 159873.90543152104, 541896.6790150353,
                     1787463.339791056], ((1.5, 1.0), (306.314, 176.299))), (
                    ((Fraction(3, 2), Fraction(2, 1)), (623.7521800036756, 545.819243769916)),
                    [18089.967980640988, 111778.0688886881, 559542.6578003977, 2470719.679039657, 10061164.053347098,
                     38731727.88533937], ((1.5, 2.0), (623.752, 545.819))), (
                    ((Fraction(5, 3), 0), (674.515344060474, 93.52470814254)),
                    [1617.1853318529588, 3667.3060046217547, 10176.033429851634, 30839.954953419994, 96443.81648202146,
                     304723.0940893776], ((1.66667, 0.0), (674.515, 93.5247))), (
                    ((Fraction(7, 4), 0), (192.79185213528302, 921.0172026961501)),
                    [10612.912005989885, 35241.758587692566, 118082.9937972425, 396726.5846888438, 1333968.1715455244,
                     4486460.534003467], ((1.75, 0.0), (192.792, 921.017))), (
                    ((Fraction(2, 1), 0), (601.3899361738712, 695.3677746959734)),
                    [11727.274331309445, 45104.92751671617, 178615.54025834304, 712657.9912248506, 2848827.7950908807,
                     11393507.010555001], ((2.0, 0.0), (601.39, 695.368))), (
                    ((Fraction(2, 1), Fraction(1, 1)), (95.64610607936808, 399.1728717576563)),
                    [12869.17800232437, 76736.83748354937, 408848.66678591946, 2043860.7495052798, 9810168.14242224,
                     45780433.96224817], ((2.0, 1.0), (95.6461, 399.173))), (
                    ((Fraction(2, 1), Fraction(2, 1)), (910.0933475245264, 649.3303470713025)),
                    [42467.23556008789, 374924.3732605948, 2660567.19495158, 16623766.97837287, 95748565.7510935,
                     521293702.00774235], ((2.0, 2.0), (910.093, 649.33))), (
                    ((Fraction(9, 4), 0), (991.6538373014305, 196.89404893784854)),
                    [5446.857587036748, 22184.29382918959, 101801.40689347989, 480526.35622160026, 2282055.9737017835,
                     10851623.32968404], ((2.25, 0.0), (991.654, 196.894))), (
                    ((Fraction(7, 3), 0), (454.80058623925424, 800.545149328823)),
                    [20787.379981321064, 102924.57970032863, 516870.1273219162, 2603024.963156712, 13116586.527189683,
                     66101616.62275291], ((2.33333, 0.0), (454.801, 800.545))), (
                    ((Fraction(5, 2), 0), (444.6771885244102, 788.914623707213)),
                    [25689.945147155224, 143253.478519879, 808293.2518647105, 4570326.31979187, 25851599.066826478,
                     146236657.2404956], ((2.5, 0.0), (444.677, 788.915))), (
                    ((Fraction(5, 2), Fraction(1, 1)), (561.6799261002398, 118.03818227888634)),
                    [8116.123591948965, 64663.26005666098, 484046.0745404187, 3419312.620222673, 23207812.621413387,
                     153160603.80521256], ((2.5, 1.0), (561.68, 118.038))), (
                    ((Fraction(5, 2), Fraction(2, 1)), (98.86684880610103, 982.9958000733947)),
                    [125922.32925820061, 1601570.0898857696, 16105502.055251304, 142353096.47013444, 1159589128.4318285,
                     8928380108.544924], ((2.5, 2.0), (0.0, 982.996))), (
                    ((Fraction(8, 3), 0), (119.32012560773262, 194.108049319692)),
                    [7945.26627894892, 49810.980751448864, 315641.6975316356, 2003561.5353809511, 12721184.440340932,
                     80773847.93606871], ((2.66667, 0.0), (119.32, 194.108))), (
                    ((Fraction(11, 4), 0), (335.21389505653997, 713.4600389668701)),
                    [32622.72952123845, 217538.8630750938, 1461501.3736992066, 9829850.300849823, 66125167.21631561,
                     444833408.7346114], ((2.75, 0.0), (335.214, 713.46))), (
                    ((Fraction(3, 1), 0), (854.0891091206599, 475.68703018220896)),
                    [31298.059040782035, 244405.84856241164, 1949268.1647354485, 15588166.694119744, 124699354.92919411,
                     997588860.809789], ((3.0, 0.0), (854.089, 475.687))), (
                    ((Fraction(3, 1), Fraction(1, 1)), (498.14812816021515, 788.4374477528575)),
                    [101418.14144052597, 1211538.0678765494, 12918257.292110976, 129178089.58795632, 1240105375.9704788,
                     11574312691.156733], ((3.0, 1.0), (498.148, 788.437))), ]
        modeler = SingleParameterModeler()
        modeler.use_crossvalidation = False
        for orig, values, (exponents, coeff) in data:
            if exponents:
                term = CompoundTerm.create(*exponents)
                term.coefficient = coeff[1]
                function = SingleParameterFunction(term)
            else:
                function = SingleParameterFunction()
            function.constant_coefficient = coeff[0]

            measurements = [Measurement(Coordinate(p), None, None, v) for p, v in zip(points, values)]
            models = modeler.model([measurements])
            self.assertEqual(1, len(models))
            self.assertApproxFunction(function, models[0].hypothesis.function, places=3)
