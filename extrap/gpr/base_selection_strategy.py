# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.entities.coordinate import Coordinate

def suggest_points_base_mode(experiment, parameter_value_series):

    if len(experiment.parameters) == 1:
        possible_points = []
        for i in range(len(parameter_value_series[0])):
            exists = False
            for j in range(len(experiment.coordinates)):
                if parameter_value_series[0][i] == experiment.coordinates[j].as_tuple()[0]:
                    exists = True
                    break
            if exists == False:
                possible_points.append(parameter_value_series[0][i])
        possible_points.sort()
        points_needed = 5-len(experiment.coordinates)
        cords = []
        for i in range(points_needed):
            cords.append(Coordinate(possible_points[i]))
        return cords

    elif len(experiment.parameters) == 2:
        
        x_line_lengths = {}
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y:
                    x.append(cord_values2[0])
            if y not in x_line_lengths:
                x_line_lengths[y] = len(x)
                x_lines[y] = x

        max_value = 0
        best_line_key = None
        for key, value in x_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = x_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[0])):
            if parameter_value_series[0][i] not in best_line:
                potential_values.append(parameter_value_series[0][i])
        potential_values.sort()
        cords_x = []
        for i in range(points_needed):
            cords_x.append(Coordinate(potential_values[i],best_line_key))

        y_line_lengths = {}
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x:
                    y.append(cord_values2[1])
            if x not in y_line_lengths:
                y_line_lengths[x] = len(y)
                y_lines[x] = y

        max_value = 0
        best_line_key = None
        for key, value in y_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = y_lines[best_line_key]
        x = parameter_value_series[0]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[1])):
            if parameter_value_series[1][i] not in best_line:
                potential_values.append(parameter_value_series[1][i])
        potential_values.sort()
        cords_y = []
        for i in range(points_needed):
            cords_y.append(Coordinate(best_line_key,potential_values[i]))

        suggested_cords = []
        for x in cords_x:
            suggested_cords.append(x)
        for x in cords_y:
            suggested_cords.append(x)

        return suggested_cords

    elif len(experiment.parameters) == 3:
        
        x_lines = {}
        x_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y and cord_values2[2] == z:
                    x.append(cord_values2[0])
            if (y,z) not in x_lines:
                x_lines[(y,z)] = x
                x_line_lengths[(y,z)] = len(x)

        #print("DEBUG x_lines:",x_lines)
        #print("DEBUG x_line_lengths:",x_line_lengths)

        max_value = 0
        best_line_key = None
        for key, value in x_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = x_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[0])):
            if parameter_value_series[0][i] not in best_line:
                potential_values.append(parameter_value_series[0][i])
        potential_values.sort()
        cords_x = []
        for i in range(points_needed):
            cords_x.append(Coordinate(potential_values[i],best_line_key[0],best_line_key[1]))

        #print("DEBUG cords_x:",cords_x)
            
        y_lines = {}
        y_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[2] == z:
                    y.append(cord_values2[1])
            if (x,z) not in y_lines:
                y_lines[(x,z)] = y
                y_line_lengths[(x,z)] = len(y)

        max_value = 0
        best_line_key = None
        for key, value in y_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = y_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[1])):
            if parameter_value_series[1][i] not in best_line:
                potential_values.append(parameter_value_series[1][i])
        potential_values.sort()
        cords_y = []
        for i in range(points_needed):
            cords_y.append(Coordinate(best_line_key[0],potential_values[i],best_line_key[1]))

        #print("DEBUG cords_y:",cords_y)

        z_lines = {}
        z_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            z = []
            y = cord_values[1]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[1] == y:
                    z.append(cord_values2[2])
            if (x,y) not in z_lines:
                z_lines[(x,y)] = z
                z_line_lengths[(x,y)] = len(z)

        max_value = 0
        best_line_key = None
        for key, value in z_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = z_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[2])):
            if parameter_value_series[2][i] not in best_line:
                potential_values.append(parameter_value_series[2][i])
        potential_values.sort()
        cords_z = []
        for i in range(points_needed):
            cords_z.append(Coordinate(best_line_key[0],best_line_key[1],potential_values[i]))

        #print("DEBUG cords_z:",cords_z)

        suggested_cords = []
        for x in cords_x:
            suggested_cords.append(x)
        for x in cords_y:
            suggested_cords.append(x)
        for x in cords_z:
            suggested_cords.append(x)

        #print("DEBUG suggested_cords:",suggested_cords)

        return suggested_cords

    elif len(experiment.parameters) == 4:
        
        x1_lines = {}
        x1_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = []
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == x2 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x1.append(cord_values2[0])
            if (x2,x3,x4) not in x1_lines:
                x1_lines[(x2,x3,x4)] = x1
                x1_line_lengths[(x2,x3,x4)] = len(x1)

        #print("DEBUG x1_lines:",x1_lines)
        #print("DEBUG x1_line_lengths:",x1_line_lengths)
                
        max_value = 0
        best_line_key = None
        for key, value in x1_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = x1_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[0])):
            if parameter_value_series[0][i] not in best_line:
                potential_values.append(parameter_value_series[0][i])
        potential_values.sort()
        cords_x1 = []
        for i in range(points_needed):
            cords_x1.append(Coordinate(potential_values[i],best_line_key[0],best_line_key[1],best_line_key[2]))

        #print("DEBUG cords_x1:",cords_x1)

        x2_lines = {}
        x2_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = []
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x2.append(cord_values2[1])
            if (x1,x3,x4) not in x2_lines:
                x2_lines[(x1,x3,x4)] = x2
                x2_line_lengths[(x1,x3,x4)] = len(x2)

        max_value = 0
        best_line_key = None
        for key, value in x2_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = x2_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[1])):
            if parameter_value_series[1][i] not in best_line:
                potential_values.append(parameter_value_series[1][i])
        potential_values.sort()
        cords_x2 = []
        for i in range(points_needed):
            cords_x2.append(Coordinate(best_line_key[0],potential_values[i],best_line_key[1],best_line_key[2]))

        #print("DEBUG cords_x2:",cords_x2)

        x3_lines = {}
        x3_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = []
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[3] == x4:
                    x3.append(cord_values2[2])
            if (x1,x2,x4) not in x3_lines:
                x3_lines[(x1,x2,x4)] = x3
                x3_line_lengths[(x1,x2,x4)] = len(x3)

        max_value = 0
        best_line_key = None
        for key, value in x3_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = x3_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[2])):
            if parameter_value_series[2][i] not in best_line:
                potential_values.append(parameter_value_series[2][i])
        potential_values.sort()
        cords_x3 = []
        for i in range(points_needed):
            cords_x3.append(Coordinate(best_line_key[0],best_line_key[1],potential_values[i],best_line_key[2]))

        #print("DEBUG cords_x3:",cords_x3)

        x4_lines = {}
        x4_line_lengths = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[2] == x3:
                    x4.append(cord_values2[3])
            if (x1,x2,x3) not in x4_lines:
                x4_lines[(x1,x2,x3)] = x4
                x4_line_lengths[(x1,x2,x3)] = len(x4)

        max_value = 0
        best_line_key = None
        for key, value in x4_line_lengths.items():
            if value > max_value:
                best_line_key = key
                max_value = value
        best_line = x4_lines[best_line_key]
        points_needed = 5-max_value

        potential_values = []
        for i in range(len(parameter_value_series[3])):
            if parameter_value_series[3][i] not in best_line:
                potential_values.append(parameter_value_series[3][i])
        potential_values.sort()
        cords_x4 = []
        for i in range(points_needed):
            cords_x4.append(Coordinate(best_line_key[0],best_line_key[1],best_line_key[2],potential_values[i]))

        #print("DEBUG cords_x4:",cords_x4)
        
        suggested_cords = []
        for x in cords_x1:
            suggested_cords.append(x)
        for x in cords_x2:
            suggested_cords.append(x)
        for x in cords_x3:
            suggested_cords.append(x)
        for x in cords_x4:
            suggested_cords.append(x)

        #print("DEBUG suggested_cords:",suggested_cords)

        return suggested_cords

    else:
        return []