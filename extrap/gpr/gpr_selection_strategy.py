# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import math
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel

from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement
from extrap.util.progress_bar import DUMMY_PROGRESS


def suggest_points_gpr_mode(experiment,
                            possible_points,
                            selected_callpaths,
                            metric,
                            calculate_cost_manual,
                            process_parameter_id,
                            number_processes,
                            budget,
                            current_cost,
                            model_generator,
                            progress_bar=DUMMY_PROGRESS):
    progress_bar.total = 100
    progress_bar.step("Suggesting additional measurement points.")

    # disable warnings from sk
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    max_repetitions = 5

    # if no callpath is selected use all callpaths
    if len(selected_callpaths) == 0:
        selected_callpaths = experiment.callpaths

    # use the values from the selected callpath
    if len(selected_callpaths) == 1:
        callpath = selected_callpaths[0]
        selected_points = copy.deepcopy(experiment.coordinates)

        # GPR parameter-value normalization for each measurement point
        normalization_factors = get_normalization_factors(experiment)

        # do an noise analysis on the existing points
        mean_noise = analyze_noise(experiment, callpath, metric)

        # nu should be [0.5, 1.5, 2.5, inf], everything else has 10x overhead
        # matern kernel + white kernel to simulate actual noise found in the measurements
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel(
            noise_level=mean_noise)

        # create a gaussian process regressor
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=20
        )

        # add all of the selected measurement points to the gaussian process
        # as training data and train it for these points
        temp_parameters = []
        for parameter in experiment.parameters:
            temp_parameters.append(str(parameter))
        gaussian_process = add_measurements_to_gpr(gaussian_process,
                                                   selected_points,
                                                   experiment.measurements,
                                                   callpath,
                                                   metric,
                                                   normalization_factors,
                                                   temp_parameters)

        # construct a dict of potential measurement points with 
        # the coordinates as the keys and the costs as the values

        modeler = None
        for i in range(len(experiment.modelers)):
            if experiment.modelers[i] == model_generator:
                modeler = experiment.modelers[i]
                break
        model = modeler.models[callpath, metric]
        hypothesis = model.hypothesis
        function = hypothesis.function

        # first use the points from the search space 
        remaining_points_gpr = {}
        predicted_runtimes = {}
        for i in range(len(possible_points)):
            # key = possible_points[i]
            point = possible_points[i]
            # c.1 predict the runtime of the possible_points using the existing performance models
            runtime = function.evaluate(point.as_tuple())
            # runtimes[point] = runtime
            # c.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
            if calculate_cost_manual:
                nr_processes = number_processes
            else:
                nr_processes = point[process_parameter_id]
            cost = runtime * nr_processes
            values = []
            for j in range(max_repetitions):
                values.append(cost)
            runtime_values = []
            for j in range(max_repetitions):
                runtime_values.append(runtime)
            remaining_points_gpr[possible_points[i]] = values
            predicted_runtimes[possible_points[i]] = runtime_values

        # extend the dict in case there are still reps missing from the base coorindates
        repetitions = {}
        costs = {}
        runtimes = {}
        for i in range(len(experiment.coordinates)):
            coordinate = experiment.coordinates[i]
            for j in range(len(experiment.measurements[(callpath, metric)])):
                if coordinate == experiment.measurements[(callpath, metric)][j].coordinate:
                    try:
                        reps = len(experiment.measurements[(callpath, metric)][i].values)
                    except TypeError:
                        reps = 1
                    repetitions[coordinate] = reps
                    cost_values = []
                    runtime_values = []
                    try:
                        for l in range(len(experiment.measurements[(callpath, metric)][i].values)):
                            runtime = experiment.measurements[(callpath, metric)][i].values[l]
                            if calculate_cost_manual:
                                nr_processes = number_processes
                            else:
                                nr_processes = point[process_parameter_id]
                            cost = runtime * nr_processes
                            cost_values.append(cost)
                            runtime_values.append(runtime)
                    except TypeError:
                        runtime = experiment.measurements[(callpath, metric)][i].mean
                        if calculate_cost_manual:
                            nr_processes = number_processes
                        else:
                            nr_processes = point[process_parameter_id]
                        cost = runtime * nr_processes
                        cost_values.append(cost)
                        runtime_values.append(runtime)
                    costs[coordinate] = cost_values
                    runtimes[coordinate] = runtime_values
                    break

        for key, value in repetitions.items():
            remaining_reps = max_repetitions - value
            if remaining_reps > 0:
                mean_cost = np.mean(costs[key])
                mean_runtime = np.mean(runtimes[key])
                values = []
                runtime_values = []
                for i in range(remaining_reps):
                    values.append(mean_cost)
                    runtime_values.append(mean_runtime)
                remaining_points_gpr[key] = values
                predicted_runtimes[key] = runtime_values

        experiment_gpr = copy.deepcopy(experiment)

        suggested_points = []
        rep_numbers = []

        while True:
            # additional limit for the loop
            # otherwise takes too long or the app might crash if set budget is too large
            # since performing the GPR is reall resource intensive
            if len(suggested_points) >= 100:
                break

            # identify all possible next points that would 
            # still fit into the modeling budget in core hours
            fitting_measurements = []
            for key, value in remaining_points_gpr.items():

                current_cost = calculate_selected_point_cost(experiment_gpr, callpath, metric, calculate_cost_manual,
                                                             number_processes)

                # always take the first value in the list, until none left
                new_cost = current_cost + value[0]

                if new_cost <= budget:
                    fitting_measurements.append(key)

            # find the next best additional measurement point using the gpr
            best_index = -1
            best_rated = sys.float_info.max

            for i in range(len(fitting_measurements)):

                parameter_values = fitting_measurements[i].as_tuple()
                x = []

                for j in range(len(parameter_values)):

                    if len(normalization_factors) != 0:
                        x.append(parameter_values[j] * normalization_factors[str(experiment.parameters[j])])

                    else:
                        x.append(parameter_values[j])

                # term_1 is cost(t)^2
                term_1 = math.pow(remaining_points_gpr[fitting_measurements[i]][0], 2)
                # predict variance of input vector x with the gaussian process
                x = [x]
                _, y_cov = gaussian_process.predict(x, return_cov=True)
                y_cov = abs(y_cov)
                # term_2 is gp_cov(t,t)^2
                term_2 = math.pow(y_cov, 2)
                # rated is h(t)

                rep = 1
                rep = max_repetitions - len(predicted_runtimes[fitting_measurements[i]]) + 1

                rep_func = 2 ** ((1 / 2) * rep - (1 / 2))
                noise_func = -math.tanh((1 / 4) * mean_noise - 2.5)
                cost_multiplier = rep_func + noise_func
                rated = (term_1 * cost_multiplier) / term_2

                if rated <= best_rated:
                    best_rated = rated
                    best_index = i

                    # if there has been a point found that is suitable
            if best_index != -1:

                # add the identified measurement point to the selected point list
                parameter_values = fitting_measurements[best_index].as_tuple()
                cord = Coordinate(parameter_values)

                # only add coordinate to selected points list if not already in there (because of reps)
                if cord not in selected_points:
                    selected_points.append(cord)

                # add the new point to the gpr and call fit()
                gaussian_process = add_measurement_to_gpr(gaussian_process,
                                                          cord,
                                                          predicted_runtimes,
                                                          normalization_factors,
                                                          experiment_gpr.parameters)

                new_value = 0

                # remove the identified measurement point from the remaining point list
                try:
                    new_value = predicted_runtimes[cord][0]
                    predicted_runtimes[cord].pop(0)
                    if len(predicted_runtimes[cord]) == 0:
                        predicted_runtimes.pop(cord)

                    # pop value from cord in remaining points list that has been selected as best next point
                    remaining_points_gpr[cord].pop(0)

                    # pop cord from remaining points when no value left anymore
                    if len(remaining_points_gpr[cord]) == 0:
                        remaining_points_gpr.pop(cord)

                except KeyError:
                    pass

                # add this point to the gpr experiment
                experiment_gpr = create_experiment(cord, experiment_gpr, new_value, callpath, metric)

                # add cord and rep number to the list of suggestions
                rep_number = 1
                for i in range(len(experiment_gpr.measurements[(callpath, metric)])):
                    if cord == experiment_gpr.measurements[(callpath, metric)][i].coordinate:
                        rep_number = len(experiment_gpr.measurements[(callpath, metric)][i].values)
                        break
                suggested_points.append(cord)
                rep_numbers.append(rep_number)

                # update progress bar
                progress_bar.update(1)

            # if there are no suitable measurement points found
            # break the while True loop
            else:
                break

    # sum up the values from the selected callpaths
    else:
        selected_points = copy.copy(experiment.coordinates)

        # GPR parameter-value normalization for each measurement point
        normalization_factors = get_normalization_factors(experiment)

        remaining_points_gpr = {}
        predicted_runtimes = {}
        mean_noise_values = []

        for l, callpath in enumerate(selected_callpaths):
            # do an noise analysis on the existing points
            mean_noise = analyze_noise(experiment, callpath, metric)
            mean_noise_values.append(mean_noise)

            # construct a dict of potential measurement points with
            # the coordinates as the keys and the costs as the values

            modeler = None
            for i in range(len(experiment.modelers)):
                if experiment.modelers[i] == model_generator:
                    modeler = experiment.modelers[i]
                    break
            model = modeler.models[callpath, metric]
            hypothesis = model.hypothesis
            function = hypothesis.function

            # first use the points from the search space
            for i in range(len(possible_points)):
                # key = possible_points[i]
                point = possible_points[i]
                # c.1 predict the runtime of the possible_points using the existing performance models
                runtime = function.evaluate(point.as_tuple())
                # runtimes[point] = runtime
                # c.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
                if calculate_cost_manual:
                    nr_processes = number_processes
                else:
                    nr_processes = point[process_parameter_id]
                cost = runtime * nr_processes
                values = []
                for j in range(max_repetitions):
                    values.append(cost)
                runtime_values = []
                for j in range(max_repetitions):
                    runtime_values.append(runtime)
                if l == 0:
                    remaining_points_gpr[possible_points[i]] = values
                    predicted_runtimes[possible_points[i]] = runtime_values
                else:
                    if possible_points[i] in remaining_points_gpr:
                        for o in range(len(remaining_points_gpr[possible_points[i]])):
                            remaining_points_gpr[possible_points[i]][o] = remaining_points_gpr[possible_points[i]][
                                                                              o] + np.mean(values)
                    else:
                        remaining_points_gpr[possible_points[i]] = values
                    if possible_points[i] in predicted_runtimes:
                        for o in range(len(predicted_runtimes[possible_points[i]])):
                            predicted_runtimes[possible_points[i]][o] = predicted_runtimes[possible_points[i]][
                                                                            o] + np.mean(runtime_values)
                    else:
                        predicted_runtimes[possible_points[i]] = runtime_values

            # extend the dict in case there are still reps missing from the base coorindates
            repetitions = {}
            costs = {}
            runtimes = {}
            for i in range(len(experiment.coordinates)):
                coordinate = experiment.coordinates[i]
                for j in range(len(experiment.measurements[(callpath, metric)])):
                    if coordinate == experiment.measurements[(callpath, metric)][j].coordinate:
                        try:
                            reps = len(experiment.measurements[(callpath, metric)][i].values)
                        except TypeError:
                            reps = 1
                        repetitions[coordinate] = reps
                        cost_values = []
                        runtime_values = []
                        try:
                            for l in range(len(experiment.measurements[(callpath, metric)][i].values)):
                                runtime = experiment.measurements[(callpath, metric)][i].values[l]
                                if calculate_cost_manual:
                                    nr_processes = number_processes
                                else:
                                    nr_processes = point[process_parameter_id]
                                cost = runtime * nr_processes
                                cost_values.append(cost)
                                runtime_values.append(runtime)
                        except TypeError:
                            runtime = experiment.measurements[(callpath, metric)][i].mean
                            if calculate_cost_manual:
                                nr_processes = number_processes
                            else:
                                nr_processes = point[process_parameter_id]
                            cost = runtime * nr_processes
                            cost_values.append(cost)
                            runtime_values.append(runtime)
                        costs[coordinate] = cost_values
                        runtimes[coordinate] = runtime_values
                        break

            for key, value in repetitions.items():
                remaining_reps = max_repetitions - value
                if remaining_reps > 0:
                    mean_cost = np.mean(costs[key])
                    mean_runtime = np.mean(runtimes[key])
                    values = []
                    runtime_values = []
                    for i in range(remaining_reps):
                        values.append(mean_cost)
                        runtime_values.append(mean_runtime)
                    if l == 0:
                        remaining_points_gpr[key] = values
                        predicted_runtimes[key] = runtime_values
                    else:
                        if key in remaining_points_gpr:
                            for o in range(len(remaining_points_gpr[possible_points[i]])):
                                remaining_points_gpr[possible_points[i]][o] = remaining_points_gpr[possible_points[i]][
                                                                                  o] + np.mean(values)
                        else:
                            remaining_points_gpr[key] = values
                        if key in predicted_runtimes:
                            for o in range(len(predicted_runtimes[possible_points[i]])):
                                predicted_runtimes[possible_points[i]][o] = predicted_runtimes[possible_points[i]][
                                                                                o] + np.mean(runtime_values)
                        else:
                            predicted_runtimes[key] = runtime_values

        # calculate mean noise from found noise values for different callpaths
        mean_noise = np.mean(mean_noise_values)

        # nu should be [0.5, 1.5, 2.5, inf], everything else has 10x overhead
        # matern kernel + white kernel to simulate actual noise found in the measurements
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel(
            noise_level=mean_noise)

        # create a gaussian process regressor
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=20
        )

        # add all of the selected measurement points to the gaussian process
        # as training data and train it for these points
        temp_parameters = []
        for parameter in experiment.parameters:
            temp_parameters.append(str(parameter))
        gaussian_process = add_measurements_to_gpr(gaussian_process,
                                                   selected_points,
                                                   experiment.measurements,
                                                   callpath,
                                                   metric,
                                                   normalization_factors,
                                                   temp_parameters)

        experiment_gpr = copy.deepcopy(experiment)

        suggested_points = []
        rep_numbers = []

        while True:
            # additional limit for the loop
            # otherwise takes too long or the app might crash if set budget is too large
            # since performing the GPR is reall resource intensive
            if len(suggested_points) >= 100:
                break

            # identify all possible next points that would 
            # still fit into the modeling budget in core hours
            fitting_measurements = []
            for key, value in remaining_points_gpr.items():

                current_cost = calculate_selected_point_cost(experiment_gpr, callpath, metric, calculate_cost_manual,
                                                             number_processes)

                # always take the first value in the list, until none left
                new_cost = current_cost + value[0]

                if new_cost <= budget:
                    fitting_measurements.append(key)

            # find the next best additional measurement point using the gpr
            best_index = -1
            best_rated = sys.float_info.max

            for i in range(len(fitting_measurements)):

                parameter_values = fitting_measurements[i].as_tuple()
                x = []

                for j in range(len(parameter_values)):

                    if len(normalization_factors) != 0:
                        x.append(parameter_values[j] * normalization_factors[str(experiment.parameters[j])])

                    else:
                        x.append(parameter_values[j])

                # term_1 is cost(t)^2
                term_1 = math.pow(remaining_points_gpr[fitting_measurements[i]][0], 2)
                # predict variance of input vector x with the gaussian process
                x = [x]
                _, y_cov = gaussian_process.predict(x, return_cov=True)
                y_cov = abs(y_cov)
                # term_2 is gp_cov(t,t)^2
                term_2 = math.pow(y_cov, 2)
                # rated is h(t)

                rep = 1
                rep = max_repetitions - len(predicted_runtimes[fitting_measurements[i]]) + 1

                rep_func = 2 ** ((1 / 2) * rep - (1 / 2))
                noise_func = -math.tanh((1 / 4) * mean_noise - 2.5)
                cost_multiplier = rep_func + noise_func
                rated = (term_1 * cost_multiplier) / term_2

                if rated <= best_rated:
                    best_rated = rated
                    best_index = i

                    # if there has been a point found that is suitable
            if best_index != -1:

                # add the identified measurement point to the selected point list
                parameter_values = fitting_measurements[best_index].as_tuple()
                cord = Coordinate(parameter_values)

                # only add coordinate to selected points list if not already in there (because of reps)
                if cord not in selected_points:
                    selected_points.append(cord)

                # add the new point to the gpr and call fit()
                gaussian_process = add_measurement_to_gpr(gaussian_process,
                                                          cord,
                                                          predicted_runtimes,
                                                          normalization_factors,
                                                          experiment_gpr.parameters)

                new_value = 0

                # remove the identified measurement point from the remaining point list
                try:
                    new_value = predicted_runtimes[cord][0]
                    predicted_runtimes[cord].pop(0)
                    if len(predicted_runtimes[cord]) == 0:
                        predicted_runtimes.pop(cord)

                    # pop value from cord in remaining points list that has been selected as best next point
                    remaining_points_gpr[cord].pop(0)

                    # pop cord from remaining points when no value left anymore
                    if len(remaining_points_gpr[cord]) == 0:
                        remaining_points_gpr.pop(cord)

                except KeyError:
                    pass

                # add this point to the gpr experiment
                experiment_gpr = create_experiment(cord, experiment_gpr, new_value, callpath, metric)

                # add cord and rep number to the list of suggestions
                rep_number = 1
                for i in range(len(experiment_gpr.measurements[(callpath, metric)])):
                    if cord == experiment_gpr.measurements[(callpath, metric)][i].coordinate:
                        rep_number = len(experiment_gpr.measurements[(callpath, metric)][i].values)
                        break
                suggested_points.append(cord)
                rep_numbers.append(rep_number)

                # update progress bar
                progress_bar.update(1)

            # if there are no suitable measurement points found
            # break the while True loop
            else:
                break

    return suggested_points, rep_numbers


def add_measurements_to_gpr(gaussian_process,
                            selected_coordinates,
                            measurements,
                            callpath,
                            metric,
                            normalization_factors,
                            parameters):
    X = []
    Y = []

    for coordinate in selected_coordinates:
        x = []
        parameter_values = coordinate.as_tuple()

        for j in range(len(parameter_values)):
            temp = 0
            if len(normalization_factors) != 0:
                temp = parameter_values[j] * normalization_factors[parameters[j]]
            else:
                temp = parameter_values[j]
                while temp < 1:
                    temp = temp * 10
            x.append(temp)

        for measurement in measurements[(callpath, metric)]:
            if measurement.coordinate == coordinate:
                Y.append(measurement.mean)
                break
        X.append(x)

    gaussian_process.fit(X, Y)

    return gaussian_process


def analyze_noise(experiment, callpath, metric):
    # use only measurements for the noise analysis where the coordinates have been selected
    selected_measurements = []
    for cord in experiment.coordinates:
        for measurement in experiment.measurements[(callpath, metric)]:
            if measurement.coordinate == cord:
                selected_measurements.append(measurement)
                break
    mean_noise_percentages = []
    # try to calculate the noise level on the measurements
    # if thats not possible because there are no repetitions available, assume its 1.0%
    try:
        for measurement in selected_measurements:
            mean_mes = np.mean(measurement.values)
            noise_percentages = []
            for val in measurement.values:
                if mean_mes == 0.0:
                    noise_percentages.append(0)
                else:
                    noise_percentages.append(abs((val / (mean_mes / 100)) - 100))
            mean_noise_percentages.append(np.mean(noise_percentages))
        mean_noise = np.mean(mean_noise_percentages)
    except TypeError:
        mean_noise = 1.0
    # print("mean_noise:",mean_noise,"%")
    return mean_noise


def get_normalization_factors(experiment):
    normalization_factors = {}
    for i in range(len(experiment.parameters)):
        param_value_max = -1
        for coord in experiment.coordinates:
            selected_measurements = coord.as_tuple()[i]
            if param_value_max < selected_measurements:
                param_value_max = selected_measurements
        param_value_max = 100 / param_value_max
        normalization_factors[str(experiment.parameters[i])] = param_value_max
    # print("normalization_factors:",normalization_factors)
    return normalization_factors


def create_experiment(cord, experiment, new_value, callpath, metric):
    # only append the new measurement value to experiment
    cord_found = False
    for i in range(len(experiment.measurements[(callpath, metric)])):
        if cord == experiment.measurements[(callpath, metric)][i].coordinate:
            experiment.measurements[(callpath, metric)][i].add_value(new_value)
            cord_found = True
            break
    if cord_found == False:
        # add new coordinate to experiment and then add a new measurement object with the new value to the experiment
        experiment.add_coordinate(cord)
        new_measurement = Measurement(cord, callpath, metric, [new_value], keep_values=True)
        experiment.add_measurement(new_measurement)
    return experiment


def calculate_selected_point_cost(experiment, callpath, metric, calculate_cost_manual, number_processes):
    selected_cost = 0
    for x in experiment.measurements[(callpath, metric)]:
        coordinate_cost = 0
        try:
            runtime = np.sum(x.values)
            if calculate_cost_manual:
                nr_processes = number_processes
            else:
                nr_processes = x.coordinate.as_tuple()[0]
            core_hours = runtime * nr_processes
            coordinate_cost += core_hours
        except TypeError:
            runtime = x.mean
            if calculate_cost_manual:
                nr_processes = number_processes
            else:
                nr_processes = x.coordinate.as_tuple()[0]
            core_hours = runtime * nr_processes
            coordinate_cost += core_hours
        selected_cost += coordinate_cost
    return selected_cost


def add_measurement_to_gpr(gaussian_process,
                           coordinate,
                           measurements,
                           normalization_factors,
                           parameters):
    X = []
    Y = []
    x = []

    parameter_values = coordinate.as_tuple()

    for j in range(len(parameter_values)):
        temp = 0
        if len(normalization_factors) != 0:
            temp = parameter_values[j] * normalization_factors[str(parameters[j])]
        else:
            temp = parameter_values[j]
            while temp < 1:
                temp = temp * 10
        x.append(temp)

    Y.append(np.mean(measurements[coordinate][0]))

    X.append(x)

    gaussian_process.fit(X, Y)

    return gaussian_process
