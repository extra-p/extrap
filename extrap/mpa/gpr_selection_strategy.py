# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import math
import numbers
import sys
import warnings
from collections.abc import Callable

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
                            calculate_cost: Callable[[tuple, numbers.Real], numbers.Real],
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

    selected_points = copy.deepcopy(experiment.coordinates)

    # GPR parameter-value normalization for each measurement point
    normalization_factors = get_normalization_factors(experiment)

    # use the values from the selected callpath
    if len(selected_callpaths) == 1:
        callpath = selected_callpaths[0]

        # do a noise analysis on the existing points
        mean_noise = analyze_noise(experiment, callpath, metric)

        gaussian_process = generate_gaussian_process(callpath, experiment, mean_noise, metric, normalization_factors,
                                                     selected_points)

        # construct a dict of potential measurement points with 
        # the coordinates as the keys and the costs as the values

        modeler = model_generator
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
            cost = calculate_cost(point, runtime)
            values = []
            for j in range(max_repetitions):
                values.append(cost)
            runtime_values = []
            for j in range(max_repetitions):
                runtime_values.append(runtime)
            remaining_points_gpr[possible_points[i]] = values
            predicted_runtimes[possible_points[i]] = runtime_values

        # extend the dict in case there are still reps missing from the base coordinates
        costs, repetitions, runtimes = extend_dict_from_base_coords(calculate_cost, callpath, experiment,
                                                                    metric, point)

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

        rep_numbers, suggested_points = search_points_with_gpr(budget, calculate_cost, callpath, experiment,
                                                               gaussian_process, max_repetitions, mean_noise, metric,
                                                               normalization_factors, predicted_runtimes, progress_bar,
                                                               remaining_points_gpr, selected_points)

    # sum up the values from the selected callpaths
    else:
        remaining_points_gpr = {}
        predicted_runtimes = {}
        mean_noise_values = []

        for l, callpath in enumerate(selected_callpaths):
            # do an noise analysis on the existing points
            mean_noise = analyze_noise(experiment, callpath, metric)
            mean_noise_values.append(mean_noise)

            # construct a dict of potential measurement points with
            # the coordinates as the keys and the costs as the values

            modeler = model_generator
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
                cost = calculate_cost(point, runtime)
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

            costs, repetitions, runtimes = extend_dict_from_base_coords(calculate_cost, callpath, experiment,
                                                                        metric, point)

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

        gaussian_process = generate_gaussian_process(callpath, experiment, mean_noise, metric, normalization_factors,
                                                     selected_points)

        rep_numbers, suggested_points = search_points_with_gpr(budget, calculate_cost, callpath, experiment,
                                                               gaussian_process, max_repetitions, mean_noise, metric,
                                                               normalization_factors, predicted_runtimes, progress_bar,
                                                               remaining_points_gpr, selected_points)

    return suggested_points, rep_numbers


def extend_dict_from_base_coords(calculate_cost, callpath, experiment, metric, point):
    # extend the dict in case there are still reps missing from the base coordinates
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
                        cost = calculate_cost(point, runtime)
                        cost_values.append(cost)
                        runtime_values.append(runtime)
                except TypeError:
                    runtime = experiment.measurements[(callpath, metric)][i].mean
                    cost = calculate_cost(point, runtime)
                    cost_values.append(cost)
                    runtime_values.append(runtime)
                costs[coordinate] = cost_values
                runtimes[coordinate] = runtime_values
                break
    return costs, repetitions, runtimes


def generate_gaussian_process(callpath, experiment, mean_noise, metric, normalization_factors, selected_points):
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
    temp_parameters = [str(parameter) for parameter in experiment.parameters]

    gaussian_process = add_measurements_to_gpr(gaussian_process,
                                               selected_points,
                                               experiment.measurements,
                                               callpath,
                                               metric,
                                               normalization_factors,
                                               temp_parameters)
    return gaussian_process


def search_points_with_gpr(budget, calculate_cost, callpath, experiment, gaussian_process, max_repetitions, mean_noise,
                           metric, normalization_factors, predicted_runtimes, progress_bar, remaining_points_gpr,
                           selected_points):
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

            current_cost = calculate_selected_point_cost(experiment_gpr, callpath, metric, calculate_cost)

            # always take the first value in the list, until none left
            new_cost = current_cost + value[0]

            if new_cost <= budget:
                fitting_measurements.append(key)

        # find the next best additional measurement point using the gpr

        best_index = find_next_best_point(experiment, fitting_measurements,
                                          gaussian_process, max_repetitions, mean_noise, normalization_factors,
                                          predicted_runtimes, remaining_points_gpr)
        if best_index != -1:

            experiment_gpr, gaussian_process = add_point_to_suggestions(best_index, callpath, experiment_gpr,
                                                                        fitting_measurements, gaussian_process,
                                                                        metric, normalization_factors,
                                                                        predicted_runtimes, progress_bar,
                                                                        remaining_points_gpr, rep_numbers,
                                                                        selected_points, suggested_points)

        # if there are no suitable measurement points found
        # break the while True loop
        else:
            break
    return rep_numbers, suggested_points


def add_point_to_suggestions(best_index, callpath, experiment_gpr, fitting_measurements, gaussian_process, metric,
                             normalization_factors, predicted_runtimes, progress_bar, remaining_points_gpr, rep_numbers,
                             selected_points, suggested_points):
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
    return experiment_gpr, gaussian_process


def find_next_best_point(experiment, fitting_measurements, gaussian_process, max_repetitions,
                         mean_noise, normalization_factors, predicted_runtimes, remaining_points_gpr):
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
    return best_index


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
    # if that is not possible because there are no repetitions available, assume its 1.0%
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
            selected_measurements = coord[i]
            if param_value_max < selected_measurements:
                param_value_max = selected_measurements
        param_value_max = 100 / param_value_max
        normalization_factors[str(experiment.parameters[i])] = param_value_max
    # print("normalization_factors:",normalization_factors)
    return normalization_factors


def create_experiment(cord, experiment, new_value, callpath, metric):
    # only append the new measurement value to experiment
    cord_found = False
    measurements = experiment.measurements[(callpath, metric)]
    for measurement in measurements:
        if cord == measurement.coordinate:
            measurement.add_value(new_value)
            cord_found = True
            break
    if not cord_found:
        # add new coordinate to experiment and then add a new measurement object with the new value to the experiment
        experiment.add_coordinate(cord)
        new_measurement = Measurement(cord, callpath, metric, [new_value], keep_values=True)
        experiment.add_measurement(new_measurement)
    return experiment


def calculate_selected_point_cost(experiment, callpath, metric, calculate_cost):
    selected_cost = 0
    for x in experiment.measurements[(callpath, metric)]:
        coordinate_cost = 0
        try:
            runtime = np.sum(x.values)
            core_hours = calculate_cost(x.coordinate, runtime)
            coordinate_cost += core_hours
        except TypeError:
            runtime = x.mean
            core_hours = calculate_cost(x.coordinate, runtime)
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
