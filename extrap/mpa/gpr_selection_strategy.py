# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
import math
import numbers
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.metric import Metric
from extrap.util.progress_bar import DUMMY_PROGRESS


@dataclass
class MeanRepPair:
    mean: numbers.Real
    repetitions: int

    def __iadd__(self, other: MeanRepPair) -> MeanRepPair:
        self.mean += other.mean
        self.repetitions = min(self.repetitions, other.repetitions)
        return self

    def __add__(self, other: MeanRepPair) -> MeanRepPair:
        return MeanRepPair(self.mean + other.mean, min(self.repetitions, other.repetitions))

    def __iter__(self):
        yield self.mean
        yield self.repetitions


def suggest_points_gpr_mode(experiment: Experiment,
                            possible_points: Sequence[Coordinate],
                            selected_callpaths: Sequence[Callpath],
                            metric: Metric,
                            calculate_cost: Callable[[Coordinate, numbers.Real], numbers.Real],
                            budget,
                            current_cost,
                            model_generator,
                            progress_bar=DUMMY_PROGRESS, *, random_state=None):
    progress_bar.total = 100
    progress_bar.step("Suggesting additional measurement points.")

    # GPR parameter-reps normalization for each measurement point
    normalization_factors = get_normalization_factors(experiment)
    max_repetitions = 5

    with warnings.catch_warnings():
        # disable warnings from sk
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        remaining_points_gpr = {}
        predicted_runtimes = {}
        mean_noise_values = []

        existing_measurements_dict: dict[Coordinate, MeanRepPair] = {}

        for l, callpath in enumerate(selected_callpaths):
            # do an noise analysis on the existing points
            mean_noise = analyze_noise(experiment, callpath, metric)
            mean_noise_values.append(mean_noise)

            # construct a dict of potential measurement points with
            # the coordinates as the keys and the costs as the values

            model = model_generator.models[callpath, metric]
            hypothesis = model.hypothesis
            function = hypothesis.function

            # first use the points from the search space
            for point in possible_points:
                # c.1 predict the runtime of the possible_points using the existing performance models
                runtime = function.evaluate(point.as_tuple())
                # c.2 calculate the cost of these points using the runtime (same calculation as for the current cost
                # in the GUI)
                cost = calculate_cost(point, runtime)

                if point in remaining_points_gpr:
                    for c, r in zip(remaining_points_gpr[point], predicted_runtimes[point]):
                        c += cost
                        r += runtime
                else:
                    remaining_points_gpr[point] = [cost] * max_repetitions
                    predicted_runtimes[point] = [runtime] * max_repetitions

            for measurement in experiment.measurements[(callpath, metric)]:
                # prepare dictionary of existing measurements
                if measurement.coordinate not in existing_measurements_dict:
                    existing_measurements_dict[measurement.coordinate] = MeanRepPair(measurement.mean,
                                                                                     measurement.repetitions)
                else:
                    existing_measurements_dict[measurement.coordinate] += MeanRepPair(measurement.mean,
                                                                                      measurement.repetitions)
                # extend the dict in case there are still reps missing from the base coordinates
                remaining_reps = max_repetitions - measurement.repetitions
                if remaining_reps > 0:
                    mean_runtime = measurement.mean
                    mean_cost = calculate_cost(measurement.coordinate, mean_runtime)
                    values = [mean_cost] * remaining_reps
                    runtime_values = [mean_runtime] * remaining_reps
                    if point in remaining_points_gpr:
                        for remaining_point_value in remaining_points_gpr[point]:
                            remaining_point_value += np.mean(values)
                        for predicted_runtime in predicted_runtimes[point]:
                            predicted_runtime += np.mean(runtime_values)
                    else:
                        remaining_points_gpr[point] = values
                        predicted_runtimes[point] = runtime_values

        # calculate mean noise from found noise values for different callpaths
        mean_noise = np.mean(mean_noise_values)

        gaussian_process = generate_gaussian_process(existing_measurements_dict, mean_noise, normalization_factors,
                                                     random_state=random_state)

        rep_numbers, suggested_points = search_points_with_gpr(budget, calculate_cost, gaussian_process,
                                                               max_repetitions, mean_noise, normalization_factors,
                                                               predicted_runtimes, progress_bar, remaining_points_gpr,
                                                               existing_measurements_dict)

    return suggested_points, rep_numbers


def generate_gaussian_process(existing_measurements: dict[Coordinate, MeanRepPair], mean_noise, normalization_factors,
                              random_state=None):
    # add all of the selected measurement points to the gaussian process
    # as training data and train it for these points
    xs = []
    ys = []

    for coordinate, (mean, _) in existing_measurements.items():
        x = []
        for p, parameter_value in enumerate(coordinate):
            if len(normalization_factors) != 0:
                temp = parameter_value * normalization_factors[p]
            else:
                temp = parameter_value
                while temp < 1:
                    temp = temp * 10
            x.append(temp)
        xs.append(x)
        ys.append(mean)

    # nu should be [0.5, 1.5, 2.5, inf], everything else has 10x overhead
    # matern kernel + white kernel to simulate actual noise found in the measurements
    kernel = 1 * Matern(length_scale=1, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel(
        noise_level=mean_noise * mean_noise, noise_level_bounds=(1e-5, 1e5))

    # create a gaussian process regressor
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=random_state)
    gaussian_process.fit(xs, ys)

    return gaussian_process


def display_gaussian_process(gaussian_process):
    xs = gaussian_process.X_train_
    ys = gaussian_process.y_train_

    X = np.linspace(start=np.min(xs), stop=np.max(xs) * 3, num=1_000).reshape(-1, 1)
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    plt.scatter(xs, ys, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.show()


def search_points_with_gpr(budget, calculate_cost, gaussian_process, max_repetitions, mean_noise, normalization_factors,
                           predicted_runtimes, progress_bar, remaining_points_gpr,
                           existing_measurements: dict[Coordinate, MeanRepPair]):
    gpr_measurements = copy.copy(existing_measurements)
    suggested_points = []
    rep_numbers = []

    # additional limit for the loop
    # otherwise takes too long or the app might crash if set budget is too large
    # since performing the GPR is really resource intensive
    while len(suggested_points) < 100:

        current_cost = get_cost_of_measurements(calculate_cost, gpr_measurements)
        # identify all possible next points that would
        # still fit into the modeling budget in core time
        fitting_measurements = []
        for point, cost in remaining_points_gpr.items():

            # always take the first value in the list, until none left
            # deletion of values happens in add_point_to_suggestions
            new_cost = current_cost + cost[0]

            if new_cost <= budget:
                fitting_measurements.append(point)

        # find the next best additional measurement point using the gpr

        best_coordinate = find_next_best_point(fitting_measurements, gaussian_process, max_repetitions, mean_noise,
                                               normalization_factors, remaining_points_gpr)
        if best_coordinate is not None:
            cord = Coordinate(best_coordinate)
            # add the new point to the gpr and call fit()
            gaussian_process = add_measurement_to_gpr(gaussian_process, cord, predicted_runtimes, normalization_factors)
            new_value = remove_point_from_remaining_points(cord, predicted_runtimes, remaining_points_gpr)
            # add this point to the gpr measurements
            gpr_measurements = update_gpr_measurements(cord, gpr_measurements, new_value)
            # add cord and rep number to the list of suggestions

            rep_number = gpr_measurements[cord].repetitions
            suggested_points.append(cord)
            rep_numbers.append(rep_number)
            # update progress bar
            progress_bar.update(1)

        # if there are no suitable measurement points found
        # break the while True loop
        else:
            break
    return rep_numbers, suggested_points


def get_cost_of_measurements(calculate_cost, gpr_measurements):
    current_cost = sum(calculate_cost(coordinate, value) * repetitions for coordinate, (value, repetitions) in
                       gpr_measurements.items())
    return current_cost


def remove_point_from_remaining_points(cord, predicted_runtimes, remaining_points_gpr):
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
    return new_value


def find_next_best_point(fitting_coordinates, gaussian_process, max_repetitions, mean_noise, normalization_factors,
                         remaining_points_gpr):
    best_coordinate = None
    best_rated = math.inf
    for coordinate in fitting_coordinates:

        x = []
        for p, parameter_value in enumerate(coordinate):
            if len(normalization_factors) != 0:
                x.append(parameter_value * normalization_factors[p])
            else:
                x.append(parameter_value)

        # term_1 is cost(t)^2
        term_1 = math.pow(remaining_points_gpr[coordinate][0], 2)
        # predict variance of input vector x with the gaussian process
        x = [x]
        _, y_cov = gaussian_process.predict(x, return_cov=True)
        y_cov = abs(y_cov)
        # term_2 is gp_cov(t,t)^2
        term_2 = math.pow(y_cov, 2)
        # rated is h(t)

        rep = max_repetitions - len(remaining_points_gpr[coordinate]) + 1

        rep_func = 2 ** ((1 / 2) * rep - (1 / 2))
        noise_func = -math.tanh((1 / 4) * mean_noise - 2.5)
        cost_multiplier = rep_func + noise_func
        rated = (term_1 * cost_multiplier) / term_2

        if rated <= best_rated:
            best_rated = rated
            best_coordinate = coordinate

            # if there has been a point found that is suitable
    return best_coordinate


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
            mean_mes = measurement.mean
            noise_percentages = []
            for val in measurement.values:
                if mean_mes == 0.0:
                    noise_percentages.append(0)
                else:
                    noise_percentages.append(abs((val / mean_mes) - 1))
            mean_noise_percentages.append(np.mean(noise_percentages))
        mean_noise = np.mean(mean_noise_percentages)
    except TypeError:
        mean_noise_percentages = []
        for measurement in selected_measurements:
            if measurement.mean != 0.0:
                mean_noise_percentages.append(measurement.std / measurement.mean)
        mean_noise = np.mean(mean_noise_percentages)
        # mean_noise = 1.0
    # print("mean_noise:",mean_noise,"%")
    return mean_noise


def get_normalization_factors(experiment):
    normalization_factors = [-1] * len(experiment.parameters)
    for i in range(len(experiment.parameters)):
        param_value_max = -1
        for coord in experiment.coordinates:
            selected_measurements = coord[i]
            if param_value_max < selected_measurements:
                param_value_max = selected_measurements
        param_value_max = 100 / param_value_max
        normalization_factors[i] = param_value_max
    # print("normalization_factors:",normalization_factors)
    return normalization_factors


def update_gpr_measurements(cord, gpr_measurements: dict[Coordinate, MeanRepPair], new_value):
    # only append the new measurement value to the set of gpr_measurements if the coordinate does not exist
    # update when measurement with same coordinate exists
    if cord in gpr_measurements:
        m = gpr_measurements[cord]
        m.mean = (m.mean * m.repetitions + new_value) / (m.repetitions + 1)
        m.repetitions += 1
    else:
        # add a new measurement object with the new value to the set of gpr_measurements
        gpr_measurements[cord] = MeanRepPair(new_value, 1)
    return gpr_measurements


def add_measurement_to_gpr(gaussian_process, coordinate, measurements, normalization_factors):
    x = []
    for p, parameter_value in enumerate(coordinate):
        if len(normalization_factors) != 0:
            temp = parameter_value * normalization_factors[p]
        else:
            temp = parameter_value
            while temp < 1:
                temp = temp * 10
        x.append(temp)

    y = np.mean(measurements[coordinate])

    gaussian_process.fit([x], [y])

    return gaussian_process
