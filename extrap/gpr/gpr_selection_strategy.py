# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np
import copy
from extrap.entities.parameter import Parameter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
import warnings
from sklearn.exceptions import ConvergenceWarning
from extrap.entities.coordinate import Coordinate


#TODO: finish this code!
#TODO: create test data
#TODO: test for 1,3,4 parameters
def suggest_points_gpr_mode(experiment, parameter_value_series, possible_points, selected_callpaths, metric, calculate_cost_manual, process_parameter_id, number_processes, budget, current_cost):
    
    # c.1 predict the runtime of these points using the existing performance models (only possible if already enough points existing for modeling)
    # c.2 calculate the cost of these points using the runtime (same calculation as for the current cost in the GUI)
    #NOTE: the search space points should have a dict like for the costs of the remaining points for my case study analysis...
    # c.3 all of the data is used as input to the GPR method
    # c.4 get the top x points suggested by the GPR method that do fit into the available budget
    # c.5 create coordinates and suggest them

    # disable warnings from sk
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # sum up the values from the selected callpaths
    if len(selected_callpaths) > 1:
        suggested_points = []
    
    # use the values from the selected callpath
    elif len(selected_callpaths) == 1:
        callpath = selected_callpaths[0]
        callpath_id = 0
        selected_points = copy.deepcopy(experiment.coordinates)

        # GPR parameter-value normalization for each measurement point
        normalization_factors = get_normalization_factors(experiment)

        # do an noise analysis on the existing points
        mean_noise = analyze_noise(experiment, callpath, metric)

        # nu should be [0.5, 1.5, 2.5, inf], everything else has 10x overhead
        # matern kernel + white kernel to simulate actual noise found in the measurements
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel(noise_level=mean_noise)

        # create a gaussian process regressor
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=20
        )

        # add all of the selected measurement points to the gaussian process
        # as training data and train it for these points
        temp_parameters = []
        for parameter in experiment.parameters:
            temp_parameters.append(Parameter(parameter))
        gaussian_process = add_measurements_to_gpr(gaussian_process, 
                        selected_points, 
                        experiment.measurements, 
                        callpath,
                        metric,
                        normalization_factors,
                        temp_parameters)
        

        """
        while True:
            
            # identify all possible next points that would 
            # still fit into the modeling budget in core hours
            fitting_measurements = []
            for key, value in remaining_points_gpr.items():
                
                #current_cost = calculate_selected_point_cost(selected_points_gpr, experiment_gpr_base, metric_id, callpath_id)
                current_cost = calculate_selected_point_cost2(experiment_gpr_base, callpath, metric)
                
                # always take the first value in the list, until none left
                #new_cost = current_cost + np.sum(value)
                new_cost = current_cost + value[0]
                if total_cost == 0.0:
                    cost_percent = 0.0
                else:
                    cost_percent = new_cost / (total_cost / 100)
                
                #if new_cost > budget_core_hours:
                #    print("new_cost <= budget_core_hours:", new_cost, budget_core_hours)
                #if cost_percent > 100:
                #    print("cost percent <= budget percent:", cost_percent, budget)
                # to make sure no mistakes occur here
                # sometimes the numbers do not perfectly add up to the target budget
                # but to 100.00001
                # this is the fix for this case
                cost_percent = float("{0:.3f}".format(cost_percent))
                if cost_percent > 100.0:
                    cost_percent = 100.0

                if cost_percent <= budget:
                    fitting_measurements.append(key)

            #print("fitting_measurements:",fitting_measurements)

            # find the next best additional measurement point using the gpr
            best_index = -1
            best_rated = sys.float_info.max

            for i in range(len(fitting_measurements)):
            
                parameter_values = fitting_measurements[i].as_tuple()
                x = []
                
                for j in range(len(parameter_values)):
                
                    if len(normalization_factors) != 0:
                        x.append(parameter_values[j] * normalization_factors[experiment_gpr_base.parameters[j]])
                
                    else:
                        x.append(parameter_values[j])
                        
                #NOTE: should recalculate the noise level here... but is already done on all measurements before so not needed here....
                # but in real extra-P implementation needs to be done...
                        
                #print("DEBUG3 remaining_points_gpr:",remaining_points_gpr[fitting_measurements[i]][0])
                
                # term_1 is cost(t)^2
                term_1 = math.pow(remaining_points_gpr[fitting_measurements[i]][0], 2)
                # predict variance of input vector x with the gaussian process
                x = [x]
                _, y_cov = gaussian_process.predict(x, return_cov=True)
                y_cov = abs(y_cov)
                # term_2 is gp_cov(t,t)^2
                term_2 = math.pow(y_cov, 2)
                # rated is h(t)
                
                if grid_search == 3 or grid_search == 4:
                    rep = 1
                    for j in range(len(measurements_gpr[(callpath, metric)])):
                        if measurements_gpr[(callpath, metric)][j].coordinate == fitting_measurements[i]:
                            rep = (nr_repetitions - len(measurements_gpr[(callpath, metric)][j].values)) + 1
                            break
                    rep_func = 2**((1/2)*rep-(1/2))
                    noise_func = -math.tanh((1/4)*mean_noise-2.5)
                    cost_multiplier = rep_func + noise_func
                    rated = (term_1 * cost_multiplier) / term_2
                else:
                    rated = term_1 / term_2

                if rated <= best_rated:
                    best_rated = rated
                    best_index = i    

            # if there has been a point found that is suitable
            if best_index != -1:

                # add the identified measurement point to the selected point list
                parameter_values = fitting_measurements[best_index].as_tuple()
                cord = Coordinate(parameter_values)
                #selected_points_gpr.append(cord)
                
                # only add coordinate to selected points list if not already in there (because of reps)
                if cord not in selected_points_gpr:
                    selected_points_gpr.append(cord)
                
                # add the new point to the gpr and call fit()
                gaussian_process = add_measurement_to_gpr(gaussian_process, 
                        cord, 
                        measurements_gpr,
                        callpath, 
                        metric,
                        normalization_factors,
                        experiment_gpr_base.parameters)
                
                new_value = 0
                
                # remove the identified measurement point from the remaining point list
                try:
                    # only pop cord when there are no values left in the measurement
                    
                    # if that's not the case pop the value from the measurement of the cord
                    measurement = None
                    cord_id = None
                    for i in range(len(measurements_gpr[(callpath, metric)])):
                        if measurements_gpr[(callpath, metric)][i].coordinate == cord:
                            cord_id = i
                            x = measurements_gpr[(callpath, metric)][i].values
                            #print("DEBUG 5:",len(x))
                            if len(x) > 0:
                                new_value = np.mean(x[0])
                                x = np.delete(x, 0, 0)
                                measurements_gpr[(callpath, metric)][i].values = x
                            break
                    
                    # pop value from cord in remaining points list that has been selected as best next point
                    remaining_points_gpr[cord].pop(0)
                    
                    # pop cord from remaining points when no value left anymore
                    if len(measurements_gpr[(callpath, metric)][cord_id].values) == 0:
                        remaining_points_gpr.pop(cord)
                    
                except KeyError:
                    pass

                # update the number of additional points used
                add_points_gpr += 1

                # add this point to the gpr experiment
                #experiment_gpr_base = create_experiment(selected_points_gpr, experiment_gpr_base, len(experiment_gpr_base.parameters), parameters, metric_id, callpath_id)
                experiment_gpr_base = create_experiment2(cord, experiment_gpr_base, new_value, callpath, metric)

            # if there are no suitable measurement points found
            # break the while True loop
            else:
                break
        
        """
        


        suggested_points = []

    # sum all callpaths runtime
    elif len(selected_callpaths) == 0:
        suggested_points = []

    return suggested_points


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
    print("mean_noise:",mean_noise,"%")
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
        normalization_factors[Parameter(experiment.parameters[i])] = param_value_max
    print("normalization_factors:",normalization_factors)
    return normalization_factors



def analyze_callpath(self, inputs):
        
    # get the values from the parallel input dict
    cost = inputs[2]
    cost_container = inputs[4]
    total_costs_container = inputs[5]
    experiment_measurements = inputs[6]
    nr_parameters = inputs[7]
    experiment_coordinates = inputs[8]
    metric = inputs[9]
    #base_values = inputs[11]
    metric_id = inputs[10]
    #nr_repetitions = inputs[13]
    parameters = inputs[11]
    #args = inputs[15]
    budget = inputs[12]
    #eval_point = inputs[17]
    #all_points_functions_strings = inputs[18]
    #coordinate_evaluation = inputs[19]
    #measurement_evaluation = inputs[20]
    normalization = inputs[13]
    min_points = inputs[14]
    #hybrid_switch = inputs[23]
    result_container = {}

    #callpath_string = callpath.name

    # get the cost values for this particular callpath
    #cost = cost_container[callpath_string]
    #total_cost = total_costs_container[callpath_string]

    # create copy of the cost dict
    #remaining_points = copy.deepcopy(cost)

    ##########################
    ## Base point selection ##
    ##########################

    # create copy of the cost dict
    #remaining_points = copy.deepcopy(cost)
    
    # create copy of the cost dict for the minimum experiment with gpr and hybrid strategies
    #remaining_points_min = copy.deepcopy(cost)

    #measurements_gpr = copy.deepcopy(experiment_measurements)
    #measurements_hybrid = copy.deepcopy(experiment_measurements)



    #shared_dict[callpath_id] = result_container

            
    #with Pool(cpu_count) as pool:
    #    _ = list(tqdm(pool.imap(self.analyze_callpath, inputs), total=len(self.selected_callpath_ids), disable=False))

    #result_dict = copy.deepcopy(shared_dict)
    
    #print("DEBUG:",result_dict)

    #NOTE: base_values, always use all available reps of a point!

    #NOTE: the number of available repetitions needs to be calculated for each
    # individual measurement point

    """# identify the number of repetitions per measurement point
    nr_repetitions = 1
    measurements = self.experiment.measurements
    try:
        nr_repetitions = len(measurements[(selected_callpath[0].path, runtime_metric)].values)
    except TypeError:
        pass
    except KeyError:
        pass"""


