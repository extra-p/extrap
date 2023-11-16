import matplotlib.pyplot as plt
from math import log2
import extrap
from extrap.entities.parameter import Parameter
from extrap.entities.metric import Metric
from extrap.entities.callpath import Callpath
from extrap.entities.experiment import Experiment
from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement
from extrap.util.progress_bar import ProgressBar
from extrap.modelers.model_generator import ModelGenerator
import argparse
from extrap.modelers import multi_parameter
from extrap.modelers import single_parameter
from itertools import chain
from extrap.util.options_parser import ModelerOptionsAction, ModelerHelpAction
from extrap.modelers.abstract_modeler import MultiParameterModeler
from extrap.util.options_parser import SINGLE_PARAMETER_MODELER_KEY, SINGLE_PARAMETER_OPTIONS_KEY
from extrap.entities.model import Model

def plot(parameter_values, changing_point, g, f, default_function_string, functions, latex_functions, changing_points, default_function_string_latex):
    plt.rcParams["text.usetex"] = True
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rcParams["figure.figsize"] = (8,4)
    min_x = min(parameter_values)
    max_x = max(parameter_values)
    steps = 1000
    if isinstance(changing_points[0], Measurement):
        range_x = (max_x - min_x)
        stepsize = range_x / steps
        y_values_3 = []
        x_values_2 = []
    elif changing_points[0] == None:
        range_x = (max_x - min_x)
        stepsize = range_x / steps
        y_values_3 = []
        x_values_2 = []
    else:
        range_x = (max_x - min_x)
        stepsize = range_x / steps
        param_value_1 = changing_points[0][0].coordinate._values[0]
        param_value_2 = changing_points[0][1].coordinate._values[0]
        range_x2 = (max_x - min_x) - (param_value_2 - param_value_1)
        stepsize2 = range_x2 / steps
        y_values_3_1 = []
        y_values_3_2 = []
        x_values_2_1 = []
        x_values_2_2 = []

    y_values = []
    y_values_2 = []
    x_values = []
    x_value = min_x
    
    for i in range(steps):
        x_values.append(x_value)
        p = x_value
        if p >= changing_point:
            y_values.append(eval(g))
        else:
            y_values.append(eval(f))
        y_values_2.append(eval(default_function_string))
        x_value += stepsize

    if isinstance(changing_points[0], Measurement):
        x_value = min_x
        param_value = changing_points[0].coordinate._values[0]
        for i in range(steps):
            x_values_2.append(x_value)
            p = x_value
            if p >= param_value:
                y_values_3.append(eval(functions[1]))
            else:
                y_values_3.append(eval(functions[0]))
            x_value += stepsize
    elif changing_points[0] == None:
        x_value = min_x
        for i in range(steps):
            x_values_2.append(x_value)
            p = x_value
            y_values_3.append(eval(functions[0]))
            x_value += stepsize
    else:
        param_value_1 = changing_points[0][0].coordinate._values[0]
        param_value_2 = changing_points[0][1].coordinate._values[0]
        x_value = min_x
        for i in range(steps):
            p = x_value
            if p <= param_value_1:
                y_values_3_1.append(eval(functions[0]))
                x_values_2_1.append(x_value)
            elif p >= param_value_1 and p <= param_value_2:
                x_value = param_value_2
                #y_values_3.append(0)
            elif p >= param_value_2:
                x_values_2_2.append(x_value)
                y_values_3_2.append(eval(functions[1]))
            x_value += stepsize2
    
    measurements = []
    for i in range(len(parameter_values)):
        p = parameter_values[i]
        if p >= changing_point:
            measurements.append(eval(g))
        else:
            measurements.append(eval(f))
    axes = plt.figure().add_subplot(111)
    if changing_points[0] != None:
        latex_functions[0] = latex_functions[0][1:]
        latex_functions[0] = latex_functions[0][:-1]
        latex_functions[1] = latex_functions[1][1:]
        latex_functions[1] = latex_functions[1][:-1]
    if isinstance(changing_points[0], Measurement):
        param_value = changing_points[0].coordinate._values[0]
        formulae = r"$f(x)=\begin{cases}"+latex_functions[0]+r" & x\le "+str(param_value)+r"\\"+latex_functions[1]+r" & x\ge "+str(param_value)+r"\end{cases}$"
    elif changing_points[0] == None:
        formulae = latex_functions[0]
    else:
        param_value_1 = changing_points[0][0].coordinate._values[0]
        param_value_2 = changing_points[0][1].coordinate._values[0]
        formulae = r"$f(x)=\begin{cases}"+latex_functions[0]+r" & x\le "+str(param_value_1)+r"\\"+latex_functions[1]+r" & x\ge "+str(param_value_2)+r"\end{cases}$"
    plt.plot(x_values, y_values_2, color="red", label="default modeler\n"+str(default_function_string_latex))
    
    if isinstance(changing_points[0], Measurement):
        plt.plot(x_values_2, y_values_3, color="blue", label="segmented modeler\n"+formulae)
    elif changing_points[0] == None:
        plt.plot(x_values_2, y_values_3, color="blue", label="segmented modeler\n"+formulae)
    else:
        plt.plot(x_values_2_1, y_values_3_1, color="blue", label="segmented modeler\n"+formulae)
        plt.plot(x_values_2_2, y_values_3_2, color="blue")

    plt.xlim(min_x-(min_x/2), max_x+1)
    #plt.xscale("log")
    plt.xticks(parameter_values)
    axes.set_xticklabels(parameter_values)
    plt.minorticks_off()
    plt.scatter(parameter_values, measurements, color="black", label="measurements")
    for i in range(len(parameter_values)):
        value = '{0:.2f}'.format(measurements[i])
        plt.annotate("("+str(parameter_values[i])+", "+value+")", (parameter_values[i]-0.5, measurements[i] + 2), fontsize="8")
    plt.ylabel("Runtime $t$")
    plt.xlabel("Number of processes $p$")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def create_experiment(g, f, parameter_values, changing_point):
    parameter = Parameter("p")
    metric = Metric("runtime")
    callpath = Callpath("main")
    experiment = Experiment()
    experiment.add_callpath(callpath)
    experiment.add_metric(metric)
    experiment.add_parameter(parameter)
    for i in range(len(parameter_values)):
        coordinate = Coordinate(parameter_values[i])
        experiment.add_coordinate(coordinate)
        p = parameter_values[i]
        if p >= changing_point:
            metric_value = eval(g)
        else:
            metric_value = eval(f)
        experiment.add_measurement(Measurement(coordinate, callpath, metric, metric_value))
    return experiment

def get_default_model(experiment, arguments):
    model_generator = ModelGenerator(experiment, modeler="DEFAULT", name="Default", use_median=True)
    # apply modeler options
    modeler = model_generator.modeler
    if isinstance(modeler, MultiParameterModeler) and arguments.modeler_options:
        # set single-parameter modeler of multi-parameter modeler
        single_modeler = arguments.modeler_options[SINGLE_PARAMETER_MODELER_KEY]
        if single_modeler is not None:
            modeler.single_parameter_modeler = single_parameter.all_modelers[single_modeler]()
        # apply options of single-parameter modeler
        if modeler.single_parameter_modeler is not None:
            for name, value in arguments.modeler_options[SINGLE_PARAMETER_OPTIONS_KEY].items():
                if value is not None:
                    setattr(modeler.single_parameter_modeler, name, value)
    for name, value in arguments.modeler_options.items():
        if value is not None:
            setattr(modeler, name, value)
    with ProgressBar(desc='Generating models', disable=True) as pbar:
        model_generator.model_all(pbar)
    modeler = experiment.modelers[0]
    models = modeler.models
    model = models[(Callpath("main"), Metric("runtime"))]
    hypothesis = model.hypothesis
    default_function = hypothesis.function
    default_function_string = default_function.to_string(*experiment.parameters)
    default_function_string_latex = default_function.to_latex_string(*experiment.parameters)
    return default_function_string, default_function_string_latex

def get_segmented_model(experiment, arguments):
    model_generator = ModelGenerator(experiment, modeler="SEGMENTED", name="Segmented", use_median=True)
    # apply modeler options
    modeler = model_generator.modeler
    if isinstance(modeler, MultiParameterModeler) and arguments.modeler_options:
        # set single-parameter modeler of multi-parameter modeler
        single_modeler = arguments.modeler_options[SINGLE_PARAMETER_MODELER_KEY]
        if single_modeler is not None:
            modeler.single_parameter_modeler = single_parameter.all_modelers[single_modeler]()
        # apply options of single-parameter modeler
        if modeler.single_parameter_modeler is not None:
            for name, value in arguments.modeler_options[SINGLE_PARAMETER_OPTIONS_KEY].items():
                if value is not None:
                    setattr(modeler.single_parameter_modeler, name, value)
    for name, value in arguments.modeler_options.items():
        if value is not None:
            setattr(modeler, name, value)
    with ProgressBar(desc='Generating models', disable=True) as pbar:
        model_generator.model_all(pbar)
    modeler = experiment.modelers[0]
    models = modeler.models
    model = models[(Callpath("main"), Metric("runtime"))]
    functions = []
    latex_functions = []
    changing_points = []
    if isinstance(model, Model):
        func = model.hypothesis.function
        functions.append(func.to_string(*experiment.parameters))
        latex_functions.append(func.to_latex_string(*experiment.parameters))
        changing_points.append(None)
    else:
        for m in model:
            func = m.hypothesis.function
            functions.append(func.to_string(*experiment.parameters))
            latex_functions.append(func.to_latex_string(*experiment.parameters))
            changing_points.append(m.changing_point)
    return functions, latex_functions, changing_points

def test1(arguments):
    parameter_values = [1,2,3,4,5,6,7,8,9,10]
    f = "p**2"
    g = "30+p"
    changing_point = 6
    experiment = create_experiment(g, f, parameter_values, changing_point)
    default_function_string, default_function_string_latex = get_default_model(experiment, arguments)
    experiment2 = create_experiment(g, f, parameter_values, changing_point)
    functions, latex_functions, changing_points = get_segmented_model(experiment2, arguments)
    plot(parameter_values, changing_point, g, f, default_function_string, functions, latex_functions, changing_points, default_function_string_latex)

def test2(arguments):
    parameter_values = [4,8,12,16,20,24,28,32,36,40]
    f = "log2(p)**1"
    g = "p**2"
    changing_point = 22
    experiment = create_experiment(g, f, parameter_values, changing_point)
    default_function_string, default_function_string_latex = get_default_model(experiment, arguments)
    experiment2 = create_experiment(g, f, parameter_values, changing_point)
    functions, latex_functions, changing_points = get_segmented_model(experiment2, arguments)
    plot(parameter_values, changing_point, g, f, default_function_string, functions, latex_functions, changing_points, default_function_string_latex)

def test3(arguments):
    parameter_values = [10,20,30,40,50,60,70,80,90,100]
    f = "p**2*log2(p)**1"
    g = "p**2*log2(p)**1"
    changing_point = 50
    experiment = create_experiment(g, f, parameter_values, changing_point)
    default_function_string, default_function_string_latex = get_default_model(experiment, arguments)
    experiment2 = create_experiment(g, f, parameter_values, changing_point)
    functions, latex_functions, changing_points = get_segmented_model(experiment2, arguments)
    plot(parameter_values, changing_point, g, f, default_function_string, functions, latex_functions, changing_points, default_function_string_latex)


def main(args=None, prog=None):
    # argparse
    modelers_list = list(set(k.lower() for k in
                             chain(single_parameter.all_modelers.keys(), multi_parameter.all_modelers.keys())))
    parser = argparse.ArgumentParser(prog=prog, description=extrap.__description__, add_help=False)
    modeling_options = parser.add_argument_group("Modeling options")
    modeling_options.add_argument("--median", action="store_true", dest="median",
                                  help="Use median values for computation instead of mean values")
    modeling_options.add_argument("--modeler", action="store", dest="modeler", default='default', type=str.lower,
                                  choices=modelers_list,
                                  help="Selects the modeler for generating the performance models")
    modeling_options.add_argument("--options", dest="modeler_options", default={}, nargs='+', metavar="KEY=VALUE",
                                  action=ModelerOptionsAction,
                                  help="Options for the selected modeler")
    modeling_options.add_argument("--help-modeler", choices=modelers_list, type=str.lower,
                                  help="Show help for modeler options and exit",
                                  action=ModelerHelpAction)
    arguments = parser.parse_args(args)
    

    test1(arguments)
    
    test2(arguments)

    test3(arguments)
    

if __name__ == "__main__":
    main()