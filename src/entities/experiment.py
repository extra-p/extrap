class Experiment:
    
    
    def __init__(self):
        self.callpaths = []
        self.metrics = []
        self.parameters = []
        self.coordinates = []
        self.measurements = []
        self.call_tree = None
        self.modeler = []
        self.scaling = None
    
    
    def set_scaling(self, scaling_type):
        self.scaling = scaling_type
        
    
    def get_scaling(self):
        return self.scaling
    
    
    def get_modeler(self, modeler_id):
        return self.modeler[modeler_id]
    
    
    def add_modeler(self, modeler):
        self.modeler.append(modeler)
    
    
    def get_new_modeler_id(self):
        return len(self.modeler)+1
    
    
    def get_call_tree(self):
        return self.call_tree
    
    
    def add_call_tree(self, call_tree):
        self.call_tree = call_tree
        

    def add_metric(self, metric):
        self.metrics.append(metric)


    def get_metric(self, metric_id):
        return self.metrics[metric_id]
    
    
    def get_metrics(self):
        return self.metrics
    
    
    def get_len_metrics(self):
        return len(self.metrics)
    
    
    def get_metric_id(self, metric_name):
        for metric_id in range(len(self.metrics)):
            if self.metrics[metric_id].get_name() == metric_name:
                return metric_id
        return -1


    def metric_exists(self, metric_name):
        for metric_id in range(len(self.metrics)):
            if self.metrics[metric_id].get_name() == metric_name:
                return True
        return False
    
    
    def add_parameter(self, parameter):
        self.parameters.append(parameter)


    def get_parameter_id(self, parameter_name):
        for parameter_id in range(len(self.parameters)):
            if self.parameters[parameter_id].get_name() == parameter_name:
                return parameter_id
        return -1
    
    
    def parameter_exists(self, parameter_name):
        for parameter_id in range(len(self.parameters)):
            if self.parameters[parameter_id].get_name() == parameter_name:
                return True
        return False
    
    
    def get_parameter(self, parameter_id):
        return self.parameters[parameter_id]
    
    
    def get_parameters(self):
        return self.parameters
    
    
    def add_coordinate(self, coordinate):
        self.coordinates.append(coordinate)


    def get_coordinate(self, coordinate_id):
        return self.coordinates[coordinate_id]
    
    
    def get_coordinates(self):
        return self.coordinates
    
    
    def get_coordinate_id(self, coordinate):
        for coordinate_id in range(len(self.coordinates)):
            if self.coordinates[coordinate_id].get_as_string() == coordinate.get_as_string():
                return coordinate_id
        return -1
    
    
    def coordinate_exists(self, coordinate):
        for coordinate_id in range(len(self.coordinates)):
            if self.coordinates[coordinate_id].get_as_string() == coordinate.get_as_string():
                return True
        return False
    
    
    def get_len_coordinates(self):
        return len(self.coordinates)


    def add_callpath(self, callpath):
        self.callpaths.append(callpath)
        
        
    def get_len_callpaths(self):
        return len(self.callpaths)


    def get_callpath(self, callpath_id):
        return self.callpaths[callpath_id]
    
    
    def get_callpaths(self):
        return self.callpaths
    
    
    def get_callpath_id(self, callpath_name):
        for callpath_id in range(len(self.callpaths)):
            if self.callpaths[callpath_id].get_name() == callpath_name:
                return callpath_id
        return -1
    
    
    def callpath_exists(self, callpath_name):
        for callpath_id in range(len(self.callpaths)):
            if self.callpaths[callpath_id].get_name() == callpath_name:
                return True
        return False


    def get_measurements(self):
        return self.measurements
    
    
    def get_measurement(self, coordinate_id, callpath_id, metric_id):
        for i in range(len(self.measurements)):
            measurement = self.measurements[i]
            if measurement.get_callpath_id() == callpath_id and measurement.get_metric_id() == metric_id and measurement.get_coordinate_id() == coordinate_id:
                return measurement
        return None


    def add_measurement(self, measurement):
        self.measurements.append(measurement)
       
        
    def clear_measurements(self):
        self.measurements = []


    def debug(self):
        for i in range(len(self.metrics)):
            print("Metric "+str(i+1)+": "+self.metrics[i].get_name())
        for i in range(len(self.parameters)):
            print("Parameter "+str(i+1)+": "+self.parameters[i].get_name())
        for i in range(len(self.callpaths)):
            print("Callpath "+str(i+1)+": "+self.callpaths[i].get_name())
        for i in range(len(self.coordinates)):
            dimensions = self.coordinates[i].get_dimensions()
            coordinate_string = "Coordinate "+str(i+1)+": ("
            for dimension in range(dimensions):
                parameter, value = self.coordinates[i].get_parameter_value(dimension)
                parameter_name = parameter.get_name()
                coordinate_string += parameter_name+"="+str(value)+","
            coordinate_string = coordinate_string[:-1]
            coordinate_string += ")"
            print(coordinate_string)
        for i in range(len(self.measurements)):
            measurement = self.measurements[i]
            callpath_id = measurement.get_callpath_id()
            callpath = self.callpaths[callpath_id]
            callpath_name = callpath.get_name()
            metric_id = measurement.get_metric_id()
            metric = self.metrics[metric_id]
            metric_name = metric.get_name()
            coordinate_id = measurement.get_coordinate_id()
            coordinate = self.coordinates[coordinate_id]
            coordinate_string = coordinate.get_as_string()
            value_mean = measurement.get_value_mean()
            value_median = measurement.get_value_median()
            print("Measurement "+str(i+1)+": "+metric_name+", "+callpath_name+", "+coordinate_string+": "+str(value_mean)+" (mean), "+str(value_median)+" (median)")
            
        