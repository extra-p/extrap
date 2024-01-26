



class MeasurementPointAdvisor():


    def __init__(self, budget, processes, callpaths, metric, experiment) -> None:
        self.budget = budget
        print("budget:",budget)

        self.processes = processes
        print("processes:",processes)

        self.experiment = experiment

        self.parameters = []
        for i in range(len(self.experiment.parameters)):
            self.parameters.append(str(self.experiment.parameters[i]))
        print("parameters:",self.parameters)

        self.metric_str = metric
        self.metric = None
        self.metric_id = -1
        for i in range(len(self.experiment.metrics)):
            if str(self.experiment.metrics[i]) == self.metric_str:
                self.metric = self.experiment.metrics[i]
                self.metric_id = i
                break

        self.callpaths = callpaths

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
