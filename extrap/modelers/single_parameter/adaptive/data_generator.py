# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from random import uniform

import numpy as np

from extrap.entities.functions import SingleParameterFunction


class TrainingDataGenerator:
    def __init__(self, terms, samplePoints, bucket_indices, noise=0.2):
        self.functions = [SingleParameterFunction(t) for t in terms]
        self.noise = noise

        self._bucket_indices = bucket_indices
        self._sample_points = np.array(samplePoints)

        self.min_coefficient = 0.001
        self.max_coefficient = 1000

    @property
    def noise(self) -> float:
        return self._noise * 2

    @noise.setter
    def noise(self, value: float):
        self._noise = value / 2

    def create_data(self, count, progress_event=lambda _: _):

        num_labels = len(self.functions)
        data_values = np.ndarray((count * num_labels, len(self._bucket_indices)))
        data_labels = np.ndarray((count * num_labels, num_labels))

        for i in range(0, count):
            progress_event(i / count)
            for flabel, fun in enumerate(self.functions):
                idx = i * num_labels + flabel
                points = self._sample_points
                fun.constant_coefficient = idx  # self.get_random_coefficient()
                for ct in fun.compound_terms:
                    ct.coefficient = self.get_random_coefficient()
                values = fun.evaluate(points)
                values_distorted = self.add_distortions(values)

                data_values[idx] = values_distorted

                label = self.make_label(flabel, num_labels)
                data_labels[idx] = label

        progress_event(None)
        return data_values, data_labels

    def add_distortions(self, values):
        # random noise with bates distribution
        noise = np.mean(np.random.uniform(1 - self._noise, 1 + self._noise, (len(values), 5)), axis=1)

        # apply noise
        values_raw = values * noise
        values_preprocessed = values_raw / self._sample_points

        # sort values into buckets
        values_bucketed = np.zeros(len(self._bucket_indices))
        for bi, pi in enumerate(self._bucket_indices):
            if pi > 0:
                values_bucketed[bi] = values_preprocessed[pi - 1]
        return values_bucketed

    @staticmethod
    def make_label(flabel, total):
        return np.eye(1, M=total, k=flabel)

    def get_random_coefficient(self):
        return uniform(self.min_coefficient, self.max_coefficient)
