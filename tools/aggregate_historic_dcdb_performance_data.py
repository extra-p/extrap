# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import dataclasses
import os
from collections import defaultdict
from datetime import datetime, timedelta

NODE_LIST_KEY = 'Node List:'


@dataclasses.dataclass
class AggSensorData:
    start: datetime
    end: datetime
    sum_value: float
    count: int = 0


class AggMetricData:
    time: timedelta = timedelta()
    sum_value: float = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobid", nargs='+')
    parser.add_argument('-m', '--metric', nargs='*', default=['energy', 'gpu0/sysfs-energy'])
    parser.add_argument('--not-time-relative', action='store_true')
    parser.add_argument('--no-total', action='store_true')
    args = parser.parse_args()

    metric_data = defaultdict(AggMetricData)
    for jobid in args.jobid:
        nodes = get_nodes_from_job(jobid)
        for metric in args.metric:
            value, time = get_agg_sensor_data(jobid, metric, nodes)
            metric_data[metric].sum_value += value
            metric_data[metric].time += time

    total_value = 0
    for metric, data in metric_data.items():
        if not data.time:
            continue

        if args.not_time_relative:
            total_value += data.sum_value
            print(metric, data.sum_value)
        else:
            total_value += data.sum_value / data.time.total_seconds()
            print(metric, data.sum_value / data.time.total_seconds())

    if not args.no_total:
        print("-" * 40)
        print("TOTAL", total_value)


def get_agg_sensor_data(jobid, metric, nodes):
    query_string = " ".join(f"{node}/{metric}" for node in nodes)
    sensor_data = {}
    with os.popen(f'dcdbquery -j {jobid} {query_string}') as stream:
        for line in stream:
            if line[0] != '/':
                continue
            data = line.split(',')
            sensor, time = data[0:2]
            time = datetime.fromisoformat(time[:-3])
            values = [float(d) for d in data[2:]]
            value = sum(values) / len(values)
            if sensor not in sensor_data:
                sensor_data[sensor] = AggSensorData(time, time, value)
            else:
                sensor_data[sensor].start = min(sensor_data[sensor].start, time)
                sensor_data[sensor].end = max(sensor_data[sensor].end, time)
                sensor_data[sensor].sum_value += value
                sensor_data[sensor].count += 1
    agg_value = 0
    agg_time = timedelta()
    for sensor, data in sensor_data.items():
        agg_time += (data.end - data.start) / data.count * (data.count + 1)
        agg_value += data.sum_value

    return agg_value, agg_time


def get_nodes_from_job(jobid):
    with os.popen(f'dcdbconfig job show {jobid}') as stream:
        for line in stream:
            if line.startswith(NODE_LIST_KEY):
                line = line[len(NODE_LIST_KEY):].strip()
                nodes = [n.strip() for n in line.split(',')]
                return nodes
        raise RuntimeError(f"Could not find nodes for job {jobid}.")


if __name__ == '__main__':
    main()
