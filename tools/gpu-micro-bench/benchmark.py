import argparse
import csv
import json
import os

import numpy as np
import scipy.optimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--sm-count", action="store", type=int)
    arguments = parser.parse_args()

    if not arguments.skip_build:
        print("Building benchmark...")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        os.system("nvcc -O3 ../src/benchmark.cu -arch=native -o benchmark.x")
        os.chdir("..")

    metrics = [
        "gpu__time_duration.sum",
        "sm__sass_inst_executed_op_st.sum",
        "sm__sass_inst_executed_op_ld.sum",
        "sm__sass_inst_executed.sum",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
        "sm__sass_thread_inst_executed.sum",
        "sm__sass_thread_inst_executed_op_memory_pred_on.sum",
        "sm__sass_inst_executed_op_branch.sum",
        "sm__sass_thread_inst_executed_op_control_pred_on.sum",
        "sm__sass_thread_inst_executed_op_conversion_pred_on.sum",
        "sm__sass_thread_inst_executed_pred_on.sum",
        "sm__sass_thread_inst_executed_op_misc_pred_on.sum",
        "sm__sass_thread_inst_executed_op_bit_pred_on.sum",
        "sm__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum",
        "sm__sass_thread_inst_executed_op_uniform_pred_on.sum",
        "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum",
        "sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum",
    ]

    if arguments.sm_count:
        sm_count = arguments.sm_count
    else:
        sm_count_info = os.popen("./build/benchmark.x").read().strip().split()
        sm_count = int(sm_count_info[0])
    step_size = sm_count
    number_halfes = 0
    while step_size > 2 and number_halfes < 4:
        step_size /= 2
        number_halfes += 1
    runs = []
    current_value = 0
    while current_value < sm_count * 2:
        current_value += step_size
        runs.append(round(current_value))

    step_size = sm_count // 2
    runs += list(range(sm_count * 3, sm_count * 10 + step_size, step_size))
    if 2 not in runs:
        runs.insert(0, 2)

    # runs=list(reversed(range(max_block_size-step_size,0,-step_size)))+list(range(max_block_size,max_block_size*2,step_size))
    # runs=list(range(max_block_size*4+step_size,max_block_size*6,step_size))

    print("Runs:", runs)
    names = ["load", "store", "double_precision", "single_precision", "branch", "other"]

    per_block_values = []

    for gs in runs:
        print(f"==== Benchmark {gs} ====")
        if not arguments.skip_run:
            os.system(
                f"ncu --csv --log-file result{gs}.csv --metrics {','.join(metrics)} ./build/benchmark.x {gs}"
            )
        values, residuals = read_data(f"result{gs}.csv", gs, sm_count)
        per_block_values.append(values)
        np.savetxt("result.csv", per_block_values, delimiter=",")

        data = {name: {} for name in names}
        data['cu_count'] = sm_count
        for func_name, func in [('min', np.min), ('mean', np.mean), ('max', np.max), ('std', np.std)]:
            vals = func(per_block_values, axis=0)
            for name, val in zip(names, vals):
                data[name][func_name] = val
        print(json.dumps(data))


def iterative_bisection(step_size, runs):
    new_step_size = step_size / 2


def read_data(filename, gs, sm_count):
    data = {}
    with open(filename) as results:
        for _ in range(2):
            results.readline()
        csv_file = csv.DictReader(results)
        for line in csv_file:

            if int(line["ID"]) not in data:
                data[int(line["ID"])] = {"name": line["Kernel Name"]}
            try:
                value = int(
                    line["Metric Value"].replace(",", "")
                )
                data[int(line["ID"])][line["Metric Name"]] = value
            except ValueError:
                data[int(line["ID"])][line["Metric Name"]] = 0

    b = np.zeros(len(data))
    a = np.zeros((len(data), 6))

    for k, v in data.items():
        b[k] = v["gpu__time_duration.sum"] * gs / np.ceil(gs / sm_count)
        total = v["sm__sass_thread_inst_executed_pred_on.sum"]
        check_total = sum(v[k] for k in [
            "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_uniform_pred_on.sum",
            "sm__sass_thread_inst_executed_op_misc_pred_on.sum",
            "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
            "sm__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum",
            "sm__sass_thread_inst_executed_op_memory_pred_on.sum",
            "sm__sass_thread_inst_executed_op_conversion_pred_on.sum",
            "sm__sass_thread_inst_executed_op_control_pred_on.sum",
            "sm__sass_thread_inst_executed_op_bit_pred_on.sum"
        ])

        if total - check_total > 0.01 * total:
            print("WARNING: total instruction mismatch", v["name"], total, check_total)

        memory_store = (v["sm__sass_inst_executed_op_st.sum"] * v["sm__sass_thread_inst_executed_op_memory_pred_on.sum"]
                        / (v["sm__sass_inst_executed_op_st.sum"] + v["sm__sass_inst_executed_op_ld.sum"]))
        memory_load = (v["sm__sass_inst_executed_op_ld.sum"] * v["sm__sass_thread_inst_executed_op_memory_pred_on.sum"]
                       / (v["sm__sass_inst_executed_op_st.sum"] + v["sm__sass_inst_executed_op_ld.sum"]))
        a[k, 0] = memory_load
        a[k, 1] = memory_store
        a[k, 2] = (v["sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum"]
                   + v["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"])
        a[k, 3] = (v["sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum"]
                   + v["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"])
        a[k, 4] = v["sm__sass_thread_inst_executed_op_control_pred_on.sum"]
        a[k, 5] = (v["sm__sass_thread_inst_executed_op_bit_pred_on.sum"]
                   + v["sm__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum"]
                   + v["sm__sass_thread_inst_executed_op_integer_pred_on.sum"]
                   + v["sm__sass_thread_inst_executed_op_uniform_pred_on.sum"]
                   + v["sm__sass_thread_inst_executed_op_conversion_pred_on.sum"])

    return scipy.optimize.nnls(a, b)


if __name__ == "__main__":
    main()
