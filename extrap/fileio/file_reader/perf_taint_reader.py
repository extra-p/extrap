#  This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
#  Copyright (c) 2021, Technical University of Darmstadt, Germany
#
#  This software may be modified and distributed under the terms of a BSD-style license.
#  See the LICENSE file in the base directory for details.
import json
import sys
import warnings
from json import JSONDecodeError
from pathlib import Path
from typing import Union, Dict, List, Optional

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node
from extrap.entities.experiment import Experiment
from extrap.fileio.file_reader.cube_file_reader2 import CubeFileReader2
from extrap.util.deprecation import deprecated
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import ProgressBar, DUMMY_PROGRESS

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    TypedDict = Dict


class PerfTaintLoop(TypedDict):
    callstack: List[Optional[List[int]]]
    data: List
    deps: List[List[str]]
    not_found_params: Optional[List[str]]


class PerfTaintFunction(TypedDict):
    file: str
    func_idx: int
    line: int
    loops: List[PerfTaintLoop]


class PerfTaintData(TypedDict):
    functions: Dict[str, PerfTaintFunction]
    functions_demangled_names: List[str]
    functions_mangled_names: List[str]
    functions_names: List[str]
    parameters: List[str]


class PerfTaintReader(CubeFileReader2):
    NAME = "perf-taint"
    GUI_ACTION = "Open per&f-taint file"
    DESCRIPTION = "Load a perf-taint file and a set of CUBE files to generate a new experiment"
    CMD_ARGUMENT = "--perf-taint"
    FILTER = "perf-taint file (*.json *.ll.json);;All Files (*)"
    LOADS_FROM_DIRECTORY = False

    use_inclusive_measurements = True

    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        path = Path(path)
        if path.is_dir():
            raise FileFormatError(f'{self.NAME.capitalize()} file path must point to a file: {path}')
        with open(path, "r") as inputfile:
            try:
                perf_taint_data: PerfTaintData = json.load(inputfile)
            except JSONDecodeError as error:
                raise FileFormatError("Perf-taint file error: " + str(error)) from error

        experiment = super().read_experiment(path.parent, progress_bar)
        parameter_map = {p.name: i for i, p in enumerate(experiment.parameters)}

        for func_name, func in perf_taint_data['functions'].items():
            for loop in func['loops']:
                for callstack in loop['callstack']:
                    node = experiment.call_tree
                    mangled_name = perf_taint_data['functions_mangled_names'][func['func_idx']]
                    if not callstack:
                        node = self._find_mangled_name(node, mangled_name, 10)
                        if not node:
                            warnings.warn(
                                f"Function could not be found: {perf_taint_data['functions_names'][func['func_idx']]}")
                            continue
                    else:
                        call_iter = iter(callstack)
                        c = next(call_iter)
                        node = self._find_mangled_name(node,
                                                       perf_taint_data['functions_mangled_names'][c], 10)
                        for c in call_iter:
                            new_node = self._find_mangled_name(node, perf_taint_data['functions_mangled_names'][c], 1)
                            if new_node:
                                node = new_node
                        print("->".join((perf_taint_data['functions_mangled_names'][c] for c in callstack)))
                        node = self._find_mangled_name(node, mangled_name, 1)
                        if not node:
                            warnings.warn(f"Function could not be found: {perf_taint_data['functions_names'][c]}")
                            continue

                    # not_found_params = node.path.tags.get('perf_taint__not_found_params', [])
                    # if loop['not_found_params']:
                    #     for p in loop['not_found_params']:
                    #         if p in parameter_map:
                    #             not_found_params.append(parameter_map[p])
                    # node.path.tags['perf_taint__not_found_params'] = not_found_params

                    depends_on_params = node.path.tags.get('perf_taint__depends_on_params', [])
                    for p_list in loop['deps']:
                        for p in p_list:
                            if p in parameter_map:
                                depends_on_params.append(parameter_map[p])
                    node.path.tags['perf_taint__depends_on_params'] = depends_on_params
        self._set_dependend_params_on_rest_of_calltree(experiment.call_tree)
        progress_bar.update()
        return experiment

    def _find_mangled_name(self, node: Node, mangled_name: str, depth: int = 1):
        if not node:
            return None
        if node.mangled_name and mangled_name == node.mangled_name:
            return node
        elif depth > 0:
            depth -= 1
            for child in node:
                n = self._find_mangled_name(child, mangled_name, depth)
                if n:
                    return n
        return None

    def _set_dependend_params_on_rest_of_calltree(self, node: Node):
        if node.path and 'perf_taint__depends_on_params' not in node.path.tags:
            node.path.tags['perf_taint__depends_on_params'] = []
        for child in node:
            self._set_dependend_params_on_rest_of_calltree(child)

    @deprecated
    def read_cube_file(self, dir_name, scaling_type, pbar=DUMMY_PROGRESS, selected_metrics=None):
        raise NotImplementedError()

    @staticmethod
    def _make_callpath_mapping(cnodes):
        callpaths = {}

        def walk_tree(parent_cnode, parent_name, parent_mangled_name):
            for cnode in parent_cnode.get_children():
                name = cnode.region.name
                mangled_name = cnode.region.mangled_name
                path_name = '->'.join((parent_name, name))
                callpaths[cnode.id] = Callpath(path_name)
                path_mangled_name = '->'.join((parent_mangled_name, mangled_name))
                callpaths[cnode.id].mangled_name = path_mangled_name
                walk_tree(cnode, path_name, path_mangled_name)

        for root_cnode in cnodes:
            name = root_cnode.region.name
            mangled_name = root_cnode.region.mangled_name
            callpath = Callpath(name)
            callpath.mangled_name = mangled_name
            callpaths[root_cnode.id] = callpath
            walk_tree(root_cnode, name, mangled_name)

        return callpaths
