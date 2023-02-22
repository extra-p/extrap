import typing
from collections import defaultdict

from PySide2.QtWidgets import QMessageBox

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.entities.metric import Metric
from extrap.modelers.aggregation.sum_aggregation import SumAggregation

if typing.TYPE_CHECKING:
    from extrap.gui.MainWidget import MainWidget


def calculate_complexity_comparison(model):
    if not model:
        return
    if isinstance(model, ComparisonModel):
        model.add_complexity_comparison_annotation()


def show_info(model, callpath):
    if not model and not callpath:
        return
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    if callpath:
        msg.setText(
            f"Tags for callpath {callpath}:")
        allComments = '\n'.join(f"{tag}: {value}" for tag, value in callpath.tags.items())
        msg.setInformativeText(allComments)
    msg.setWindowTitle("Model Info")
    # msg.setDetailedText("The details are as follows:")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def filter_1_percent_time(tree_view, on, tree_model):
    tree_view._filter_1_percent_time_state = on
    filter_id_percent_time = 'develop__filter_1_percent_time'
    if on:
        model_set = tree_view._selector_widget.getCurrentModel()
        use_median = model_set.modeler.use_median
        t_metric = Metric('time')
        total_time = defaultdict(int)
        for (callpath,
             metric), measurements in tree_view._selector_widget.main_widget.getExperiment().measurements.items():
            if metric != t_metric:
                continue
            if callpath.lookup_tag(SumAggregation.TAG_CATEGORY) is None and \
                    not callpath.lookup_tag(SumAggregation.TAG_USAGE_DISABLED, False):
                for measurement in measurements:
                    total_time[measurement.coordinate] += measurement.value(use_median)

        def filter_(node):
            if model_set is None or node.path is None:
                return True

            model = model_set.models.get((node.path, t_metric))
            if model:
                ratios = [measurement.value(use_median) / total_time[measurement.coordinate] for measurement in
                          model.measurements]
                node.path.tags['devel__filter__ratio'] = ratios
                return any(r >= 0.01 for r in ratios)
            else:
                return True

        tree_model.item_filter.put_condition(filter_id_percent_time, filter_)
    else:
        tree_model.item_filter.remove_condition(filter_id_percent_time)


def delete_subtree(tree_view, model):
    if not tree_view.selectedIndexes():
        return
    selectedCallpaths = [model.getValue(i) for i in tree_view.selectedIndexes()]

    for selectedCallpath in selectedCallpaths:
        if not selectedCallpath.path:
            continue
        callpath = selectedCallpath.path
        main_widget: MainWidget = tree_view._selector_widget.main_widget
        experiment = main_widget.getExperiment()
        callpaths_to_delete = [(i, c) for i, c in enumerate(experiment.callpaths) if
                               c.name.startswith(callpath.name)]

        for callpath_index, callpath_to_delete in reversed(callpaths_to_delete):
            del experiment.callpaths[callpath_index]  # make sure to delete only once
            for metric in experiment.metrics:
                key = (callpath_to_delete, metric)
                experiment.measurements.pop(key, None)
                for modeler in experiment.modelers:
                    modeler.models.pop(key, None)
    tree_view.model().valuesChanged()
