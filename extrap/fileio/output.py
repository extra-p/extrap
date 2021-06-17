from extrap.entities.experiment import Experiment
import extrap.fileio.io_helper as io
import re


def fmt_output(experiment: Experiment, printtype: str):
    print_str = printtype.lower()
    models = experiment.modelers[0].models
    options = re.findall(r"\{(.*?)\}", print_str)
    text = ""

    for m in models.values():
        temp = print_str

        for o in options:
            if o == "callpath":
                temp = temp.replace("{callpath}", m.callpath.name)
            elif o == "metric":
                temp = temp.replace("{metric}", m.metric.name)
            elif o == "model":
                temp = temp.replace("{model}",
                                    m.hypothesis.function.to_string(*experiment.parameters))
            elif o == "coordinates":
                pass  # not sure yet
            elif o == "smape":
                temp = temp.replace("{smape}", "{:.2E}".format(m.hypothesis.SMAPE))
            elif o == "rrss":
                temp = temp.replace("{rrss}", "{:.2E}".format(m.hypothesis.rRSS))
            elif o == "rss":
                temp = temp.replace("{rss}", "{:.2E}".format(m.hypothesis.RSS))
            elif o == "ar2":
                temp = temp.replace("{ar2}", "{:.2E}".format(m.hypothesis.AR2))
            elif o == "re":
                temp = temp.replace("{re}", "{:.2E}".format(m.hypothesis.RE))
            else:
                return ValueError("invalid input")

        text += (temp + "\n")

    return text
