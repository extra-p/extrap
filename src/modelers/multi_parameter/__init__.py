from modelers.loader import load_modelers
from modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler as default

all_modelers = load_modelers(__path__, __name__)
all_modelers['default'] = default
