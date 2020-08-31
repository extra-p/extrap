from extrap.modelers.loader import load_modelers
from extrap.modelers.single_parameter.basic import SingleParameterModeler as Default

all_modelers = load_modelers(__path__, __name__)
all_modelers['Default'] = Default
