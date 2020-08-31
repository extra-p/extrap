import itertools

from extrap.util.deprecation import deprecated
from extrap.util.serialization_schema import make_value_schema


class Parameter:
    """
    Counter for global parameter ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, name):
        self.name = name
        self.id = next(Parameter.ID_COUNTER)

    @deprecated("Use property directly.")
    def set_name(self, name):
        self.name = name

    @deprecated("Use property directly.")
    def get_name(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return NotImplemented
        return self is other or self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Parameter({self.name})"


ParameterSchema = make_value_schema(Parameter, 'name')
