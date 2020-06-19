class Parameter:

    def __init__(self, name):
        self.name = name

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return self is other or self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Parameter({self.name})"
