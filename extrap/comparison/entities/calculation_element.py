from __future__ import annotations

try:
    from typing import Protocol
except ImportError:
    class Protocol:
        pass


class CalculationElement(Protocol):
    def __add__(self, other) -> CalculationElement:
        pass

    def __sub__(self, other) -> CalculationElement:
        pass

    def __mul__(self, other) -> CalculationElement:
        pass

    def __truediv__(self, other) -> CalculationElement:
        pass

    def __radd__(self, other) -> CalculationElement:
        pass

    def __rmul__(self, other) -> CalculationElement:
        pass

    def __rsub__(self, other) -> CalculationElement:
        pass

    def __rtruediv__(self, other) -> CalculationElement:
        pass

    def __neg__(self) -> CalculationElement:
        pass
