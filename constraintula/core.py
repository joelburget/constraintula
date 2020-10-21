# Copyright 2020 The constraintula Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides tools for analyzing a system of symbols and constraints
on those symbols. For example, a circle can be described by four different
symbols:
    radius
    diameter
    perimeter
    area
but there's only one degree of freedom because the symbols are constrained by
a set of constraint equations:
    diameter - 2 radius = 0
    perimeter - 2 pi radius = 0
    area - pi radius^2 = 0 .
This module provides classes that help handle this kind of situation. For
example, one class allows you to declare the symbols and their constraints, and
then choose a subset of the symbols to be considered independent. The class then
provides functions that map the independents to the dependents.
"""
import functools
import inspect
from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    overload,
    Sequence,
    Set,
    Type,
    Union,
)

import sympy
from sympy import Expr, Symbol, symbols


def collect_symbols(expr: sympy.Expr) -> Set[sympy.Symbol]:
    """Collect all symbols in this expression.

    Eg `x + y` -> `{Symbol('x'), Symbol('y')}.
    """
    symbols = set()

    def walk(expr):
        if isinstance(expr, Symbol):
            symbols.add(expr)
        else:
            for arg in expr.args:
                walk(arg)

    walk(expr)
    return symbols


def make_wrapper(
    func: Callable, constraints: Sequence[Expr], skip_first_arg: bool = False
) -> Callable:
    """Wrap a function to allow calling with any complete set of parameters.

    Args:
        func: Callable that takes keyword arguments.
        constraints: Relationships that must hold between args to func.
        skip_first_arg: If True, skip the first argument of func. This is useful
            when making a wrapper for class methods that have a special first
            argument like `self` or `cls`

    Returns:
        A new callable that takes any complete set of parameters for the given
        constraints. We solve for the values of the unspecified parameters and
        then call the given func.
    """
    parameters = inspect.signature(func).parameters
    arg_types = list(
        (k, int if ty.annotation is int else float) for k, ty in parameters.items()
    )
    if skip_first_arg:
        arg_types = arg_types[skip_first_arg:]

    # Make a symbol for each arg. We can't use integer=(ty is int) because eg
    # integer=False excludes real number solutions that happen to be integers.
    arg_symbols = {
        k: Symbol(k, integer=True) if ty is int else Symbol(k) for k, ty in arg_types
    }

    @functools.wraps(func)
    def wrapper(*args, **kw):
        for k, val in kw.items():
            if k not in arg_symbols:
                arg_symbols[k] = Symbol(k)

        # Collect all the symbols appearing in all constraints
        constraint_symbols = [collect_symbols(constraint) for constraint in constraints]
        constraint_symbols = set().union(*constraint_symbols)

        # Rewrite all constraints so that we use the right symbol, eg
        # `Symbol('x', integer=True)` rather than `Symbol('x')`.
        for i, constraint in enumerate(constraints):
            for sym in constraint_symbols:
                if sym.name in arg_symbols:
                    constraints[i] = constraints[i].subs(sym, arg_symbols[sym.name])

        # Extend the set of explicit constraints with a constraint for each arg
        # value.
        extended_constraints = constraints + [arg_symbols[k] - v for k, v in kw.items()]

        values = sympy.solve(extended_constraints)

        # sympy sometimes returns a list of solutions, sometimes just a single
        # dict.
        if isinstance(values, list):
            if not len(values):
                raise ValueError("System has no solution")
            values = values[0]

        # Use `ty` to convert each solved value from the sympy type to either
        # int or float. `values` is indexed by symbol rather than string.
        kwargs = {k: ty(values[arg_symbols[k]]) for k, ty in arg_types}
        return func(*args, **kwargs)

    return wrapper


def constrain(constraints: Sequence[Expr]) -> Callable[[Type], Type]:
    """Make a function or class callable by any complete set of parameters.

    This decorates a callable object, either a function or class, to make it
    possible to call the object with any complete set of parameters. The
    given constraints will then be solved to determine the values of the other
    parameters and all values will be passed to the underlying function or
    class constructor.

    For classes, this can be used with the attr library for optimally DRY code:

        radius, area = constraintula.symbols('radius area')

        @constrain([area - pi * radius**2])
        @attr.dataclass(frozen=True)
        class Circle:
            radius: float
            area: float

        circle_by_area = Circle(area=4)
        circle_by_radius = Circle(radius=1)

    Using constrain in combination with the attr library makes mypy happy. The
    two variants shown below require explicit signals to tell the typechecker
    that we know what we're doing.

    Here's constrain with a vanilla class:

        radius, area = constraintula.symbols('radius area')

        @constrain([area - pi * radius**2])
        class Circle:
            def __init__(self, radius, area):
                self.radius = radius
                self.area = area

        circle = Circle(area=1)  # pylint: disable=no-value-for-parameter

    The decorator knows to only pass keywords that the class expects, so it
    can be used with classes that expect only a single independent subset of
    its interrelated attributes to be specified. Note, however, that in this
    case the relationship between the variables is written twice: once in
    the call to constrain and again in the @property. In addition, pylint does
    not know about the constraint decorator, so it will warn about calls that
    use alternate parameters and the warning must be explicitly disabled.

        radius, area = constraintula.symbols('radius area')

        @constrain([area - pi * radius**2])
        class Circle:
            def __init__(self, radius):
                self.radius = radius
            @property
            def area(self):
                return pi * self.radius**2

        circle = Circle(area=1)  # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
    """
    return _Constrainer(constraints)


class _Constrainer:
    """Implements the `constrain` decorator; see docstring above for details."""

    def __init__(self, constraints: Sequence[Expr]):
        self.constraints = constraints

    # pylint: disable=function-redefined
    @overload
    def __call__(self, obj: Type) -> Type:
        pass

    @overload
    def __call__(self, obj: Callable) -> Callable:
        pass

    def __call__(self, obj: Union[Type, Callable]) -> Union[Type, Callable]:
        if isinstance(obj, type):
            if obj.__new__ is not object.__new__:
                method = '__new__'
            else:
                method = '__init__'
            wrapped = make_wrapper(getattr(obj, method), self.constraints, skip_first_arg=True)
            setattr(obj, method, wrapped)
            return obj

        return make_wrapper(obj, self.constraints, skip_first_arg=False)

    # pylint: enable=function-redefined
