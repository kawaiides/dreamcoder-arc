from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tboolean
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import binary_dilation, label
from scipy import ndimage
from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
import numpy as np
from collections import deque
from statistics import mode

from typing import Tuple, NewType, List, Callable, Dict, Type, FrozenSet, Iterable, Any

tcolour = baseType("colour") # Any colour. We could use 1x1 grids for this, but by typing it we reduce the search space
Colour = NewType("Colour", int)



tpos = baseType("pos") # Position-only type
Position = NewType("Position", Tuple[int, int])

tsize = baseType("size")
Size = NewType("Size", Tuple[int, int])

tcount = baseType("count")
Count = NewType("Count", int)
    
tgrid = baseType("grid") # Any grid. Position is always included

# michael's types
Boolean = bool
Integer = int
IntegerTuple = NewType("IntegerTuple", Tuple[Integer, Integer])
Numerical = NewType("Numerical", Union[Integer, IntegerTuple])
IntegerSet = NewType("IntegerSet", FrozenSet[Integer])
Grid = NewType("Grid", Tuple[Tuple[Integer]])
Cell = NewType("Cell", Tuple[Integer, IntegerTuple])
Object = NewType("Object", FrozenSet[Cell])
Objects = NewType("Objects", FrozenSet[Object])
Indices = NewType("Indices", FrozenSet[IntegerTuple])
IndicesSet = NewType("IndicesSet", FrozenSet[Indices])
Patch = NewType("Patch", Union[Object, Indices])
Element = NewType("Element", Union[Object, Grid])
Piece = NewType("Piece", Union[Grid, Patch])
TupleTuple = NewType("TupleTuple", Tuple[Tuple])
ContainerContainer = NewType("ContainerContainer", Container[Container])

tIntegerTuple = baseType("IntegerTuple")
tNumerical = baseType("Numerical")
tIntegerSet = baseType("IntegerSet")
tGrid = baseType("Grid")
tCell = baseType("Cell")
tObject = baseType("Object")
tObjects = baseType("Objects")
tIndices = baseType("Indices")
tIndicesSet = baseType("IndicesSet")
tPatch = baseType("Patch")
tElement = baseType("Element")
tPiece = baseType("Piece")
tTupleTuple = baseType("TupleTuple")
tContainerContainer = baseType("ContainerContainer")


class Grid():
    """
    Represents an ARC grid, along with a position.
    For an unshifted grid (e.g. the input), the position is (0,0).
    Position holds the canonical index of the top-left grid pixel.
    This encompasses the Input, Grid and Object types from Alford.

    Instantiation with cutout=True will remove any background.
    """

    def __init__(self, grid: np.ndarray, position: Tuple[int, int]=(0, 0), cutout=False):
        self.position = position

        if grid.shape[0] > 30 or grid.shape[1] > 30:
            raise PrimitiveException(f"Grid size {grid.shape} too large")

        self.cutout = cutout
        if cutout:
            self.grid, (xpos, ypos) = Grid.cutout(grid)
            self.position = (self.position[0] + xpos, self.position[1] + ypos)
        else:
            self.grid = grid

    @property
    def size(self) -> Size:
        return self.grid.shape

    @classmethod
    def cutout(grid: np.ndarray) -> np.ndarray:
        xr, yr = np.nonzero(grid)
        xpos, ypos = min(xr), min(yr)
        return grid[xpos:max(xr)+1, ypos:max(yr)+1], (xpos, ypos)
    
    def newgrid(self, grid: np.ndarray, offset=None, cutout=False) -> "Grid":
        """
        Return a new Grid object containing a new userprovided grid.
        The position is cloned from the parent.
        """
        position = self.position
        if offset:
            position = (position[0] + offset[0], position[1] + offset[1])

        if grid.shape[0] > 30 or grid.shape[1] > 30:
            raise PrimitiveException(f"Grid size {grid.shape} too large")

        return Grid(grid, position, cutout)
    
    def count(self) -> int:
        """
        Returns the number of non-zero elements in the grid
        """
        return np.count_nonzero(self.grid)

    def __eq__(self, other) -> bool:
        """
        Score a grid. Returns True iff the two grids are equal, ignoring position
        """
        if isinstance(other, Grid):
            return self.size == other.size and (self.grid == other.grid).all()
        return False

    def __repr__(self):
        return f"Grid({self.grid.shape[0]}x{self.grid.shape[1]} at {self.position})"
    
typemap: Dict[Type, TypeConstructor] = {
    Colour: tcolour,
    Position: tpos,
    Size: tsize,
    Count: tcount,
    Grid: tgrid,
    IntegerTuple : tIntegerTuple,
    Numerical : tNumerical,
    IntegerSet : tIntegerSet,
    Grid : tGrid,
    Cell : tCell,
    Object : tObject,
    Objects : tObjects,
    Indices : tIndices,
    IndicesSet : tIndicesSet,
    Patch : tPatch,
    Element : tElement,
    Piece : tPiece,
    TupleTuple : tTupleTuple,
    ContainerContainer : tContainerContainer
}

def primitive_assert(boolean, message=None):
    """
    Raise a PrimitiveException if the condition is false.
    This stops execution on the current program and does not raise an error.
    """
    if not boolean:
        raise PrimitiveException(message)

import inspect, typing

class DSL:
    def __init__(self, typemap: Dict[Type, TypeConstructor], verbose=False):
        self.typemap = typemap
        self.primitives = {}
        self.verbose = verbose

    def cvt_type(self, anno):
        # Handle list types natively
        # These annotations have type typing._GenericAlias
        # __origin__ attr is list, __args__ attr is a tuple of constituent types
        if hasattr(anno, '__origin__'):
            if anno.__origin__ == list:
                # We recursively convert the constituent
                return tlist(self.cvt_type(anno.__args__[0]))
        
        if anno in self.typemap:
            return self.typemap[anno]
        
        raise TypeError(f"Annotation {anno} has no corresponding DreamCoder type")

    def register(self, f: Callable, name: str=None, typesig: List[TypeConstructor]=None, autocurry: bool=True):
        if not isinstance(f, typing.Callable):
            # This is a value, not a function
            if len(typesig) != 1:
                raise TypeError('Value passed to Primitive constructor, typesig must be of length 1')
            if name is None:
                raise ValueError('Value passed to Primitive constructor, name must be specified')
            dc_type = typesig[-1]

            primitive = Primitive(name, dc_type, f)
            primitive.typesig = typesig # Allow later reflection
            self.primitives[name] = primitive

            if self.verbose:
                print(f"Registered value {name} of type {dc_type}.")

            return

        if name is None:
            name = f.__name__
            if name == '<lambda>':
                raise ValueError('<lambda> passed to Primitive constructor, name must be specified')
            
        fn_sig = inspect.signature(f)
        params = list(fn_sig.parameters.items())
        param_count = len(params)
        if typesig is None:
            # Generate a DreamCoder type signature for this function by inspection
            arrow_args = []

            for arg, argtype in params:
                anno = argtype.annotation
                arrow_args.append(self.cvt_type(anno))

            typesig = arrow_args + [self.cvt_type(fn_sig.return_annotation)    ]

        dc_type = arrow(*typesig)

        # This function has more than 1 input and needs to be curried
        # We have special cases for 2/3 params because these are significantly faster
        if autocurry and param_count > 1:
            if param_count == 2:
                func = lambda x: lambda y: f(x, y)
            elif param_count == 3:
                func = lambda x: lambda y: lambda z: f(x, y, z)
            else:
                def curry(f, n, args):
                    if n:
                        return lambda x: curry(f, n-1, args + [x])
                    return f(*args)
                func = curry(f, param_count, [])
        else:
            func = f

        if self.verbose:
            print(f"Registered {name} with inferred type {dc_type}.")

        primitive = Primitive(name, dc_type, func)
        primitive.typesig = typesig # Allow later reflection
        self.primitives[name] = primitive
    
    def registerMany(self, funcs: List[Callable]):
        for func in funcs:
            try:
                self.register(func)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error occured on {f}")
                raise

    # Decorator function to define a primitive
    def primitive(self, func: Callable=None, name: str=None, typesig: List[TypeConstructor]=None, autocurry: bool=True):
        # First, we define a decorator factory
        def decorator(func):
            self.register(func, name, typesig, autocurry)
            return func
        
        # If we are called as a decorator factory, return the decorator
        if func is None:
            return decorator
        else:
            return decorator(func)

    @staticmethod
    def parse_primitive_names(ocaml_contents):
        contents = ''.join([c[:-1] for c in ocaml_contents if c[0:2] + c[-3:-1] != '(**)'])
        contents = contents.split('primitive "')[1:]
        primitives = [p[:p.index('"')] for p in contents if '"' in p]
        return primitives

    def generate_ocaml_primitives(self):
        primitives = list(self.primitives.values())

        with open("solvers/program.ml", "r") as f:
            contents = f.readlines()

        start_ix = min([i for i in range(len(contents)) if contents[i][0:7] == '(* AUTO'])
        end_ix = min([i for i in range(len(contents)) if contents[i][0:11] == '(* END AUTO'])

        non_auto_contents = contents[0:start_ix+1] + contents[end_ix:]
        # get the existing primitive names. We won't auto-create any primitives
        # whose name matches an existing name.
        existing_primitives = self.parse_primitive_names(non_auto_contents)

        lines = [p.ocaml_string() + '\n' for p in primitives
                if p.name not in existing_primitives]

        for p in primitives:
            if p.name in existing_primitives:
                print('Primitive {} already exists, skipping ocaml code generation for it'.format(p.name))

        contents = contents[0:start_ix+1] + lines + contents[end_ix:]

        with open("solvers/program.ml", "w") as f:
            f.write(''.join(contents))

dsl = DSL(typemap, verbose=False)

#############################
###### Shyam x Hodel kekw
#############################

 # from typing import (
#     List,
#     Union,
#     Tuple,
#     Any,
#     Container,
#     Callable,
#     FrozenSet,
#     Iterable
# )

# Boolean = bool
# Integer = int
# IntegerTuple = Tuple[Integer, Integer]
# Numerical = Union[Integer, IntegerTuple]
# IntegerSet = FrozenSet[Integer]
# Grid = Tuple[Tuple[Integer]]
# Cell = Tuple[Integer, IntegerTuple]
# Object = FrozenSet[Cell]
# Objects = FrozenSet[Object]
# Indices = FrozenSet[IntegerTuple]
# IndicesSet = FrozenSet[Indices]
# Patch = Union[Object, Indices]
# Element = Union[Object, Grid]
# Piece = Union[Grid, Patch]
# TupleTuple = Tuple[Tuple]
# ContainerContainer = Container[Container]



# # constants


# ZERO = 0
# ONE = 1
# TWO = 2
# THREE = 3
# FOUR = 4
# FIVE = 5
# SIX = 6
# SEVEN = 7
# EIGHT = 8
# NINE = 9
# TEN = 10
# F = False
# T = True

# NEG_ONE = -1

# ORIGIN = (0, 0)
# UNITY = (1, 1)
# DOWN = (1, 0)
# RIGHT = (0, 1)
# UP = (-1, 0)
# LEFT = (0, -1)

# NEG_TWO = -2
# NEG_UNITY = (-1, -1)
# UP_RIGHT = (-1, 1)
# DOWN_LEFT = (1, -1)

# ZERO_BY_TWO = (0, 2)
# TWO_BY_ZERO = (2, 0)
# TWO_BY_TWO = (2, 2)
# THREE_BY_THREE = (3, 3)



# # primitives


@dsl.primitive
def identity(
    x: Any
) -> Any:
    """ identity function """
    return x


@dsl.primitive
def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)


@dsl.primitive
def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


@dsl.primitive
def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)
    

@dsl.primitive
def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)


@dsl.primitive
def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])


@dsl.primitive
def even(
    n: Integer
) -> Boolean:
    """ evenness """
    return n % 2 == 0


@dsl.primitive
def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)


@dsl.primitive
def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)


@dsl.primitive
def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b


@dsl.primitive
def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b


@dsl.primitive
def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container


@dsl.primitive
def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))


@dsl.primitive
def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b


@dsl.primitive
def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)


@dsl.primitive
def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)


@dsl.primitive
def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))


@dsl.primitive
def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))


@dsl.primitive
def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b


@dsl.primitive
def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)


@dsl.primitive
def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)


@dsl.primitive
def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)


@dsl.primitive
def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)


@dsl.primitive
def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))


@dsl.primitive
def valmin(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ minimum by custom function """
    return compfunc(min(container, key=compfunc, default=0))


@dsl.primitive
def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)


@dsl.primitive
def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)


@dsl.primitive
def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)


@dsl.primitive
def leastcommon(
    container: Container
) -> Any:
    """ least common item """
    return min(set(container), key=container.count)


@dsl.primitive
def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})


@dsl.primitive
def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b


@dsl.primitive
def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b


@dsl.primitive
def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)


@dsl.primitive
def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)


@dsl.primitive
def crement(
    x: Numerical
) -> Numerical:
    """ incrementing positive and decrementing negative """
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1)
    )


@dsl.primitive
def sign(
    x: Numerical
) -> Numerical:
    """ sign """
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )


@dsl.primitive
def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0


@dsl.primitive
def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)


@dsl.primitive
def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)


@dsl.primitive
def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))


@dsl.primitive
def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))


@dsl.primitive
def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))


@dsl.primitive
def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)


@dsl.primitive
def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))


@dsl.primitive
def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]


@dsl.primitive
def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))


@dsl.primitive
def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)


@dsl.primitive
def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))


@dsl.primitive
def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))


@dsl.primitive
def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)


@dsl.primitive
def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)


@dsl.primitive
def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))


@dsl.primitive
def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value


@dsl.primitive
def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))


@dsl.primitive
def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))


@dsl.primitive
def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target


@dsl.primitive
def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)


@dsl.primitive
def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)


@dsl.primitive
def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))


@dsl.primitive
def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))


@dsl.primitive
def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)


@dsl.primitive
def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)


@dsl.primitive
def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))


@dsl.primitive
def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))


@dsl.primitive
def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))


@dsl.primitive
def prapply(
    function: Callable,
    a: Container,
    b: Container
) -> FrozenSet:
    """ apply function on cartesian product """
    return frozenset(function(i, j) for j in b for i in a)


@dsl.primitive
def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)
    

@dsl.primitive
def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)


@dsl.primitive
def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1


@dsl.primitive
def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1


@dsl.primitive
def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))


@dsl.primitive
def portrait(
    piece: Piece
) -> Boolean:
    """ whether height is greater than width """
    return height(piece) > width(piece)


@dsl.primitive
def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)


@dsl.primitive
def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)


@dsl.primitive
def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)


@dsl.primitive
def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))


@dsl.primitive
def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)


@dsl.primitive
def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))


@dsl.primitive
def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


@dsl.primitive
def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


@dsl.primitive
def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))


@dsl.primitive
def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])


@dsl.primitive
def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch


@dsl.primitive
def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))


@dsl.primitive
def shift(
    patch: Patch,
    directions: IntegerTuple
) -> Patch:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)


@dsl.primitive
def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))


@dsl.primitive
def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})


@dsl.primitive
def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})


@dsl.primitive
def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)


@dsl.primitive
def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)


@dsl.primitive
def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )


@dsl.primitive
def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )


@dsl.primitive
def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))


@dsl.primitive
def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))


@dsl.primitive
def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))


@dsl.primitive
def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))


@dsl.primitive
def square(
    piece: Piece
) -> Boolean:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)


@dsl.primitive
def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1


@dsl.primitive
def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1


@dsl.primitive
def hmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0


@dsl.primitive
def vmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a column for which both patches have cells """
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0


@dsl.primitive
def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))


@dsl.primitive
def adjacent(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1


@dsl.primitive
def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1


@dsl.primitive
def centerofmass(
    patch: Patch
) -> IntegerTuple:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))


@dsl.primitive
def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})


@dsl.primitive
def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))


@dsl.primitive
def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]


@dsl.primitive
def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)


@dsl.primitive
def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))


@dsl.primitive
def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))


@dsl.primitive
def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])


@dsl.primitive
def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]


@dsl.primitive
def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)


@dsl.primitive
def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


@dsl.primitive
def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


@dsl.primitive
def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))


@dsl.primitive
def fill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)


@dsl.primitive
def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)


@dsl.primitive
def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)


@dsl.primitive
def underpaint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid where there is background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if grid_painted[i][j] == bg:
                grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)


@dsl.primitive
def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return upscaled_grid


@dsl.primitive
def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_grid = upscaled_grid + tuple(row for num in range(factor))
    return upscaled_grid


@dsl.primitive
def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return upscaled_grid
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        upscaled_obj = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    upscaled_obj.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(upscaled_obj), (di_inv, dj_inv))


@dsl.primitive
def downscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    downscaled_grid = tuple()
    for i in range(h):
        downscaled_row = tuple()
        for j in range(w):
            if j % factor == 0:
                downscaled_row = downscaled_row + (grid[i][j],)
        downscaled_grid = downscaled_grid + (downscaled_row, )
    h = len(downscaled_grid)
    downscaled_grid2 = tuple()
    for i in range(h):
        if i % factor == 0:
            downscaled_grid2 = downscaled_grid2 + (downscaled_grid[i],)
    return downscaled_grid2


@dsl.primitive
def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))


@dsl.primitive
def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b


@dsl.primitive
def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))


@dsl.primitive
def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))


@dsl.primitive
def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))


@dsl.primitive
def cellwise(
    a: Grid,
    b: Grid,
    fallback: Integer
) -> Grid:
    """ cellwise match of two grids """
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid


@dsl.primitive
def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)


@dsl.primitive
def switch(
    grid: Grid,
    a: Integer,
    b: Integer
) -> Grid:
    """ color switching """
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)


@dsl.primitive
def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)


@dsl.primitive
def position(
    a: Patch,
    b: Patch
) -> IntegerTuple:
    """ relative position between two patches """
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)


@dsl.primitive
def index(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]] 


@dsl.primitive
def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))


@dsl.primitive
def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})


@dsl.primitive
def connect(
    a: IntegerTuple,
    b: IntegerTuple
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()


@dsl.primitive
def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))


@dsl.primitive
def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])


@dsl.primitive
def move(
    grid: Grid,
    obj: Object,
    offset: IntegerTuple
) -> Grid:
    """ move object on grid """
    return paint(cover(grid, obj), shift(obj, offset))


@dsl.primitive
def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]


@dsl.primitive
def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]


@dsl.primitive
def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))


@dsl.primitive
def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))


@dsl.primitive
def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))


@dsl.primitive
def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))


@dsl.primitive
def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


@dsl.primitive
def delta(
    patch: Patch
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)


@dsl.primitive
def gravitate(
    source: Patch,
    destination: Patch
) -> IntegerTuple:
    """ direction to move source until adjacent to destination """
    source_i, source_j = center(source)
    destination_i, destination_j = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if source_i < destination_i else -1
    else:
        j = 1 if source_j < destination_j else -1
    direction = (i, j)
    gravitation_i, gravitation_j = i, j
    maxcount = 42
    c = 0
    while not adjacent(source, destination) and c < maxcount:
        c += 1
        gravitation_i += i
        gravitation_j += j
        source = shift(source, direction)
    return (gravitation_i - i, gravitation_j - j)


@dsl.primitive
def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


@dsl.primitive
def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


@dsl.primitive
def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


@dsl.primitive
def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


@dsl.primitive
def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occurrences = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if 0 <= a < h and 0 <= b < w:
                    if grid[a][b] != v:
                        occurs = False
                        break
                else:
                    occurs = False
                    break
            if occurs:
                occurrences.add((i, j))
    return frozenset(occurrences)


@dsl.primitive
def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers


@dsl.primitive
def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)


@dsl.primitive
def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w


@dsl.primitive
def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h

#############################
###### Define primitives now
#############################

@dsl.primitive
def ic_invert(g: Grid) -> Grid:
    """
    In icecuber, this was filtercol with ID 0, we make it explicit
    Replaces all colours with zeros, and replaces the zeros with the first colour
    In our case, we replace it with the most common colour (arbitrary choice)
    """
    mode = np.argmax(np.bincount(g.grid.ravel())[1:])+1 # skip 0

    grid = np.zeros_like(g.grid)
    grid[g.grid == 0] = mode
    return g.newgrid(grid)

@dsl.primitive
def ic_filtercol(c: Colour, g: Grid) -> Grid:
    "Remove all colours except the selected colour"
    primitive_assert(c != 0, "filtercol with 0 has no effect")

    grid = np.copy(g.grid) # Do we really need to copy? old one thrown away anyway
    grid[grid != c] = 0
    return g.newgrid(grid)

@dsl.primitive
def ic_erasecol(c: Colour, g: Grid) -> Grid:
    "Remove a specified colour from the grid, keeping others intact"
    primitive_assert(c != 0, "erasecol with 0 has no effect")
    grid = np.copy(g.grid)
    grid[grid == c] = 0
    return g.newgrid(grid)

@dsl.primitive
def setcol(c: Colour, g: Grid) -> Grid:
    """
    Set all pixels in the grid to the specified colour.
    This was named colShape in icecuber. 
    """
    primitive_assert(c != 0, "setcol with 0 has no effect")

    grid = np.zeros_like(g.grid)
    grid[np.nonzero(g.grid)] = c
    return g.newgrid(grid)

@dsl.primitive
def set_bg(c: Colour, g: Grid) -> Grid:
    """
    Set all zero-pixels to the specified colour
    """
    primitive_assert(c != 0, "background with 0 has no effect")

    grid = np.copy(g.grid)
    grid[grid == 0] = c
    return g.newgrid(grid)

def ic_compress(g: Grid) -> Grid:
    raise NotImplementedError()

@dsl.primitive
def getpos(g: Grid) -> Position:
    return g.position

@dsl.primitive
def getsize(g: Grid) -> Size:
    return g.size

# TODO: Have a think about position/size/hull and how they fit in
# For now I skip getSize0, getHull, getHull0

@dsl.primitive
def ic_toorigin(g: Grid) -> Grid:
    "Reset a grid's position to zero"
    return Grid(g.grid)

struct4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
@dsl.primitive
def fillobj(c: Colour, g: Grid) -> Grid:
    """
    Fill in any closed objects in the grid with a specified colour.
    The 4-connectedness is used to determine what is a closed object.
    """
    primitive_assert(c != 0, "fill with 0 has no effect")

    binhole = ndimage.binary_fill_holes(g.grid != 0, structure=struct4)
    newgrid = np.copy(g.grid)
    newgrid[binhole & (g.grid == 0)] = c

    return g.newgrid(newgrid)

@dsl.primitive
def ic_fill(g: Grid) -> Grid:
    """
    Returns a grid with all closed objects filled in with the most common colour
    Note that like Icecuber, this also colours everything not connected to the border with the most common colour
    i.e. the result is a single colour
    """
    return setcol(topcol(g), fillobj(1, g))

@dsl.primitive
def ic_interior(g: Grid) -> Grid:
    """
    Returns the *interior* of fillobj - i.e. the filled in objects, but not the original border
    Any result is cast to the top colour.
    """
    filled = fillobj(topcol(g), g)
    filled.grid[g.grid != 0] = 0
    return filled

def ic_interior2(g: Grid) -> Grid:
    raise NotImplementedError

def ic_border(g: Grid) -> Grid:
    raise NotImplementedError

@dsl.primitive
def ic_center(g: Grid) -> Grid:
    # TODO: Figure out why this is useful
    w,h = g.size
    
    newsize = ((w + 1) % 2 + 1, (h + 1) % 2 + 1)
    newgrid = np.ones(newsize)
    newpos = (
        g.position[0] + (newsize[0] - w) / 2,
        g.position[1] + (newsize[1] - h) / 2
    )

    return Grid(newgrid, newpos)

@dsl.primitive
def topcol(g: Grid) -> Colour:
    """
    Returns the most common colour, excluding black.
    majCol in icecuber.
    """
    return np.argmax(np.bincount(g.grid.ravel())[1:])+1

@dsl.primitive
def rarestcol(g: Grid) -> Colour:
    """
    Returns the least common colour, excluding black. 
    Excludes any colours with zero count.
    """
    counts = np.bincount(g.grid.ravel())[1:]
    counts[counts == 0] = 9999
    return np.argmin(counts)+1

## Rigid transformations

@dsl.primitive
def rot90(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid))

@dsl.primitive
def rot180(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid, k=2))

@dsl.primitive
def rot270(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid, k=3))

@dsl.primitive
def flipx(g: Grid) -> Grid:
    return g.newgrid(np.flip(g.grid, axis=0))

@dsl.primitive
def flipy(g: Grid) -> Grid:
    return g.newgrid(np.flip(g.grid, axis=1))

@dsl.primitive
def swapxy(g: Grid) -> Grid:
    return g.newgrid(g.grid.T)

# TODO: Transpose on other dimension

def mirrorHeuristic(g: Grid) -> Grid:
    # Performs a transformation based on centre of gravity
    # TODO
    raise NotImplementedError

####################################
# To/from Counts
####################################

@dsl.primitive
def countPixels(g: Grid) -> Count:
    return np.count_nonzero(g.grid)

@dsl.primitive
def countColours(g: Grid) -> Count:
    """Return the number of unique colours in the grid, excluding zero"""
    return np.count_nonzero(np.bincount(g.grid.ravel())[1:])

@dsl.primitive
def countComponents(g: Grid) -> Count:
    """
    Returns the number of objects in the grid
    Matching the behaviour in icecuber:
    - colours are IGNORED (object can have multiple)
    - diagonals count as the same object (8-structural)
    """
    raise NotImplementedError

# TODO: Figure out how I want to do colours here - is this the best way?
@dsl.primitive
def countToXY(c: Count, col: Colour) -> Grid:
    return Grid(np.zeros((c, c))+col)

@dsl.primitive
def countToX(c: Count, col: Colour) -> Grid:
    return Grid(np.zeros((c, 1))+col)

@dsl.primitive
def countToY(c: Count, col: Colour) -> Grid:
    return Grid(np.zeros((1, c))+col)

@dsl.primitive
def colourHull(c: Colour, g: Grid) -> Grid:
    """
    Returns a grid with the same size as the input, but of the specified colour
    """
    return Grid(np.zeros(g.size)+c)

####################################
# Smearing
####################################

def ic_smear_kernel(g: Grid, kernels: List[Tuple[int, int]]):
    """
    Smear a grid using a kernel.
    The kernel is a tuple of (x, y) offsets.
    """
    raise NotImplementedError

sk = { # smear kernels
    "R": {1, 0},
    "L": {-1, 0},
    "D": {0, 1},
    "U": {0, -1},
    "X": {1, 1},
    "Y": {-1, -1},
    "Z": {1, -1},
    "W": {-1, 1}
}

smear_functions = {
    "R": lambda g: ic_smear_kernel(g, [sk["R"]]),
    "L": lambda g: ic_smear_kernel(g, [sk["L"]]),
    "D": lambda g: ic_smear_kernel(g, [sk["D"]]),
    "U": lambda g: ic_smear_kernel(g, [sk["U"]]),
    "RL": lambda g: ic_smear_kernel(g, [sk["R"], sk["L"]]),
    "DU": lambda g: ic_smear_kernel(g, [sk["D"], sk["U"]]),
    "RLDU": lambda g: ic_smear_kernel(g, [sk["R"], sk["L"], sk["D"], sk["U"]]),
    "X": lambda g: ic_smear_kernel(g, [sk["X"]]),
    "Y": lambda g: ic_smear_kernel(g, [sk["Y"]]),
    "Z": lambda g: ic_smear_kernel(g, [sk["Z"]]),
    "W": lambda g: ic_smear_kernel(g, [sk["W"]]),
    "XY": lambda g: ic_smear_kernel(g, [sk["X"], sk["Y"]]),
    "ZW": lambda g: ic_smear_kernel(g, [sk["Z"], sk["W"]]),
    "XYZW": lambda g: ic_smear_kernel(g, [sk["X"], sk["Y"], sk["Z"], sk["W"]]),
    "RLDUXYZW": lambda g: ic_smear_kernel(g, [sk["R"], sk["L"], sk["D"], sk["U"], sk["X"], sk["Y"], sk["Z"], sk["W"]]),
}

####################################
# border/compression
####################################

@dsl.primitive
def ic_makeborder(g: Grid) -> Grid:
    """
    Return a new grid which is the same as the input, but with a border of 1s around it.
    Only elements which are 0 in the original grid are set to 1 in the new grid.
    8-structuring element used to determine border positions.
    """

    binary_grid = g.grid > 0
    output_grid = np.zeros_like(g.grid)

    grown_binary_grid = binary_dilation(binary_grid, structure=np.ones((3, 3)))
    output_grid[grown_binary_grid & ~binary_grid] = 1

    return g.newgrid(output_grid)

def ic_makeborder2(g: Grid) -> Grid:
    pass

def ic_makeborder2_maj(g: Grid) -> Grid:
    pass

@dsl.primitive
def ic_compress2(g: Grid) -> Grid:
    """Deletes any black rows/columns in the grid"""
    keep_rows = np.any(g.grid, axis=1)
    keep_cols = np.any(g.grid, axis=0)

    return g.newgrid(g.grid[keep_rows][:, keep_cols])

@dsl.primitive
def ic_compress3(g: Grid) -> Grid:
    """
    Keep any rows/columns which differ in any way from the previous row/column
    The first row/column is always kept.
    """
    keep_rows = np.ones(g.grid.shape[0], dtype=bool)
    keep_cols = np.ones(g.grid.shape[1], dtype=bool)

    for row in range(1, g.grid.shape[0]):
        if np.all(g.grid[row] == g.grid[row-1]):
            keep_rows[row] = False

    for col in range(1, g.grid.shape[1]):
        if np.all(g.grid[:, col] == g.grid[:, col-1]):
            keep_cols[col] = False

    return g.newgrid(g.grid[keep_rows][:, keep_cols])

####################################
# Connect
####################################

def ic_connect_kernel(g: Grid, x: bool, y: bool) -> Grid:
    """
    Implements a generic connect (not a primitive)
    x and y control whether it is enabled in the horizontal and vertical directions
    
    Connect works as follows:
    Two cells in the same row/column are connected iff:
    - they are both non-zero
    - they have the same value c
    - there are only zeros in between them
    Connect fills in the zero values in-between with c.
    Note that vertical connection happens after horizontal connection and "overrides" it.

    This function is SLOW
    TODO: Verify this actually works because I'm not confident
    """
    ret = np.zeros_like(g.grid)

    if x:
        for row in range(g.grid.shape[0]):
            last = last_value = -1
            for col in range(g.grid.shape[1]):
                if g.grid[row, col]:
                    if g.grid[row, col] == last_value:
                        ret[row, last+1:col] = last_value
                    last_value = g.grid[row, col]
                    last = col
    
    if y:
        for col in range(g.grid.shape[1]):
            last = last_value = -1
            for row in range(g.grid.shape[0]):
                if g.grid[row, col]:
                    if g.grid[row, col] == last_value:
                        ret[last+1:row, col] = last_value
                    last_value = g.grid[row, col]
                    last = row

    return g.newgrid(ret)


ic_connectX = lambda g: ic_connect_kernel(g, True, False)
ic_connectY = lambda g: ic_connect_kernel(g, False, True)
ic_connectXY = lambda g: ic_connect_kernel(g, True, True)

dsl.register(ic_connectX, "ic_connectX", [tgrid, tgrid])
dsl.register(ic_connectY, "ic_connectY", [tgrid, tgrid])
dsl.register(ic_connectXY, "ic_connectY", [tgrid, tgrid])

####################################
# Spread colours
####################################

@dsl.primitive
def ic_spread(g: Grid) -> Grid:
    """
    Loop through each cell in the grid.
    Place all indices and colours in a queue.

    Later, take indices out of the queue, and for each one:
    - Take the 4 structuring element, if any of the neighbours are not yet coloured, colour them with this colour.

    SLOW
    """

    # Create a queue of (x, y, colour) tuples
    queue = deque()

    # Create a new grid to store the output
    output_grid = np.copy(g.grid)

    # Reproduce the done array
    done = np.bool(g.grid)
    queue = deque([(row, col, g.grid[row, col]) for row, col in zip(*np.where(done))])

    # Loop through the queue
    while queue:
        # Get the next item from the queue
        row, col, colour = queue.popleft()

        # Add the neighbours to the queue
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= x < g.grid.shape[0] and 0 <= y < g.grid.shape[1] and not done[x, y]:
                output_grid[x, y] = colour
                done[x, y] = 1
                queue.append((x, y, colour))

    return g.newgrid(output_grid)

@dsl.primitive
def ic_spread_minor(g: Grid) -> Grid:
    """
    Same as spread, but ignoring the most common colour.
    SLOW
    """
    # Create a queue of (x, y, colour) tuples
    queue = deque()

    # Create a new grid to store the output
    output_grid = np.copy(g.grid)

    # Reproduce the done array
    done = np.bool(g.grid & (g.grid != np.bincount(g.grid.ravel()).argmax()))
    queue = deque([(row, col, g.grid[row, col]) for row, col in zip(*np.where(done))])

    # Loop through the queue
    while queue:
        # Get the next item from the queue
        row, col, colour = queue.popleft()

        # Add the neighbours to the queue
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= x < g.grid.shape[0] and 0 <= y < g.grid.shape[1] and not done[x, y]:
                output_grid[x, y] = colour
                done[x, y] = 1
                queue.append((x, y, colour))

    return g.newgrid(output_grid)

####################################
# Cropping
####################################

@dsl.primitive
def left_half(g: Grid) -> Grid:
    primitive_assert(g.size[1] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:, :g.grid.shape[1]//2])

@dsl.primitive
def right_half(g: Grid) -> Grid:
    """Note that left_half + right_half != identity, middle column may be lost"""
    primitive_assert(g.size[1] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:, -g.grid.shape[1]//2:], 
                    offset=(0, g.grid.shape[1]//2 + g.grid.shape[1]%2))

@dsl.primitive
def top_half(g: Grid) -> Grid:
    primitive_assert(g.size[0] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:g.grid.shape[0]//2])

@dsl.primitive
def bottom_half(g: Grid) -> Grid:
    """Note that top_half + bottom_half != identity, middle row may be lost"""
    primitive_assert(g.size[0] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[-g.grid.shape[0]//2:], 
                    offset=(g.grid.shape[0]//2 + g.grid.shape[0]%2, 0))

# TODO: 25 move primitives, seems kind of excessive...

####################################
# Binary operations
####################################
# In the original code, the 2nd argument is generated by a static shape detector
# We can implment the shape detector as a separate thing that yields Shape
# TODO: binary ops

@dsl.primitive
def ic_embed(img: Grid, shape: Grid) -> Grid:
    """
    Embeds a grid into a larger shape defined by a 2nd argument (zero-padded).
    If the image is larger than the shape, it is cropped.
    """
    ret = np.zeros_like(shape.grid)
    
    xoffset = shape.position[0] - img.position[0]
    yoffset = shape.position[1] - img.position[1]

    xsize = min(img.grid.shape[0], shape.grid.shape[0] - xoffset)
    ysize = min(img.grid.shape[1], shape.grid.shape[1] - yoffset)

    ret[xoffset:xoffset+xsize, yoffset:yoffset+ysize] = img.grid[:xsize, :ysize]
    return shape.newgrid(ret)

def ic_wrap(line: Grid, area: Grid) -> Grid:
    primitive_assert(line.size > (0, 0))
    primitive_assert(area.size > (0, 0))

    zeros_like = np.zeros_like(area.grid)
    raise NotImplementedError

####################################
# Split operations
####################################

def ic_cut(g: Grid) -> List[Grid]:
    """
    Cut up a grid.
    Returns a single colour, cut into at least 2 disjoint pieces. 
    Must touch at least 2 opposite sides, and smallest piece should be as big as possible
    """

    return NotImplementedError
    top_colour = topcol(g)
    ret_score = -1

    colour_mask = g.grid == top_colour
    done = np.zeros_like(colour_mask, dtype=np.bool)

@dsl.primitive
def ic_splitcols(g: Grid) -> List[Grid]:
    """
    Split a grid into multiple grids, each with a single colour.
    """
    ret = []
    for colour in np.unique(g.grid):
        if colour:
            ret.append(g.newgrid(g.grid == colour))
    return ret

@dsl.primitive
def ic_splitall(g: Grid) -> List[Grid]:
    """
    Find all objects using 4-structuring element
    Each colour is separated
    """
    colours = np.unique(g.grid)
    ret = []
    for colour in colours:
        if colour:
            objects = ndimage.find_objects(ndimage.label(g.grid == colour)[0])
            ret += [g.newgrid(g.grid[obj], offset=(obj[0].start, obj[1].start)) for obj in objects]
    return ret

struct8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int)
@dsl.primitive
def split8(g: Grid) -> List[Grid]:
    """
    Find all objects using 8-structuring element
    Each colour is separated
    """
    colours = np.unique(g.grid)
    ret = []
    for colour in colours:
        if colour:
            objects = ndimage.find_objects(ndimage.label(g.grid == colour, structure=struct8)[0])
            ret += [g.newgrid(g.grid[obj], offset=(obj[0].start, obj[1].start)) for obj in objects]
    return ret

@dsl.primitive
def ic_splitcolumns(g: Grid) -> List[Grid]:
    """
    Return all the columns 
    """
    return [g.newgrid(g.grid[:, i:i+1], offset=(0, i)) for i in range(g.grid.shape[1])]

@dsl.primitive
def ic_splitrows(g: Grid) -> List[Grid]:
    """
    Return all the rows 
    """
    return [g.newgrid(g.grid[i:i+1], offset=(i, 0)) for i in range(g.grid.shape[0])]

def ic_insidemarked(g: Grid) -> List[Grid]:
    raise NotImplementedError

####################################
# Join
####################################

def _compress_count_key(o):
    """
    Returns the number of zeros in the grid once compressed.
    """
    comp = ic_compress(o)
    count = np.sum(comp.grid != 0)
    return o.grid.shape[0] * o.grid.shape[1] - count

def majority_col(o):
    """Returns the color that is most common in the grid."""
    return np.argmax(np.bincount(o.grid.ravel()))

def icecuber_fill(o):
    """Returns the grid with all holes filled."""

    a = o.grid
    majority_colour = majority_col(o)
    ret = np.full(o.grid.shape, majority_colour, dtype=np.int8)

    # Determine the majority color of the input grid

    # Create a binary mask of the input grid with True where pixels have a value of 0
    mask = a == 0

    # Create an empty binary mask for storing connected border pixels
    connected_border = np.zeros_like(mask, dtype=bool)

    # Set the border pixels in the connected_border mask
    connected_border[0, :] = mask[0, :]
    connected_border[-1, :] = mask[-1, :]
    connected_border[:, 0] = mask[:, 0]
    connected_border[:, -1] = mask[:, -1]

    # Iteratively perform binary dilation until there are no changes in the connected_border mask
    while True:
        new_connected_border = binary_dilation(connected_border, structure=np.ones((3, 3)), mask=mask)
        if np.array_equal(new_connected_border, connected_border):
            break
        connected_border = new_connected_border

    # Set the connected border pixels to 0 in the `ret` grid
    ret[connected_border] = 0

    # Return the filled grid
    return ret


def _icecuber_interior(o):
    filled = icecuber_fill(o)
    filled[o.grid != 0] = 0
    return filled
    
def _interior_count_key(o):
    interior = _icecuber_interior(o)
    count = np.sum(interior != 0)
    return o.grid.shape[0] * o.grid.shape[1] - count

pickmax_functions = {
    "count":     lambda l: max(l, key=lambda o: np.sum(o != 0)),
    "neg_count": lambda l: min(l, key=lambda o: -np.sum(o != 0)),
    "size":      lambda l: max(l, key=lambda o: o.grid.shape[0] * o.grid.shape[1]),
    "neg_size":  lambda l: min(l, key=lambda o: -o.grid.shape[0] * o.grid.shape[1]),
    "cols":      lambda l: max(l, key=lambda o: len(np.unique(o.grid))),

    # "components": lambda l: max(l, key=lambda o: len(countComponents(o))), # SLOW!

    # "compress_count": lambda l: max(l, key=lambda o: _compress_count_key(o)), # TODO
    # "neg_compress_count": lambda l: min(l, key=lambda o: -_compress_count_key(o)), # TODO
    "interior_count": lambda l: max(l, key=lambda o: _interior_count_key(o)),
    "neg_interior_count": lambda l: min(l, key=lambda o: -_interior_count_key(o)),

    # TODO: p.x/p.y pos/neg
    "x_pos": lambda l: max(l, key=lambda o: o.position[0]),
    "x_neg": lambda l: min(l, key=lambda o: o.position[0]),
    "y_pos": lambda l: max(l, key=lambda o: o.position[1]),
    "y_neg": lambda l: min(l, key=lambda o: o.position[1]),
}

from collections import Counter
@dsl.primitive
def pickcommon(l: List[Grid]) -> Grid:
    """
    Given a list of grids, return the grid which is most common (exactly)
    """
    primitive_assert(len(l) > 0)
    hashes = [hash(g.grid.data.tobytes()) for g in l]
    return l[hashes.index(Counter(hashes).most_common(1)[0][0])]

@dsl.primitive
def ic_pickunique(l: List[Grid]) -> Grid:
    """
    Given a list of grids, return the one which has a unique colour unused by any other grid
    If there are no such grids or more than one, terminate.
    """
    # TODO: think about this implementation a bit more, it might be wrong
    counts = np.zeros(10)
    uniques = [np.unique(g.grid) for g in l]
    for u in uniques:
        counts[u] += 1
    
    colour_mask = counts == 1
    ccount = np.sum(colour_mask)
    if not ccount:
        raise PrimitiveException("pickunique: no unique grids")
    
    for g, u in zip(l, uniques):
        if np.sum(colour_mask[u]) == ccount:
            return g
        
    raise PrimitiveException("pickunique: no unique grids (2)")

@dsl.primitive
def ic_composegrowing(l: List[Grid]) -> Grid:
    """
    Compose a list of grids, to the minimum sized grid that holds all images
    This is position-aware, so two grids which do not overlap will generate one larger grid
    """

    xpos = min([g.position[0] for g in l])
    ypos = min([g.position[1] for g in l])
    xsize = max([g.position[0]+g.size[0] for g in l]) - xpos
    ysize = max([g.position[1]+g.size[1] for g in l]) - ypos

    newgrid = Grid(np.zeros((xsize, ysize)), position=(xpos, ypos))

    # Order grids in DESCENDING order of number of filled pixels
    # TODO: the original code seems like it is ascending, although this doesnt make much sense?
    sorted_list = sorted(l, key=lambda g: g.count())

    for g in sorted_list:
        xstart = g.position[0] - xpos
        ystart = g.position[1] - ypos

        # slice returns a view, assignment to this modifies the original array
        slice = newgrid.grid[xstart:xstart+g.size[0], ystart:ystart+g.size[1]]

        mask = np.nonzero(g.grid)
        slice[mask] = g.grid[mask]

    return newgrid

# TODO: stackLine, myStack, pickMaxes, pickNotMaxes


#############################
# CUSTOM
#############################

@dsl.primitive
def mklist(g: Grid, h: Grid) -> List[Grid]:
    return [g, h]

# cons is taken so we use lcons
@dsl.primitive
def lcons(g: Grid, h: List[Grid]) -> List[Grid]:
    return [g] + h

@dsl.primitive
def overlay(g: Grid, h: Grid) -> Grid:
    """
    If two grids have the same size, overlay them without taking into account position. This returns a new grid at position (0, 0)
    If they have different sizes, we compose them using positions (and provide minimum bounding box), similar to composeGrowing
    Grid h goes on top of grid g.
    """
    if g.size == h.size:
        # New grid, replace any non-zero values with the other grid
        newgrid = Grid(g.grid.copy())
        newgrid.grid[h.grid != 0] = h.grid[h.grid != 0]
        return newgrid
    else:
        xpos = min(g.position[0], h.position[0])
        ypos = min(g.position[1], h.position[1])
        xsize = max(g.position[0]+g.size[0], h.position[0]+h.size[0]) - xpos
        ysize = max(g.position[1]+g.size[1], h.position[1]+h.size[1]) - ypos

        newgrid = Grid(np.zeros((xsize, ysize)), position=(xpos, ypos))

        # First assignment doesnt need any masking
        newgrid.grid[g.position[0]-xpos:g.position[0]-xpos+g.size[0], 
                    g.position[1]-ypos:g.position[1]-ypos+g.size[1]] = g.grid

        # Second assignment does
        mask = np.nonzero(h.grid)
        slice = newgrid.grid[h.position[0]-xpos:h.position[0]-xpos+h.size[0],
                                h.position[1]-ypos:h.position[1]-ypos+h.size[1]]
        slice[mask] = h.grid[mask]

        return newgrid
    
# def colourPixel(c: Colour) -> Grid:
#     """
#     Create a 1x1 grid with a single pixel of colour c
#     """
#     return Grid(np.array([[c]]))

@dsl.primitive
def repeatX(g: Grid) -> Grid:
    """
    Repeat the grid g horizontally, with no gaps
    """
    return Grid(np.tile(g.grid, (1, 2)))

@dsl.primitive
def repeatY(g: Grid) -> Grid:
    """
    Repeat the grid g vertically, with no gaps
    """
    return Grid(np.tile(g.grid, (2, 1)))

@dsl.primitive
def mirrorX(g: Grid) -> Grid:
    """
    Append a reflection of the grid g horizontally
    """
    return Grid(np.hstack((g.grid, np.fliplr(g.grid))))

@dsl.primitive
def mirrorY(g: Grid) -> Grid:
    """
    Append a reflection of the grid g vertically
    """
    return Grid(np.vstack((g.grid, np.flipud(g.grid))))

# def map(f: Callable[[Grid], Grid], l: List[Grid]) -> List[Grid]:
    # print('map', f, len(l), len([f(g) for g in l]))
    # return [f(g) for g in l]

@dsl.primitive(typesig=[arrow(tgrid, tgrid), tgrid, tgrid])
def mapSplit8(f: Callable[[Grid], Grid], g: Grid) -> Grid:
    """
    Split the grid g into objects, apply f to each, and then reassemble
    """
    l = split8(g)
    l = [f(g) for g in l]
    return ic_composegrowing(l)


# def map

@dsl.primitive
def get_bg(c: Colour, g: Grid) -> Grid:
    """
    Return a grid of all the background pixels in g, coloured c
    Essentially same as invert.
    """ 
    return Grid(np.where(g.grid == 0, c, 0))

@dsl.primitive
def logical_and(g: Grid, h: Grid) -> Grid:
    """
    Logical AND between two grids. Use the colour of the first argument
    Logical OR is given by overlay.
    """
    primitive_assert(g.size == h.size, "logical_and: grids must be the same size")

    mask = np.logical_and(g.grid != 0, h.grid != 0)
    return  g.newgrid(np.where(mask, g.grid, 0))
    # return Grid(np.logical_and(g.grid, h.grid))

#############################
# GRAVITY
#############################

def gravity(g: Grid, dx=False, dy=False) -> Grid:
    assert dx or dy

    pieces = ic_splitall(g)

    # Sort pieces by gravity direction
    pieces = sorted(pieces, 
                    key=lambda g: -(g.position[0]*dy+g.position[1]*dx))
    
    # Start with empty grid
    newgrid = Grid(np.zeros(g.size))

    # Iterate over pieces in turn
    for p in pieces:
        while True:
            # Move piece by gravity direction
            p.position = (p.position[0]+dy, p.position[1]+dx)

            # Check that piece is still in bounds
            if p.position[0] < 0 or p.position[0]+p.size[0] > g.size[0] or \
                p.position[1] < 0 or p.position[1]+p.size[1] > g.size[1]:
                p.position = (p.position[0]-dy, p.position[1]-dx)
                break

            # Check that compositing this piece with the new grid doesn't delete any existing pixels
            
            # 1. get the slice of the new grid that this piece will occupy
            slice = newgrid.grid[p.position[0]:p.position[0]+p.size[0],
                                p.position[1]:p.position[1]+p.size[1]]
            # 2. filter that slice by the piece's mask
            # 3. check that any pixels in the slice are not already occupied
            if slice[p.grid != 0].any():
                p.position = (p.position[0]-dy, p.position[1]-dx)
                break

        # Piece now has its new position
        # We can composite it with the new grid
        newgrid = overlay(newgrid, p)
    
    return newgrid

@dsl.primitive
def gravity_down(g: Grid) -> Grid:
    return gravity(g, dy=1)

@dsl.primitive
def gravity_up(g: Grid) -> Grid:
    return gravity(g, dy=-1)

@dsl.primitive
def gravity_left(g: Grid) -> Grid:
    return gravity(g, dx=-1)

@dsl.primitive
def gravity_right(g: Grid) -> Grid:
    return gravity(g, dx=1)


#############################
# PRIMITIVE GENERATION
#############################

# p.registerMany([rot90, rot180, rot270, flipx, flipy, swapxy])

# # Redo the above
# p.registerMany([
#     ic_filtercol,
#     ic_erasecol,
#     setcol,
#     set_bg,

#     # ic_compress,
#     getpos,
#     getsize,

#     #hull?
#     ic_toorigin,
#     fillobj,
#     ic_fill,
#     ic_interior,
#     # ic_interior2,
#     # ic_border,
#     ic_center,
#     topcol,
#     rarestcol
# ])

# rigid?
# count?
# for dir, f in smear_functions.items():
    # p.register(f, f"smear{dir}", arrow(tgrid, tgrid))

# p.registerMany([
#     countPixels,
#     countColours,
#     # countComponents,

#     countToX,
#     countToY,
#     countToXY,
# ])

# p.register(ic_makeborder)
# # p.register(ic_makeborder2)
# # p.register(ic_makeborder2_maj)

# p.register(ic_compress2)
# p.register(ic_compress3)

# dsl.register(ic_connectX, "ic_connectX", [tgrid, tgrid])
# dsl.register(ic_connectY, "ic_connectY", [tgrid, tgrid])
# dsl.register(ic_connectXY, "ic_connectY", [tgrid, tgrid])

# # spreadcols?
# p.registerMany([left_half, right_half, top_half, bottom_half])

# # move?
# p.registerMany([
#     ic_embed, 
#     # ic_wrap
#     # broadcast, repeat, mirror?
#     ])

# p.registerMany([
#     # ic_cut,
#     ic_splitcols,
#     ic_splitall,
#     split8,
#     ic_splitcolumns,
#     ic_splitrows,
#     pickcommon,
#     # ic_insidemarked,
# ])

for name, func in pickmax_functions.items():
    dsl.register(func, f"pickmax_{name}", [tlist(tgrid), tgrid])

# p.register(ic_pickunique)
# p.register(ic_composegrowing)
# stackline, mystrack, pickmaxes, picknotmaxes?

# p.registerMany([mklist, lcons, overlay, 
                # colourPixel, # need to figure out how to do this well: want it for output, but not for input
                # repeatX, repeatY, mirrorX, mirrorY, colourHull, get_bg, logical_and])

# flying too close to the sun
# p.register(mapSplit8, "mapSplit8", [arrow(tgrid, tgrid), tgrid, tgrid])

# Add ground colours
# Skip colour 0 because this doesnt make much sense as an input to colour functions
for i in range(1, 10):
    dsl.register(i, f"c{i}", [tcolour])

# p.registerMany([gravity_down, gravity_up, gravity_left, gravity_right])

# for i in range(1, 2):
#     p.register(i, f"i{i}", [tcount])

print(f"Registered {len(dsl.primitives)} total primitives.")

p = dsl # backwards compatibility
