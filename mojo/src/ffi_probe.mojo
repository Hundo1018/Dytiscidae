"""Can Mojo take a numpy buffer address from Python and write through it?

Everything downstream depends on marshalling arrays without copying them
element by element through PythonObject, so this is checked before any real
kernel is written.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_ffi_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("ffi_probe")
        m.def_function[double_in_place]("double_in_place")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))


def double_in_place(arr_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Double `n` float64s living at the given address."""
    var addr = Int(py=arr_addr)
    var count = Int(py=n)
    var ptr = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr)
    for i in range(count):
        ptr[i] = ptr[i] * 2.0
    return PythonObject(count)
