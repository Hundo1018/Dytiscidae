"""Device-side math the GPU stdlib does not provide.

`atan2` does not link for NVIDIA device code -- a kernel that calls it dies at
compile time with:

    ptxas fatal : Unresolved extern function 'atan2f'

and `atan` is CPU-only ("libm operations are only available on CPU targets").
The fluid solver needs two atan2 calls per panel to get angle of attack, so
this is on the critical path of any GPU port and has to be supplied here.

Accuracy is measured, not assumed -- see tests/test_mathx.mojo.  Over 8192
samples spanning all four quadrants and both sides of the |y|>|x| swap, the
worst error against numpy's atan2 is 1.17e-05 rad (0.00067 degrees).  That is
about 100x float32 epsilon and irrelevant here: the leading-edge vortex gate it
feeds switches near 40 degrees of incidence, and the solver's own quasi-steady
assumption is good to maybe 30%.
"""

from std.math import abs

comptime PI: Float32 = 3.14159265358979323846
comptime HALF_PI: Float32 = PI / 2.0


@always_inline
def atan_unit(x: Float32) -> Float32:
    """Rational minimax approximation of atan on |x| <= 1."""
    var z = x * x
    return x * (
        0.9998660
        + z * (-0.3302995 + z * (0.1801410 + z * (-0.0851330 + z * 0.0208351)))
    )


@always_inline
def atan2f(y: Float32, x: Float32) -> Float32:
    """Two-argument arctangent, valid on the whole plane.

    Reduces to |ratio| <= 1 before evaluating the polynomial -- the series is
    only accurate on the unit interval, and the swap is what keeps the error
    flat instead of blowing up as the ratio grows.
    """
    var ax = abs(x)
    var ay = abs(y)
    var swap = ay > ax
    var num = ax if swap else ay
    var den = ay if swap else ax
    var r = atan_unit(num / den) if den != 0.0 else Float32(0.0)
    if swap:
        r = HALF_PI - r
    if x < 0.0:
        r = PI - r
    if y < 0.0:
        r = -r
    return r
