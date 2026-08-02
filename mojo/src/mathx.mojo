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


comptime PI_D: Float64 = 3.14159265358979323846
comptime HALF_PI_D: Float64 = PI_D / 2.0
comptime QUARTER_PI_D: Float64 = PI_D / 4.0
# tan(pi/8): above this the ratio gets folded again so the polynomial only ever
# sees |u| <= 0.4143, which is where the coefficients below are accurate.
comptime TAN_PI_8: Float64 = 0.41421356237309504880


@always_inline
def _atan_small_d(x: Float64) -> Float64:
    """atan on |x| <= tan(pi/8), to double precision.

    FDLIBM's `aT` coefficients, evaluated as two interleaved even/odd chains so
    the polynomial has some instruction-level parallelism.  These are exact
    arithmetic -- no libm call -- which is the whole reason this exists: `atan`
    and `atan2` do not link for NVIDIA device code, and float64 transcendentals
    are unsupported there besides.
    """
    var z = x * x
    var w = z * z
    var s1 = z * (
        3.33333333333329318027e-01
        + w * (
            1.42857142725034663711e-01
            + w * (
                9.09088713343650656196e-02
                + w * (
                    6.66107313738753120669e-02
                    + w * (4.97687799461593236017e-02 + w * 1.62858201153657823623e-02)
                )
            )
        )
    )
    var s2 = w * (
        -1.99999999998764832476e-01
        + w * (
            -1.11111104054623557880e-01
            + w * (
                -7.69187620504482999495e-02
                + w * (-5.83357013379057348645e-02 + w * -3.65315727442169155270e-02)
            )
        )
    )
    return x - x * (s1 + s2)


@always_inline
def atan2d(y: Float64, x: Float64) -> Float64:
    """Double-precision two-argument arctangent for device code.

    Two reductions before the polynomial: swap to bring |ratio| <= 1, then the
    pi/8 identity atan(t) = pi/4 + atan((t-1)/(t+1)) to bring it inside the
    coefficients' range.  Quadrant is restored afterwards.
    """
    var ax = abs(x)
    var ay = abs(y)
    if ax == 0.0 and ay == 0.0:
        return 0.0
    var swap = ay > ax
    var num = ax if swap else ay
    var den = ay if swap else ax
    var t = num / den

    var r: Float64
    if t > TAN_PI_8:
        r = QUARTER_PI_D + _atan_small_d((t - 1.0) / (t + 1.0))
    else:
        r = _atan_small_d(t)

    if swap:
        r = HALF_PI_D - r
    if x < 0.0:
        r = PI_D - r
    if y < 0.0:
        r = -r
    return r


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
