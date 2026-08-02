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

from std.math import abs, round
from std.memory import bitcast

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


# --------------------------------------------------------------------------
# sin / exp / log10, float64
#
# `lift_coefficient` needs all three and none of them exist for float64 device
# code.  Each follows the same shape: reduce the argument to a small interval
# by an exact-as-possible identity, evaluate a polynomial there, then undo the
# reduction.  The reduction is where precision is normally lost, so the
# constants are split hi/lo (Cody-Waite) and subtracted in two steps.
# --------------------------------------------------------------------------

comptime TWO_OVER_PI: Float64 = 0.63661977236758134308
# pi/2 split so that hi is exact in binary and the product n*hi is too.
comptime PIO2_HI: Float64 = 1.57079632673412561417e+00
comptime PIO2_LO: Float64 = 6.07710050650619224932e-11
comptime LN2_HI: Float64 = 6.93147180369123816490e-01
comptime LN2_LO: Float64 = 1.90821492927058770002e-10
comptime INV_LN2: Float64 = 1.44269504088896338700
comptime INV_LN10: Float64 = 0.43429448190325182765
comptime SQRT_HALF: Float64 = 0.70710678118654752440


@always_inline
def _pow2i(k: Int) -> Float64:
    """2^k, built from the exponent field rather than by multiplying."""
    var biased = UInt64(k + 1023) << 52
    return bitcast[DType.float64](biased)


@always_inline
def _sin_poly(r: Float64) -> Float64:
    """sin on |r| <= pi/4."""
    var z = r * r
    return r + r * z * (
        -1.66666666666666324348e-01
        + z * (
            8.33333333332248946124e-03
            + z * (
                -1.98412698298579493134e-04
                + z * (
                    2.75573137070700676789e-06
                    + z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10)
                )
            )
        )
    )


@always_inline
def _cos_poly(r: Float64) -> Float64:
    """cos on |r| <= pi/4.

    Written as w + ((1-w)-hz + ...) rather than 1 - z/2 + ... because the
    leading subtraction cancels badly near pi/4 and this recovers the bits.
    """
    var z = r * r
    var poly = z * z * (
        4.16666666666666019037e-02
        + z * (
            -1.38888888888741095749e-03
            + z * (
                2.48015872894767294178e-05
                + z * (
                    -2.75573143513906633035e-07
                    + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11)
                )
            )
        )
    )
    var hz = 0.5 * z
    var w = 1.0 - hz
    return w + ((1.0 - w) - hz + poly)


@always_inline
def sind(x: Float64) -> Float64:
    """sin for device code, double precision.

    Cody-Waite reduction mod pi/2.  The solver only ever asks for sin(2*alpha)
    with alpha folded into [-pi/2, pi/2], so the argument stays small and the
    naive round-to-nearest reduction is accurate; there is no Payne-Hanek path
    and none is needed.
    """
    var n = Int(round(x * TWO_OVER_PI))
    var qn = Float64(n)
    var r = (x - qn * PIO2_HI) - qn * PIO2_LO
    var quad = n & 3
    if quad == 0:
        return _sin_poly(r)
    if quad == 1:
        return _cos_poly(r)
    if quad == 2:
        return -_sin_poly(r)
    return -_cos_poly(r)


@always_inline
def expd(x: Float64) -> Float64:
    """exp for device code, double precision."""
    if x > 709.0:
        return bitcast[DType.float64](UInt64(0x7FF0000000000000))  # +inf
    if x < -745.0:
        return 0.0
    var k = Int(round(x * INV_LN2))
    var fk = Float64(k)
    var r = (x - fk * LN2_HI) - fk * LN2_LO
    # Taylor on |r| <= ln2/2 = 0.347; through 1/15! the truncation is far below
    # a double's last bit.
    var s = 1.0 + r * (
        1.0
        + r * (
            5.00000000000000000000e-01
            + r * (
                1.66666666666666666667e-01
                + r * (
                    4.16666666666666666667e-02
                    + r * (
                        8.33333333333333333333e-03
                        + r * (
                            1.38888888888888888889e-03
                            + r * (
                                1.98412698412698412698e-04
                                + r * (
                                    2.48015873015873015873e-05
                                    + r * (
                                        2.75573192239858906526e-06
                                        + r * (
                                            2.75573192239858906526e-07
                                            + r * (
                                                2.50521083854417187751e-08
                                                + r * (
                                                    2.08767569878680989792e-09
                                                    + r * (
                                                        1.60590438368216145994e-10
                                                        + r * 1.14707455977297247139e-11
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return s * _pow2i(k)


@always_inline
def logd(x: Float64) -> Float64:
    """Natural log for device code, double precision.

    Splits off the exponent, folds the mantissa to [sqrt(1/2), sqrt(2)) so the
    atanh series converges fast, then sums 2*atanh((m-1)/(m+1)).  Callers here
    only ever pass Reynolds numbers >= 10, so the zero and negative cases
    return sentinels rather than being handled properly.
    """
    if x <= 0.0:
        return -1.0e308
    var bits = bitcast[DType.uint64](x)
    var e = Int((bits >> 52) & 0x7FF) - 1023
    # mantissa forced into [1, 2)
    var m = bitcast[DType.float64]((bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000)
    if m > 1.0 / SQRT_HALF:
        m = m * 0.5
        e += 1
    var s = (m - 1.0) / (m + 1.0)
    var z = s * s
    # 2*(s + s^3/3 + ... + s^17/17); relative truncation ~1e-15 at |s| <= 0.1716
    var poly = s * (
        1.0
        + z * (
            3.33333333333333333333e-01
            + z * (
                2.00000000000000000000e-01
                + z * (
                    1.42857142857142857143e-01
                    + z * (
                        1.11111111111111111111e-01
                        + z * (
                            9.09090909090909090909e-02
                            + z * (
                                7.69230769230769230769e-02
                                + z * (
                                    6.66666666666666666667e-02
                                    + z * 5.88235294117647058824e-02
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return 2.0 * poly + Float64(e) * (LN2_HI + LN2_LO)


@always_inline
def log10d(x: Float64) -> Float64:
    return logd(x) * INV_LN10


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
