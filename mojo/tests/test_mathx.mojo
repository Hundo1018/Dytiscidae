"""Pin the accuracy of the hand-rolled device math.

These bounds are not aspirational -- they are the measured worst case with
headroom.  If a future Mojo gains a working device `atan2`, delete `mathx` and
these tests together rather than loosening them.
"""

from std.math import abs, atan2, sin, exp, log10
from std.testing import assert_true, TestSuite

from mathx import atan2f, atan2d, sind, expd, logd, log10d, PI


def _worst_error() -> Float32:
    """Largest disagreement with the CPU atan2 over a full sweep."""
    var worst: Float32 = 0.0
    comptime N = 8192
    for i in range(N):
        var x = Float32(i % 127) - 63.0
        var y = Float32((i * 7) % 131) - 65.0
        var got = atan2f(y, x)
        var expected = atan2(y, x)
        var e = abs(got - expected)
        # Both branches of the +/-pi wrap are correct answers.
        var wrapped = abs(e - 2.0 * PI)
        if wrapped < e:
            e = wrapped
        if e > worst:
            worst = e
    return worst


def test_atan2_matches_libm() raises:
    var worst = _worst_error()
    print("worst atan2 error:", worst, "rad")
    # Measured 1.168e-05; fail if it degrades by more than ~2x.
    assert_true(worst < 2.5e-5, "atan2f drifted from the measured envelope")


def test_atan2_quadrants() raises:
    """The quadrant fixups are where a hand-rolled atan2 usually goes wrong."""
    assert_true(abs(atan2f(0.0, 1.0)) < 1e-5, "+x axis")
    assert_true(abs(atan2f(1.0, 0.0) - PI / 2.0) < 1e-5, "+y axis")
    assert_true(abs(abs(atan2f(0.0, -1.0)) - PI) < 1e-5, "-x axis")
    assert_true(abs(atan2f(-1.0, 0.0) + PI / 2.0) < 1e-5, "-y axis")
    # Signs must survive into every quadrant.
    assert_true(atan2f(1.0, 1.0) > 0.0 and atan2f(1.0, 1.0) < PI / 2.0, "Q1")
    assert_true(atan2f(1.0, -1.0) > PI / 2.0, "Q2")
    assert_true(atan2f(-1.0, -1.0) < -PI / 2.0, "Q3")
    assert_true(atan2f(-1.0, 1.0) < 0.0, "Q4")


def test_atan2d_is_double_precision() raises:
    """The float64 path exists so the ported solver can stay bit-comparable."""
    var worst: Float64 = 0.0
    comptime N = 20000
    for i in range(N):
        # Sweep magnitudes as well as angles: the pi/8 reduction and the swap
        # both have seams, and a uniform sweep misses them.
        var x = Float64((i % 211) - 105) * (1.0 + Float64(i % 7))
        var y = Float64(((i * 13) % 223) - 111) * (1.0 + Float64(i % 5))
        var got = atan2d(y, x)
        var expected = atan2(y, x)
        var e = abs(got - expected)
        var wrapped = abs(e - 2.0 * Float64(PI))
        if wrapped < e:
            e = wrapped
        if e > worst:
            worst = e
    print("worst atan2d error:", worst, "rad")
    assert_true(worst < 1e-14, "atan2d is not double precision")


def test_sind_matches_libm() raises:
    """sin over the range the solver asks for, and well past it.

    `lift_coefficient` wants sin(2*alpha) with alpha folded to [-pi/2, pi/2],
    so |x| <= ~3.3 covers it.  Swept to 20 anyway: the reduction has a seam at
    every multiple of pi/2 and a short sweep would only ever see the first two.
    """
    var worst: Float64 = 0.0
    comptime N = 40000
    for i in range(N):
        var x = (Float64(i) / Float64(N) - 0.5) * 40.0
        var e = abs(sind(x) - sin(x))
        if e > worst:
            worst = e
    print("worst sind error:", worst)
    assert_true(worst < 1e-15, "sind is not double precision")


def test_expd_beats_the_stdlib_it_cannot_use_as_an_oracle() raises:
    """exp, checked against exact values rather than against `std.math.exp`.

    Mojo's float64 `exp` is not correctly rounded: its relative error grows
    linearly with the argument, reaching 1.2e-11 by x=30 (measured against
    CPython's math.exp, which is correctly rounded here).  `expd` is exact at
    every one of these points, so testing it against the stdlib reported the
    stdlib's error as this function's and failed a correct implementation.

    Hence literals.  Each is the exactly-rounded double for that argument,
    generated once from CPython; they are the oracle precisely because they do
    not depend on any runtime exp.
    """
    var xs = [-30.0, -12.5, -3.0, -0.5, 0.5, 3.0, 12.5, 30.0, 700.0]
    var expected = [
        9.357622968840175e-14,
        3.726653172078671e-06,
        0.049787068367863944,
        0.6065306597126334,
        1.6487212707001282,
        20.085536923187668,
        268337.2865208745,
        10686474581524.463,
        1.0142320547350045e304,
    ]
    var worst: Float64 = 0.0
    for i in range(len(xs)):
        var e = abs(expd(xs[i]) - expected[i]) / expected[i]
        if e > worst:
            worst = e
    print("worst expd relative error vs exact:", worst)
    assert_true(worst < 1e-15, "expd is not double precision")


def test_expd_and_logd_invert_each_other() raises:
    """A dense check that needs no oracle at all.

    log(exp(x)) == x over the whole range exercises both reductions against
    each other, including every power-of-two seam that a spot check walks past.
    """
    var worst: Float64 = 0.0
    comptime N = 40000
    for i in range(N):
        var x = (Float64(i) / Float64(N) - 0.5) * 60.0
        var back = logd(expd(x))
        var denom = abs(x) if abs(x) > 1.0 else 1.0
        var e = abs(back - x) / denom
        if e > worst:
            worst = e
    print("worst exp/log round-trip error:", worst)
    assert_true(worst < 1e-14, "expd and logd do not invert")


def test_expd_saturates_instead_of_producing_garbage() raises:
    """The sigmoid feeds it large negatives; those must go to zero cleanly."""
    assert_true(expd(-800.0) == 0.0, "large negative should flush to zero")
    assert_true(expd(800.0) > 1.0e307, "large positive should saturate high")


def test_log10d_matches_libm() raises:
    """log10 over Reynolds numbers, which is the only thing that calls it.

    `lift_coefficient` clamps its argument to >= 10 and this project's Reynolds
    numbers reach ~1e7, so the sweep spans 1 to 1e10 geometrically rather than
    linearly -- a linear sweep of that range would put almost every sample in
    the top decade and never test the mantissa fold.
    """
    var worst: Float64 = 0.0
    comptime N = 40000
    for i in range(N):
        var x = expd(Float64(i) / Float64(N) * 23.03)  # 1 .. ~1e10
        var got = log10d(x)
        var expected = log10(x)
        var denom = abs(expected) if abs(expected) > 1.0 else 1.0
        var e = abs(got - expected) / denom
        if e > worst:
            worst = e
    print("worst log10d relative error:", worst)
    assert_true(worst < 1e-15, "log10d is not double precision")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
