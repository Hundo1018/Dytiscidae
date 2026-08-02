"""Pin the accuracy of the hand-rolled device math.

These bounds are not aspirational -- they are the measured worst case with
headroom.  If a future Mojo gains a working device `atan2`, delete `mathx` and
these tests together rather than loosening them.
"""

from std.math import abs, atan2
from std.testing import assert_true, TestSuite

from mathx import atan2f, PI


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
