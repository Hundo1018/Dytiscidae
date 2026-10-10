"""Warp kernels for Q6, kept out of run.py so run.py's --selftest never imports warp."""
import warp as wp


@wp.kernel
def add_one(a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a[i] = a[i] + 1.0
