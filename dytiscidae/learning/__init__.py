"""Learned controllers that are shared across morphologies.

Everything in `control/` belongs to one machine: a CPG tuned to its joints, a
mobility basis measured on its body, a policy whose weights are inherited down
one lineage.  Everything here is the opposite -- one set of parameters trained on
every machine at once.
"""
