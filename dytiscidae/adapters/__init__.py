"""Adapters: everything that knows about a technology.

The outside of the hexagon.  Each subpackage implements one or more ports:

    filesystem/   JSON and JSONL on disk -- the default for every port
    sqlite/       the record and the lifecycle in a database
    trainers/     the Trainer port: the MuJoCo search, and a dependency-free
                  reference implementation
    launchers     process isolation: inline, and an own-session subprocess
    composition   the one module that names all of the above

Importing this package imports none of them.  A submodule is imported when it
is used, so ``import dytiscidae.application`` stays free of numpy, torch and
MuJoCo -- which is the property the whole arrangement exists to have, and which
``tests/test_architecture.py`` measures rather than assumes.
"""
