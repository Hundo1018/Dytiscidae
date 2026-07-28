"""Dytiscidae -- a generative design + control-learning loop for triphibian
(air / water / land) flapping-wing machines.

The package is deliberately layered so that each layer can be tested, replaced
or calibrated on its own:

    physics/    medium, quasi-steady blade-element fluid, energy, structure
    core/       genome -> phenotype -> MJCF compilation
    envs/       the triphibian mission env and the actuator-skill env
    evolution/  MAP-Elites archive, CMA-ES, the curator, the co-evolution loop
    learning/   policy representations and the (optional) PPO learner
    viz/        telemetry readers, archive plots, offscreen rendering
    ops/        the orchestrator: run, checkpoint, resume

Nothing here assumes a GPU.  See docs/DESIGN.md for the physics conventions.
"""

__version__ = "0.1.0"

# World conventions used everywhere in this package:
#   * Right-handed world frame, +Z is up.
#   * z = 0 is the still-water free surface.  z > 0 is air, z < 0 is water.
#   * Land is a heightfield that pokes through the surface; the shoreline is
#     therefore an emergent feature of the terrain, not a special case.
#   * All SI units.  Angles in radians.  Power in watts, energy in joules.
WORLD_UP = 2  # index of the up axis
