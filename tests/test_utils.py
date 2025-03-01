"""Basic tests for a2c_ase.utils module."""

import numpy as np
from pymatgen.core.composition import Composition

from a2c_ase.utils import get_diameter, get_target_temperature


def test_get_diameter():
    """Test the get_diameter function."""
    comp = Composition("Si")
    diameter = get_diameter(comp)
    assert diameter > 0, "Diameter should be positive"

    comp = Composition("Fe2O3")
    diameter = get_diameter(comp)
    assert diameter > 0, "Diameter should be positive"


def test_get_target_temperature():
    """Test the get_target_temperature function."""
    # During high-temp phase
    temp = get_target_temperature(50, 100, 200, 2000.0, 300.0)
    assert temp == 2000.0

    # During cooling phase (halfway)
    temp = get_target_temperature(200, 100, 200, 2000.0, 300.0)
    assert np.isclose(temp, 1150.0)

    # After cooling phase
    temp = get_target_temperature(350, 100, 200, 2000.0, 300.0)
    assert temp == 300.0
