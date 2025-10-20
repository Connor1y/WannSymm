"""
Tests for wannorb module

Test Wannier orbital structures.
"""

import pytest
from wannsymm.vector import Vector
from wannsymm.wannorb import WannOrb, init_wannorb, find_index_of_wannorb


class TestWannOrbCreation:
    """Test WannOrb creation and initialization."""

    def test_s_orbital_creation(self):
        """Test creation of s orbital (l=0)."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)

        assert orb.site == site
        assert orb.l == 0
        assert orb.r == 1
        assert orb.mr == 0
        assert orb.ms == 0
        assert len(orb.axis) == 3

    def test_p_orbital_creation(self):
        """Test creation of p orbital (l=1)."""
        site = Vector(0.5, 0.5, 0.5)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=2, l=1, mr=1, ms=0)

        assert orb.l == 1
        assert orb.r == 2
        assert orb.mr == 1

    def test_d_orbital_creation(self):
        """Test creation of d orbital (l=2)."""
        site = Vector(0.25, 0.25, 0.25)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=3, l=2, mr=2, ms=1)

        assert orb.l == 2
        assert orb.r == 3
        assert orb.ms == 1  # spin down

    def test_f_orbital_creation(self):
        """Test creation of f orbital (l=3)."""
        site = Vector(0.75, 0.75, 0.75)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=4, l=3, mr=3, ms=0)

        assert orb.l == 3
        assert orb.r == 4

    def test_spin_up_orbital(self):
        """Test orbital with spin up (ms=0)."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)

        assert orb.ms == 0

    def test_spin_down_orbital(self):
        """Test orbital with spin down (ms=1)."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=1)

        assert orb.ms == 1


class TestWannOrbValidation:
    """Test validation of quantum numbers and spin values."""

    def test_negative_r_raises_error(self):
        """Test that negative r raises ValueError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        with pytest.raises(
            ValueError, match="Radial quantum number r must be non-negative"
        ):
            WannOrb(site=site, axis=axis, r=-1, l=0, mr=0, ms=0)

    def test_invalid_l_negative_raises_error(self):
        """Test that negative l raises ValueError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        with pytest.raises(ValueError, match="Angular momentum l must be"):
            WannOrb(site=site, axis=axis, r=1, l=-1, mr=0, ms=0)

    def test_invalid_l_too_large_raises_error(self):
        """Test that l > 3 raises ValueError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        with pytest.raises(ValueError, match="Angular momentum l must be"):
            WannOrb(site=site, axis=axis, r=1, l=4, mr=0, ms=0)

    def test_invalid_ms_negative_raises_error(self):
        """Test that ms not in {0, 1} raises ValueError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        with pytest.raises(ValueError, match="Spin quantum number ms must be"):
            WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=-1)

    def test_invalid_ms_too_large_raises_error(self):
        """Test that ms > 1 raises ValueError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        with pytest.raises(ValueError, match="Spin quantum number ms must be"):
            WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=2)

    def test_invalid_axis_length_raises_error(self):
        """Test that axis not having 3 vectors raises ValueError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0)]  # Only 2 vectors

        with pytest.raises(ValueError, match="axis must contain exactly 3 vectors"):
            WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)

    def test_invalid_axis_type_raises_error(self):
        """Test that axis containing non-Vector raises TypeError."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), (0, 0, 1)]  # Last is tuple

        with pytest.raises(TypeError, match="axis\\[2\\] must be a Vector instance"):
            WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)


class TestWannOrbEquality:
    """Test equality checks for WannOrb."""

    def test_equal_orbitals(self):
        """Test that identical orbitals are equal."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)

        assert orb1 == orb2

    def test_different_site_not_equal(self):
        """Test that orbitals with different sites are not equal."""
        site1 = Vector(0.0, 0.0, 0.0)
        site2 = Vector(1.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site1, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site2, axis=axis, r=1, l=0, mr=0, ms=0)

        assert orb1 != orb2

    def test_different_l_not_equal(self):
        """Test that orbitals with different l are not equal."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis, r=1, l=1, mr=0, ms=0)

        assert orb1 != orb2

    def test_different_ms_not_equal(self):
        """Test that orbitals with different spin are not equal."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=1)

        assert orb1 != orb2

    def test_different_r_not_equal(self):
        """Test that orbitals with different r are not equal."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis, r=2, l=0, mr=0, ms=0)

        assert orb1 != orb2

    def test_different_mr_not_equal(self):
        """Test that orbitals with different mr are not equal."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis, r=1, l=0, mr=1, ms=0)

        assert orb1 != orb2

    def test_different_axis_not_equal(self):
        """Test that orbitals with different axis are not equal."""
        site = Vector(0.0, 0.0, 0.0)
        axis1 = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        axis2 = [Vector(0, 1, 0), Vector(1, 0, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis1, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis2, r=1, l=0, mr=0, ms=0)

        assert orb1 != orb2

    def test_comparison_with_non_wannorb(self):
        """Test that comparison with non-WannOrb returns False."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)

        assert orb != "not a wannorb"
        assert orb != 42
        assert orb is not None


class TestWannOrbRepr:
    """Test string representation of WannOrb."""

    def test_s_orbital_repr(self):
        """Test repr of s orbital."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)

        repr_str = repr(orb)
        assert "WannOrb" in repr_str
        assert "l=0(s)" in repr_str
        assert "ms=0(up)" in repr_str

    def test_p_orbital_repr(self):
        """Test repr of p orbital."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=2, l=1, mr=0, ms=1)

        repr_str = repr(orb)
        assert "l=1(p)" in repr_str
        assert "ms=1(down)" in repr_str

    def test_d_orbital_repr(self):
        """Test repr of d orbital."""
        site = Vector(0.5, 0.5, 0.5)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=3, l=2, mr=1, ms=0)

        repr_str = repr(orb)
        assert "l=2(d)" in repr_str
        assert "r=3" in repr_str

    def test_f_orbital_repr(self):
        """Test repr of f orbital."""
        site = Vector(0.25, 0.25, 0.25)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb = WannOrb(site=site, axis=axis, r=4, l=3, mr=2, ms=1)

        repr_str = repr(orb)
        assert "l=3(f)" in repr_str
        assert "mr=2" in repr_str


class TestInitWannOrb:
    """Test init_wannorb function."""

    def test_init_wannorb_basic(self):
        """Test basic initialization with init_wannorb."""
        site = Vector(0.0, 0.0, 0.0)
        axisx = Vector(1.0, 0.0, 0.0)
        axisy = Vector(0.0, 1.0, 0.0)
        axisz = Vector(0.0, 0.0, 1.0)

        orb = init_wannorb(
            site, l=1, mr=0, ms=0, r=2, axisz=axisz, axisx=axisx, axisy=axisy
        )

        assert orb.site == site
        assert orb.l == 1
        assert orb.mr == 0
        assert orb.ms == 0
        assert orb.r == 2
        assert orb.axis[0] == axisx
        assert orb.axis[1] == axisy
        assert orb.axis[2] == axisz

    def test_init_wannorb_axis_order(self):
        """Test that axis order is [x, y, z] as in C code."""
        site = Vector(0.0, 0.0, 0.0)
        axisx = Vector(1.0, 0.0, 0.0)
        axisy = Vector(0.0, 1.0, 0.0)
        axisz = Vector(0.0, 0.0, 1.0)

        orb = init_wannorb(
            site, l=0, mr=0, ms=0, r=1, axisz=axisz, axisx=axisx, axisy=axisy
        )

        # Check that axis is stored in correct order [x, y, z]
        assert orb.axis[0].x == 1.0 and orb.axis[0].y == 0.0
        assert orb.axis[1].x == 0.0 and orb.axis[1].y == 1.0
        assert orb.axis[2].x == 0.0 and orb.axis[2].z == 1.0


class TestFindIndexOfWannOrb:
    """Test find_index_of_wannorb function."""

    def test_find_existing_orbital(self):
        """Test finding an existing orbital in a list."""
        site1 = Vector(0.0, 0.0, 0.0)
        site2 = Vector(1.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        orb1 = WannOrb(site=site1, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site2, axis=axis, r=1, l=1, mr=0, ms=0)
        wann_list = [orb1, orb2]

        idx = find_index_of_wannorb(wann_list, site1, r=1, l=0, mr=0, ms=0)
        assert idx == 0

        idx = find_index_of_wannorb(wann_list, site2, r=1, l=1, mr=0, ms=0)
        assert idx == 1

    def test_find_nonexistent_orbital(self):
        """Test that nonexistent orbital returns -1."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        wann_list = [orb]

        # Different site
        idx = find_index_of_wannorb(
            wann_list, Vector(1.0, 0.0, 0.0), r=1, l=0, mr=0, ms=0
        )
        assert idx == -1

        # Different r
        idx = find_index_of_wannorb(wann_list, site, r=2, l=0, mr=0, ms=0)
        assert idx == -1

        # Different l
        idx = find_index_of_wannorb(wann_list, site, r=1, l=1, mr=0, ms=0)
        assert idx == -1

        # Different ms
        idx = find_index_of_wannorb(wann_list, site, r=1, l=0, mr=0, ms=1)
        assert idx == -1

    def test_find_in_empty_list(self):
        """Test finding in empty list returns -1."""
        wann_list = []
        idx = find_index_of_wannorb(wann_list, Vector(0, 0, 0), r=1, l=0, mr=0, ms=0)
        assert idx == -1

    def test_find_multiple_orbitals(self):
        """Test finding orbitals in a list with multiple entries."""
        site = Vector(0.0, 0.0, 0.0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        # Create multiple orbitals with same site but different quantum numbers
        orbitals = []
        for l_val in range(4):  # s, p, d, f
            for ms_val in [0, 1]:  # spin up, down
                orb = WannOrb(site=site, axis=axis, r=1, l=l_val, mr=0, ms=ms_val)
                orbitals.append(orb)

        # Find each orbital
        for l_val in range(4):
            for ms_val in [0, 1]:
                idx = find_index_of_wannorb(
                    orbitals, site, r=1, l=l_val, mr=0, ms=ms_val
                )
                assert idx >= 0
                assert orbitals[idx].l == l_val
                assert orbitals[idx].ms == ms_val
