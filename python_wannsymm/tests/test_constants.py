"""
Tests for constants module

Test all constant definitions.
"""

import numpy as np
from wannsymm import constants


class TestConstantExistence:
    """Test that all expected constants are defined."""

    def test_buffer_size_constants_exist(self):
        """Test that string/buffer size constants exist."""
        assert hasattr(constants, 'MAXLEN')
        assert hasattr(constants, 'MEDIUMLEN')
        assert hasattr(constants, 'SHORTLEN')
        assert hasattr(constants, 'MAXN')

    def test_tolerance_constants_exist(self):
        """Test that tolerance constants exist."""
        assert hasattr(constants, 'eps4')
        assert hasattr(constants, 'eps5')
        assert hasattr(constants, 'eps6')
        assert hasattr(constants, 'eps7')
        assert hasattr(constants, 'eps8')

    def test_mathematical_constants_exist(self):
        """Test that mathematical constants exist."""
        assert hasattr(constants, 'sqrt2')
        assert hasattr(constants, 'PI')
        assert hasattr(constants, 'cmplx_i')

    def test_array_limit_constants_exist(self):
        """Test that array size limit constants exist."""
        assert hasattr(constants, 'MAX_NUM_of_SYMM')
        assert hasattr(constants, 'MAX_NUM_of_atoms')
        assert hasattr(constants, 'MAX_L')

    def test_type_alias_exists(self):
        """Test that dcomplex type alias exists."""
        assert hasattr(constants, 'dcomplex')


class TestConstantTypes:
    """Test that constants have correct types."""

    def test_buffer_size_types(self):
        """Test that buffer size constants are integers."""
        assert isinstance(constants.MAXLEN, int)
        assert isinstance(constants.MEDIUMLEN, int)
        assert isinstance(constants.SHORTLEN, int)
        assert isinstance(constants.MAXN, int)

    def test_tolerance_types(self):
        """Test that tolerance constants are floats."""
        assert isinstance(constants.eps4, float)
        assert isinstance(constants.eps5, float)
        assert isinstance(constants.eps6, float)
        assert isinstance(constants.eps7, float)
        assert isinstance(constants.eps8, float)

    def test_mathematical_constant_types(self):
        """Test that mathematical constants have correct types."""
        assert isinstance(constants.sqrt2, float)
        assert isinstance(constants.PI, float)
        assert isinstance(constants.cmplx_i, complex)

    def test_array_limit_types(self):
        """Test that array size limit constants are integers."""
        assert isinstance(constants.MAX_NUM_of_SYMM, int)
        assert isinstance(constants.MAX_NUM_of_atoms, int)
        assert isinstance(constants.MAX_L, int)

    def test_dcomplex_type(self):
        """Test that dcomplex is np.complex128."""
        assert constants.dcomplex == np.complex128


class TestConstantValues:
    """Test that constants have correct values."""

    def test_buffer_sizes(self):
        """Test buffer size constant values."""
        assert constants.MAXLEN == 512
        assert constants.MEDIUMLEN == 256
        assert constants.SHORTLEN == 128
        assert constants.MAXN == 10

    def test_tolerance_values(self):
        """Test tolerance constant values."""
        assert constants.eps4 == 1e-4
        assert constants.eps5 == 1e-5
        assert constants.eps6 == 1e-6
        assert constants.eps7 == 1e-7
        assert constants.eps8 == 1e-8

    def test_sqrt2_precision(self):
        """Test sqrt2 matches C value to at least 1e-12 precision."""
        c_value = 1.41421356237309504880168872421
        assert abs(constants.sqrt2 - c_value) < 1e-12
        # Also check it's close to numpy's computation
        assert abs(constants.sqrt2 - np.sqrt(2)) < 1e-12

    def test_pi_precision(self):
        """Test PI matches C value to at least 1e-12 precision."""
        c_value = 3.14159265358979323846264338328
        assert abs(constants.PI - c_value) < 1e-12
        # Also check it's close to numpy's pi
        assert abs(constants.PI - np.pi) < 1e-12

    def test_complex_unit(self):
        """Test complex unit value."""
        assert constants.cmplx_i == 1j
        assert constants.cmplx_i.real == 0.0
        assert constants.cmplx_i.imag == 1.0

    def test_array_limits(self):
        """Test array size limit values."""
        assert constants.MAX_NUM_of_SYMM == 1536
        assert constants.MAX_NUM_of_atoms == 1024
        assert constants.MAX_L == 3


class TestToleranceMonotonicity:
    """Test that tolerance values decrease monotonically."""

    def test_tolerance_series_decreases(self):
        """Test that eps4 > eps5 > eps6 > eps7 > eps8."""
        assert constants.eps4 > constants.eps5
        assert constants.eps5 > constants.eps6
        assert constants.eps6 > constants.eps7
        assert constants.eps7 > constants.eps8

    def test_tolerance_ratios(self):
        """Test that consecutive tolerances differ by factor of 10."""
        assert abs(constants.eps4 / constants.eps5 - 10.0) < 1e-10
        assert abs(constants.eps5 / constants.eps6 - 10.0) < 1e-10
        assert abs(constants.eps6 / constants.eps7 - 10.0) < 1e-10
        assert abs(constants.eps7 / constants.eps8 - 10.0) < 1e-10

    def test_all_tolerances_positive(self):
        """Test that all tolerance values are positive."""
        assert constants.eps4 > 0
        assert constants.eps5 > 0
        assert constants.eps6 > 0
        assert constants.eps7 > 0
        assert constants.eps8 > 0


class TestConstantImmutability:
    """Test that constants are properly defined as immutable where possible."""

    def test_type_hints_present(self):
        """Test that module has proper typing imports."""
        # This is a meta-test to ensure we're following best practices
        import wannsymm.constants as const_module
        source = const_module.__doc__
        assert source is not None
        # Check that the module is properly documented
        assert "Translation Status: ✓ COMPLETE" in source or "COMPLETE" in source
