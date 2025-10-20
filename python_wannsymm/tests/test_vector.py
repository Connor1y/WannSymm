"""
Tests for vector module

Test vector operations and structures.
"""

import pytest
import numpy as np
from wannsymm.vector import (
    Vector, Symm, VecLList,
    init_vector, array2vector,
    vector_scale, vector_multiply, vector_add, vector_sub,
    dot_product, cross_product, volume_product,
    vector_norm, vector_normalization, vector_round,
    vector_rotate, vector_Rotate,
    equal, equale, vec_comp,
    distance, isenclosed, translate_match, find_vector, kpt_equivalent
)
from wannsymm.constants import eps5, eps6


class TestVectorCreation:
    """Test Vector creation and initialization."""
    
    def test_vector_default_init(self):
        """Test default vector initialization."""
        v = Vector()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0
    
    def test_vector_init_with_values(self):
        """Test vector initialization with values."""
        v = Vector(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
    
    def test_vector_init_integer_conversion(self):
        """Test that integers are converted to floats."""
        v = Vector(1, 2, 3)
        assert isinstance(v.x, float)
        assert isinstance(v.y, float)
        assert isinstance(v.z, float)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
    
    def test_init_vector_function(self):
        """Test init_vector function."""
        v = init_vector(4.0, 5.0, 6.0)
        assert v.x == 4.0
        assert v.y == 5.0
        assert v.z == 6.0
    
    def test_array2vector(self):
        """Test conversion from numpy array to Vector."""
        arr = np.array([1.0, 2.0, 3.0])
        v = array2vector(arr)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
    
    def test_vector_to_array(self):
        """Test conversion from Vector to numpy array."""
        v = Vector(1.0, 2.0, 3.0)
        arr = v.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
    
    def test_vector_from_array(self):
        """Test Vector.from_array classmethod."""
        arr = np.array([7.0, 8.0, 9.0])
        v = Vector.from_array(arr)
        assert v.x == 7.0
        assert v.y == 8.0
        assert v.z == 9.0


class TestVectorArithmetic:
    """Test vector arithmetic operations."""
    
    def test_vector_add_operator(self):
        """Test vector addition using + operator."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        v3 = v1 + v2
        assert v3.x == 5.0
        assert v3.y == 7.0
        assert v3.z == 9.0
    
    def test_vector_sub_operator(self):
        """Test vector subtraction using - operator."""
        v1 = Vector(5.0, 7.0, 9.0)
        v2 = Vector(1.0, 2.0, 3.0)
        v3 = v1 - v2
        assert v3.x == 4.0
        assert v3.y == 5.0
        assert v3.z == 6.0
    
    def test_vector_mul_operator(self):
        """Test vector scalar multiplication using * operator."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = v1 * 2.0
        assert v2.x == 2.0
        assert v2.y == 4.0
        assert v2.z == 6.0
    
    def test_vector_rmul_operator(self):
        """Test vector scalar multiplication (scalar on left)."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = 2.0 * v1
        assert v2.x == 2.0
        assert v2.y == 4.0
        assert v2.z == 6.0
    
    def test_vector_add_function(self):
        """Test vector_add function."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        v3 = vector_add(v1, v2)
        assert v3.x == 5.0
        assert v3.y == 7.0
        assert v3.z == 9.0
    
    def test_vector_sub_function(self):
        """Test vector_sub function."""
        v1 = Vector(5.0, 7.0, 9.0)
        v2 = Vector(1.0, 2.0, 3.0)
        v3 = vector_sub(v1, v2)
        assert v3.x == 4.0
        assert v3.y == 5.0
        assert v3.z == 6.0
    
    def test_vector_scale(self):
        """Test vector_scale function."""
        v = Vector(1.0, 2.0, 3.0)
        v2 = vector_scale(3.0, v)
        assert v2.x == 3.0
        assert v2.y == 6.0
        assert v2.z == 9.0
    
    def test_vector_multiply(self):
        """Test vector_multiply with integer array."""
        v = Vector(1.0, 2.0, 3.0)
        n = np.array([2, 3, 4], dtype=np.int32)
        v2 = vector_multiply(v, n)
        assert v2.x == 2.0
        assert v2.y == 6.0
        assert v2.z == 12.0


class TestVectorProducts:
    """Test vector product operations."""
    
    def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors."""
        v1 = Vector(1.0, 0.0, 0.0)
        v2 = Vector(0.0, 1.0, 0.0)
        result = dot_product(v1, v2)
        assert abs(result) < 1e-10
    
    def test_dot_product_parallel(self):
        """Test dot product of parallel vectors."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(2.0, 4.0, 6.0)
        result = dot_product(v1, v2)
        # v1·v2 = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        assert abs(result - 28.0) < 1e-10
    
    def test_dot_product_general(self):
        """Test dot product of general vectors."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        result = dot_product(v1, v2)
        # v1·v2 = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert abs(result - 32.0) < 1e-10
    
    def test_cross_product_standard_basis(self):
        """Test cross product of standard basis vectors."""
        v1 = Vector(1.0, 0.0, 0.0)
        v2 = Vector(0.0, 1.0, 0.0)
        v3 = cross_product(v1, v2)
        assert abs(v3.x) < 1e-10
        assert abs(v3.y) < 1e-10
        assert abs(v3.z - 1.0) < 1e-10
    
    def test_cross_product_anticommutative(self):
        """Test that cross product is anticommutative."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        v3 = cross_product(v1, v2)
        v4 = cross_product(v2, v1)
        assert abs(v3.x + v4.x) < 1e-10
        assert abs(v3.y + v4.y) < 1e-10
        assert abs(v3.z + v4.z) < 1e-10
    
    def test_cross_product_parallel_is_zero(self):
        """Test that cross product of parallel vectors is zero."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(2.0, 4.0, 6.0)
        v3 = cross_product(v1, v2)
        assert abs(v3.x) < 1e-10
        assert abs(v3.y) < 1e-10
        assert abs(v3.z) < 1e-10
    
    def test_volume_product_unit_cube(self):
        """Test volume product of unit cube vectors."""
        v1 = Vector(1.0, 0.0, 0.0)
        v2 = Vector(0.0, 1.0, 0.0)
        v3 = Vector(0.0, 0.0, 1.0)
        vol = volume_product(v1, v2, v3)
        assert abs(vol - 1.0) < 1e-10
    
    def test_volume_product_coplanar(self):
        """Test volume product of coplanar vectors is zero."""
        v1 = Vector(1.0, 0.0, 0.0)
        v2 = Vector(0.0, 1.0, 0.0)
        v3 = Vector(1.0, 1.0, 0.0)  # In xy-plane
        vol = volume_product(v1, v2, v3)
        assert abs(vol) < 1e-10


class TestVectorNormAndNormalization:
    """Test vector norm and normalization."""
    
    def test_vector_norm_zero(self):
        """Test norm of zero vector."""
        v = Vector(0.0, 0.0, 0.0)
        norm = vector_norm(v)
        assert abs(norm) < 1e-10
    
    def test_vector_norm_unit(self):
        """Test norm of unit vectors."""
        v1 = Vector(1.0, 0.0, 0.0)
        v2 = Vector(0.0, 1.0, 0.0)
        v3 = Vector(0.0, 0.0, 1.0)
        assert abs(vector_norm(v1) - 1.0) < 1e-10
        assert abs(vector_norm(v2) - 1.0) < 1e-10
        assert abs(vector_norm(v3) - 1.0) < 1e-10
    
    def test_vector_norm_pythagorean(self):
        """Test norm using Pythagorean triple."""
        v = Vector(3.0, 4.0, 0.0)
        norm = vector_norm(v)
        assert abs(norm - 5.0) < 1e-10
    
    def test_vector_normalization(self):
        """Test vector normalization to unit length."""
        v = Vector(3.0, 4.0, 0.0)
        v_norm = vector_normalization(v)
        # Check that result has unit length
        assert abs(vector_norm(v_norm) - 1.0) < 1e-10
        # Check that direction is preserved
        assert abs(v_norm.x - 0.6) < 1e-10
        assert abs(v_norm.y - 0.8) < 1e-10
        assert abs(v_norm.z) < 1e-10
    
    def test_vector_normalization_general(self):
        """Test normalization of general vector."""
        v = Vector(1.0, 2.0, 3.0)
        v_norm = vector_normalization(v)
        # Check unit length
        assert abs(vector_norm(v_norm) - 1.0) < 1e-10
        # Check dot product with self gives 1
        assert abs(dot_product(v_norm, v_norm) - 1.0) < 1e-10
    
    def test_vector_round(self):
        """Test vector rounding."""
        v = Vector(1.4, 2.6, 3.5)
        v_round = vector_round(v)
        assert v_round.x == 1.0
        assert v_round.y == 3.0
        assert v_round.z == 4.0  # Note: round() uses banker's rounding in Python
    
    def test_vector_round_negative(self):
        """Test rounding with negative values."""
        v = Vector(-1.4, -2.6, -3.5)
        v_round = vector_round(v)
        assert v_round.x == -1.0
        assert v_round.y == -3.0
        assert v_round.z == -4.0


class TestVectorRotation:
    """Test vector rotation operations."""
    
    def test_vector_rotate_identity(self):
        """Test rotation with identity matrix."""
        v = Vector(1.0, 2.0, 3.0)
        rot = np.eye(3)
        v_rot = vector_rotate(v, rot)
        assert abs(v_rot.x - 1.0) < 1e-10
        assert abs(v_rot.y - 2.0) < 1e-10
        assert abs(v_rot.z - 3.0) < 1e-10
    
    def test_vector_rotate_90deg_z(self):
        """Test 90 degree rotation around z-axis."""
        v = Vector(1.0, 0.0, 0.0)
        # 90 degree rotation around z: x -> y, y -> -x, z -> z
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        v_rot = vector_rotate(v, rot)
        assert abs(v_rot.x) < 1e-10
        assert abs(v_rot.y - 1.0) < 1e-10
        assert abs(v_rot.z) < 1e-10
    
    def test_vector_Rotate_with_vectors(self):
        """Test vector_Rotate with rotation as list of vectors."""
        v = Vector(1.0, 2.0, 3.0)
        # Identity rotation
        rot = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        v_rot = vector_Rotate(v, rot)
        assert abs(v_rot.x - 1.0) < 1e-10
        assert abs(v_rot.y - 2.0) < 1e-10
        assert abs(v_rot.z - 3.0) < 1e-10


class TestVectorComparison:
    """Test vector comparison and equality functions."""
    
    def test_equal_exact(self):
        """Test exact equality."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0, 2.0, 3.0)
        assert equal(v1, v2)
    
    def test_equal_within_tolerance(self):
        """Test equality within eps5 tolerance."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6)
        assert equal(v1, v2)
    
    def test_not_equal(self):
        """Test inequality."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0, 2.0, 4.0)
        assert not equal(v1, v2)
    
    def test_equale_custom_tolerance(self):
        """Test equality with custom tolerance."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.001, 2.001, 3.001)
        assert equale(v1, v2, 0.01)
        assert not equale(v1, v2, 0.0001)
    
    def test_vector_eq_operator(self):
        """Test == operator."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0, 2.0, 3.0)
        v3 = Vector(1.0, 2.0, 4.0)
        assert v1 == v2
        assert not (v1 == v3)
    
    def test_vec_comp_equal(self):
        """Test vec_comp with equal vectors."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7)
        result = vec_comp(v1, v2, eps6)
        assert result == 0
    
    def test_vec_comp_x_greater(self):
        """Test vec_comp when x component differs."""
        v1 = Vector(2.0, 1.0, 1.0)
        v2 = Vector(1.0, 1.0, 1.0)
        result = vec_comp(v1, v2, eps6)
        assert result > 0
    
    def test_vec_comp_y_greater(self):
        """Test vec_comp when only y component differs."""
        v1 = Vector(1.0, 2.0, 1.0)
        v2 = Vector(1.0, 1.0, 1.0)
        result = vec_comp(v1, v2, eps6)
        assert result > 0
    
    def test_vec_comp_z_greater(self):
        """Test vec_comp when only z component differs."""
        v1 = Vector(1.0, 1.0, 2.0)
        v2 = Vector(1.0, 1.0, 1.0)
        result = vec_comp(v1, v2, eps6)
        assert result > 0


class TestDistanceAndGeometry:
    """Test distance and geometric operations."""
    
    def test_distance_zero(self):
        """Test distance between same point."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0, 2.0, 3.0)
        dist = distance(v1, v2)
        assert abs(dist) < 1e-10
    
    def test_distance_pythagorean(self):
        """Test distance using Pythagorean theorem."""
        v1 = Vector(0.0, 0.0, 0.0)
        v2 = Vector(3.0, 4.0, 0.0)
        dist = distance(v1, v2)
        assert abs(dist - 5.0) < 1e-10
    
    def test_distance_general(self):
        """Test distance between general points."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 6.0, 8.0)
        # Distance = sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9+16+25) = sqrt(50)
        dist = distance(v1, v2)
        assert abs(dist - np.sqrt(50)) < 1e-10
    
    def test_isenclosed_true(self):
        """Test isenclosed returns True when enclosed."""
        v1 = Vector(0.5, 0.5, 0.5)
        v2 = Vector(1.0, 1.0, 1.0)
        assert isenclosed(v1, v2)
    
    def test_isenclosed_false(self):
        """Test isenclosed returns False when not enclosed."""
        v1 = Vector(1.5, 0.5, 0.5)
        v2 = Vector(1.0, 1.0, 1.0)
        assert not isenclosed(v1, v2)
    
    def test_isenclosed_negative(self):
        """Test isenclosed with negative components."""
        v1 = Vector(-0.5, -0.5, -0.5)
        v2 = Vector(1.0, 1.0, 1.0)
        assert isenclosed(v1, v2)
    
    def test_translate_match_integer(self):
        """Test translate_match with integer translation."""
        x1 = Vector(2.1, 3.1, 4.1)
        x2 = Vector(0.1, 0.1, 0.1)
        rv, match = translate_match(x1, x2)
        assert match
        assert rv.x == 2.0
        assert rv.y == 3.0
        assert rv.z == 4.0
    
    def test_translate_match_no_match(self):
        """Test translate_match when no match possible."""
        x1 = Vector(1.5, 2.0, 3.0)
        x2 = Vector(0.0, 0.0, 0.0)
        rv, match = translate_match(x1, x2)
        # Should round to (2, 2, 3) but won't match within tolerance
        assert not match
    
    def test_find_vector_found(self):
        """Test find_vector when vector is in list."""
        vlist = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        idx = find_vector(Vector(0, 1, 0), vlist)
        assert idx == 1
    
    def test_find_vector_not_found(self):
        """Test find_vector when vector is not in list."""
        vlist = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        idx = find_vector(Vector(1, 1, 1), vlist)
        assert idx == -1
    
    def test_kpt_equivalent_true(self):
        """Test kpt_equivalent with equivalent k-points."""
        kpt1 = Vector(0.1, 0.2, 0.3)
        kpt2 = Vector(1.1, 2.2, 3.3)
        lattice = np.eye(3)
        assert kpt_equivalent(kpt1, kpt2, lattice)
    
    def test_kpt_equivalent_false(self):
        """Test kpt_equivalent with non-equivalent k-points."""
        kpt1 = Vector(0.1, 0.2, 0.3)
        kpt2 = Vector(0.5, 0.6, 0.7)
        lattice = np.eye(3)
        assert not kpt_equivalent(kpt1, kpt2, lattice)


class TestSymmClass:
    """Test Symm dataclass."""
    
    def test_symm_creation_default(self):
        """Test default Symm creation."""
        s = Symm()
        assert len(s.rot) == 3
        assert isinstance(s.trans, Vector)
    
    def test_symm_creation_with_values(self):
        """Test Symm creation with values."""
        rot = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        trans = Vector(0.5, 0.5, 0.5)
        s = Symm(rot, trans)
        assert len(s.rot) == 3
        assert s.rot[0].x == 1.0
        assert s.trans.x == 0.5
    
    def test_symm_invalid_rotation(self):
        """Test that Symm raises error with wrong number of rotation vectors."""
        with pytest.raises(ValueError):
            Symm(rot=[Vector(1, 0, 0), Vector(0, 1, 0)])  # Only 2 vectors


class TestVecLList:
    """Test VecLList linked list operations."""
    
    def test_vecllist_init(self):
        """Test VecLList initialization."""
        vlist = VecLList()
        assert len(vlist) == 0
    
    def test_vecllist_add(self):
        """Test adding elements to list."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        vlist.add(Vector(0, 1, 0))
        assert len(vlist) == 2
    
    def test_vecllist_add_inorder(self):
        """Test adding elements in order."""
        vlist = VecLList()
        vlist.add_inorder(Vector(2, 0, 0))
        vlist.add_inorder(Vector(1, 0, 0))
        vlist.add_inorder(Vector(3, 0, 0))
        assert vlist[0].x == 1.0
        assert vlist[1].x == 2.0
        assert vlist[2].x == 3.0
    
    def test_vecllist_add_inorder_duplicate(self):
        """Test adding duplicate without force flag."""
        vlist = VecLList()
        result1 = vlist.add_inorder(Vector(1, 0, 0))
        result2 = vlist.add_inorder(Vector(1, 0, 0), flag_force=False)
        assert result1 == 1
        assert result2 == -1
        assert len(vlist) == 1
    
    def test_vecllist_add_inorder_duplicate_forced(self):
        """Test adding duplicate with force flag."""
        vlist = VecLList()
        vlist.add_inorder(Vector(1, 0, 0))
        result = vlist.add_inorder(Vector(1, 0, 0), flag_force=True)
        assert result == 1
        assert len(vlist) == 2
    
    def test_vecllist_pop(self):
        """Test popping element from list."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        vlist.add(Vector(0, 1, 0))
        v, err = vlist.pop()
        assert err == 1
        assert v.x == 1.0
        assert len(vlist) == 1
    
    def test_vecllist_pop_empty(self):
        """Test popping from empty list."""
        vlist = VecLList()
        v, err = vlist.pop()
        assert err == -1
        assert v.x == 0.5  # Default return value
    
    def test_vecllist_delete(self):
        """Test deleting specific element."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        vlist.add(Vector(0, 1, 0))
        v, err = vlist.delete(Vector(1, 0, 0))
        assert err == 1
        assert v.x == 1.0
        assert len(vlist) == 1
    
    def test_vecllist_delete_not_found(self):
        """Test deleting element not in list."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        v, err = vlist.delete(Vector(0, 1, 0))
        assert err == -1
    
    def test_vecllist_find(self):
        """Test finding element in list."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        vlist.add(Vector(0, 1, 0))
        vlist.add(Vector(0, 0, 1))
        idx = vlist.find(Vector(0, 1, 0))
        assert idx == 1
    
    def test_vecllist_find_not_found(self):
        """Test finding element not in list."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        idx = vlist.find(Vector(0, 1, 0))
        assert idx == -1
    
    def test_vecllist_clear(self):
        """Test clearing list."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        vlist.add(Vector(0, 1, 0))
        vlist.clear()
        assert len(vlist) == 0
    
    def test_vecllist_getitem(self):
        """Test accessing elements by index."""
        vlist = VecLList()
        vlist.add(Vector(1, 0, 0))
        vlist.add(Vector(0, 1, 0))
        v = vlist[1]
        assert v.x == 0.0
        assert v.y == 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_vector_norm(self):
        """Test operations with zero vector."""
        v = Vector(0, 0, 0)
        norm = vector_norm(v)
        assert norm == 0.0
    
    def test_very_small_vectors(self):
        """Test with very small vector components."""
        v1 = Vector(1e-10, 1e-10, 1e-10)
        v2 = Vector(2e-10, 2e-10, 2e-10)
        # Should still work correctly
        v3 = v1 + v2
        assert abs(v3.x - 3e-10) < 1e-15
    
    def test_large_magnitude_vectors(self):
        """Test with large magnitude vectors."""
        v1 = Vector(1e10, 1e10, 1e10)
        v2 = Vector(1e10, 1e10, 1e10)
        v3 = v1 + v2
        assert abs(v3.x - 2e10) < 1.0
    
    def test_collinear_vectors(self):
        """Test operations with collinear vectors."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(2, 4, 6)
        # Cross product should be zero
        v_cross = cross_product(v1, v2)
        assert abs(v_cross.x) < 1e-10
        assert abs(v_cross.y) < 1e-10
        assert abs(v_cross.z) < 1e-10
    
    def test_nearly_collinear_vectors(self):
        """Test with nearly collinear vectors."""
        v1 = Vector(1.0, 0.0, 0.0)
        v2 = Vector(1.0, 1e-10, 0.0)
        # Should have very small cross product
        v_cross = cross_product(v1, v2)
        norm = vector_norm(v_cross)
        assert norm < 1e-9
    
    def test_vector_string_representation(self):
        """Test vector __repr__ method."""
        v = Vector(1.0, 2.0, 3.0)
        s = repr(v)
        assert "Vector" in s
        assert "1.0" in s
        assert "2.0" in s
        assert "3.0" in s


class TestAnalyticalCorrectness:
    """Test analytical properties and mathematical correctness."""
    
    def test_dot_product_commutative(self):
        """Test that dot product is commutative."""
        v1 = Vector(1.5, 2.7, 3.9)
        v2 = Vector(4.2, 5.1, 6.8)
        assert abs(dot_product(v1, v2) - dot_product(v2, v1)) < 1e-10
    
    def test_cross_product_perpendicular(self):
        """Test that cross product is perpendicular to both inputs."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        v_cross = cross_product(v1, v2)
        # Cross product should be perpendicular to both
        assert abs(dot_product(v_cross, v1)) < 1e-10
        assert abs(dot_product(v_cross, v2)) < 1e-10
    
    def test_rotation_preserves_norm(self):
        """Test that rotation preserves vector magnitude."""
        v = Vector(1.0, 2.0, 3.0)
        # Create a rotation matrix (90 degrees around z)
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        v_rot = vector_rotate(v, rot)
        assert abs(vector_norm(v) - vector_norm(v_rot)) < 1e-10
    
    def test_normalization_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        v = Vector(3.7, 4.2, 5.9)
        v_norm = vector_normalization(v)
        # Normalized vector should be parallel to original
        # Check by verifying cross product is zero
        v_cross = cross_product(v, v_norm)
        assert vector_norm(v_cross) < 1e-10
    
    def test_distributive_property(self):
        """Test distributive property of dot product."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        v3 = Vector(7.0, 8.0, 9.0)
        # v1·(v2+v3) = v1·v2 + v1·v3
        left = dot_product(v1, v2 + v3)
        right = dot_product(v1, v2) + dot_product(v1, v3)
        assert abs(left - right) < 1e-10
    
    def test_scalar_triple_product_cyclic(self):
        """Test cyclic property of scalar triple product."""
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        v3 = Vector(7.0, 8.0, 10.0)  # Changed to make non-coplanar
        # v1·(v2×v3) = v2·(v3×v1) = v3·(v1×v2)
        vol1 = volume_product(v1, v2, v3)
        vol2 = volume_product(v2, v3, v1)
        vol3 = volume_product(v3, v1, v2)
        assert abs(vol1 - vol2) < 1e-10
        assert abs(vol2 - vol3) < 1e-10
