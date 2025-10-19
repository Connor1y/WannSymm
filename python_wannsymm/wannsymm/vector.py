"""
Vector operations module for WannSymm

This module provides vector operations, vector structures, and linked list utilities.
Translated from: src/vector.h and src/vector.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import numpy as np
import numpy.typing as npt

from .constants import eps5, eps6


@dataclass
class Vector:
    """
    A 3D vector with x, y, z components.
    
    Equivalent to C struct __vector.
    
    Attributes
    ----------
    x : float
        X component of the vector
    y : float
        Y component of the vector
    z : float
        Z component of the vector
    
    Examples
    --------
    >>> v = Vector(1.0, 2.0, 3.0)
    >>> v.x, v.y, v.z
    (1.0, 2.0, 3.0)
    >>> v1 = Vector(1, 0, 0)
    >>> v2 = Vector(0, 1, 0)
    >>> v3 = v1 + v2
    >>> v3.x, v3.y, v3.z
    (1.0, 1.0, 0.0)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __post_init__(self):
        """Convert components to float to ensure numeric type."""
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """
        Convert Vector to numpy array.
        
        Returns
        -------
        np.ndarray
            Array of shape (3,) containing [x, y, z]
        """
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64]) -> Vector:
        """
        Create Vector from numpy array.
        
        Parameters
        ----------
        arr : np.ndarray
            Array of shape (3,) or (3, 1) containing vector components
        
        Returns
        -------
        Vector
            New Vector instance
        """
        arr = np.asarray(arr).flatten()
        return cls(arr[0], arr[1], arr[2])
    
    def __add__(self, other: Vector) -> Vector:
        """
        Add two vectors.
        
        Parameters
        ----------
        other : Vector
            Vector to add
        
        Returns
        -------
        Vector
            Sum of the two vectors
        """
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Vector) -> Vector:
        """
        Subtract another vector from this vector.
        
        Parameters
        ----------
        other : Vector
            Vector to subtract
        
        Returns
        -------
        Vector
            Difference of the two vectors (self - other)
        """
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: Union[float, int]) -> Vector:
        """
        Multiply vector by a scalar.
        
        Parameters
        ----------
        scalar : float or int
            Scalar multiplier
        
        Returns
        -------
        Vector
            Scaled vector
        """
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: Union[float, int]) -> Vector:
        """
        Right multiplication by scalar (scalar * vector).
        
        Parameters
        ----------
        scalar : float or int
            Scalar multiplier
        
        Returns
        -------
        Vector
            Scaled vector
        """
        return self.__mul__(scalar)
    
    def __eq__(self, other: object) -> bool:
        """
        Check exact equality of two vectors.
        
        Parameters
        ----------
        other : Vector
            Vector to compare
        
        Returns
        -------
        bool
            True if vectors are exactly equal
        """
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __repr__(self) -> str:
        """String representation of Vector."""
        return f"Vector({self.x}, {self.y}, {self.z})"


@dataclass
class Symm:
    """
    Symmetry operation consisting of rotation and translation.
    
    Equivalent to C struct __symm.
    
    Attributes
    ----------
    rot : List[Vector]
        List of 3 rotation vectors (rotation matrix rows)
    trans : Vector
        Translation vector
    
    Examples
    --------
    >>> rot = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    >>> trans = Vector(0.5, 0.5, 0.5)
    >>> s = Symm(rot, trans)
    """
    rot: List[Vector] = field(default_factory=lambda: [Vector(), Vector(), Vector()])
    trans: Vector = field(default_factory=Vector)
    
    def __post_init__(self):
        """Ensure rot has exactly 3 vectors."""
        if len(self.rot) != 3:
            raise ValueError("rot must contain exactly 3 vectors")


class VecLList:
    """
    Linked list for Vector objects.
    
    Equivalent to C struct __vec_llist with associated functions.
    This implementation uses a Python list internally for simplicity
    and efficiency, while maintaining the same interface semantics.
    
    Attributes
    ----------
    _data : List[Vector]
        Internal list storage
    
    Examples
    --------
    >>> vlist = VecLList()
    >>> vlist.add(Vector(1, 0, 0))
    >>> vlist.add(Vector(0, 1, 0))
    >>> len(vlist)
    2
    """
    
    def __init__(self):
        """Initialize an empty linked list."""
        self._data: List[Vector] = []
    
    def add(self, val: Vector) -> None:
        """
        Add vector to end of list.
        
        Parameters
        ----------
        val : Vector
            Vector to add
        """
        self._data.append(Vector(val.x, val.y, val.z))
    
    def add_inorder(self, val: Vector, flag_force: bool = False) -> int:
        """
        Add vector to list maintaining sorted order.
        
        Uses vec_comp to determine ordering (x, then y, then z).
        If flag_force is False and val already exists, does not add.
        
        Parameters
        ----------
        val : Vector
            Vector to add
        flag_force : bool, optional
            If True, add even if duplicate exists (default: False)
        
        Returns
        -------
        int
            1 if added successfully, -1 if duplicate and not forced
        """
        # Find insertion position using vec_comp
        idx = 0
        while idx < len(self._data):
            cmp = vec_comp(self._data[idx], val, eps6)
            if cmp >= 0:
                break
            idx += 1
        
        # Check if duplicate exists
        if idx < len(self._data) and vec_comp(self._data[idx], val, eps6) == 0:
            if not flag_force:
                return -1
        
        # Insert at position
        self._data.insert(idx, Vector(val.x, val.y, val.z))
        return 1
    
    def delete(self, val: Vector) -> Tuple[Vector, int]:
        """
        Delete a vector from list and return it.
        
        Parameters
        ----------
        val : Vector
            Vector to delete
        
        Returns
        -------
        Tuple[Vector, int]
            (deleted vector, error code). 
            error code: 1 for success, -1 for not found
        """
        for idx, v in enumerate(self._data):
            if vec_comp(v, val, eps6) == 0:
                deleted = self._data.pop(idx)
                return deleted, 1
        
        # Not found - return a default vector
        return Vector(0.5, 0.5, 0.5), -1
    
    def pop(self) -> Tuple[Vector, int]:
        """
        Pop first element from list.
        
        Returns
        -------
        Tuple[Vector, int]
            (popped vector, error code).
            error code: 1 for success, -1 if list is empty
        """
        if len(self._data) == 0:
            return Vector(0.5, 0.5, 0.5), -1
        
        return self._data.pop(0), 1
    
    def find(self, val: Vector) -> int:
        """
        Find position of vector in list.
        
        Parameters
        ----------
        val : Vector
            Vector to find
        
        Returns
        -------
        int
            Index of vector, or -1 if not found
        """
        for idx, v in enumerate(self._data):
            if vec_comp(v, val, eps6) == 0:
                return idx
        return -1
    
    def clear(self) -> None:
        """Clear all elements from list."""
        self._data.clear()
    
    def __len__(self) -> int:
        """Return number of elements in list."""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Vector:
        """Get element at index."""
        return self._data[idx]


# ============================================================================
# Vector Initialization and Conversion
# ============================================================================

def init_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector:
    """
    Initialize a vector with given components.
    
    Equivalent to C function: void init_vector(vector * v, double x, double y, double z)
    
    Parameters
    ----------
    x : float, optional
        X component (default: 0.0)
    y : float, optional
        Y component (default: 0.0)
    z : float, optional
        Z component (default: 0.0)
    
    Returns
    -------
    Vector
        New Vector instance
    
    Examples
    --------
    >>> v = init_vector(1.0, 2.0, 3.0)
    >>> v.x, v.y, v.z
    (1.0, 2.0, 3.0)
    """
    return Vector(x, y, z)


def array2vector(arr: npt.NDArray[np.float64]) -> Vector:
    """
    Convert numpy array to Vector.
    
    Equivalent to C function: vector array2vector(double * in)
    
    Parameters
    ----------
    arr : np.ndarray
        Array of length 3 containing [x, y, z]
    
    Returns
    -------
    Vector
        New Vector instance
    
    Examples
    --------
    >>> v = array2vector(np.array([1.0, 2.0, 3.0]))
    >>> v.x, v.y, v.z
    (1.0, 2.0, 3.0)
    """
    return Vector.from_array(arr)


# ============================================================================
# Vector Arithmetic Operations
# ============================================================================

def vector_scale(a: float, v: Vector) -> Vector:
    """
    Scale a vector by a scalar.
    
    Equivalent to C function: vector vector_scale(double a, vector v)
    
    Parameters
    ----------
    a : float
        Scalar multiplier
    v : Vector
        Vector to scale
    
    Returns
    -------
    Vector
        Scaled vector (a * v)
    
    Examples
    --------
    >>> v = Vector(1.0, 2.0, 3.0)
    >>> v2 = vector_scale(2.0, v)
    >>> v2.x, v2.y, v2.z
    (2.0, 4.0, 6.0)
    """
    return Vector(a * v.x, a * v.y, a * v.z)


def vector_multiply(v: Vector, n: npt.NDArray[np.int32]) -> Vector:
    """
    Component-wise multiplication of vector with integer array.
    
    Equivalent to C function: vector vector_multiply(vector v1, int * n)
    
    Parameters
    ----------
    v : Vector
        Input vector
    n : np.ndarray
        Integer array of length 3 for component-wise multiplication
    
    Returns
    -------
    Vector
        Result vector with components [v.x*n[0], v.y*n[1], v.z*n[2]]
    
    Examples
    --------
    >>> v = Vector(1.0, 2.0, 3.0)
    >>> n = np.array([2, 3, 4], dtype=np.int32)
    >>> v2 = vector_multiply(v, n)
    >>> v2.x, v2.y, v2.z
    (2.0, 6.0, 12.0)
    """
    n = np.asarray(n, dtype=np.int32)
    return Vector(v.x * n[0], v.y * n[1], v.z * n[2])


def vector_add(v1: Vector, v2: Vector) -> Vector:
    """
    Add two vectors.
    
    Equivalent to C function: vector vector_add(vector v1, vector v2)
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    
    Returns
    -------
    Vector
        Sum v1 + v2
    
    Examples
    --------
    >>> v1 = Vector(1.0, 2.0, 3.0)
    >>> v2 = Vector(4.0, 5.0, 6.0)
    >>> v3 = vector_add(v1, v2)
    >>> v3.x, v3.y, v3.z
    (5.0, 7.0, 9.0)
    """
    return v1 + v2


def vector_sub(v1: Vector, v2: Vector) -> Vector:
    """
    Subtract two vectors.
    
    Equivalent to C function: vector vector_sub(vector v1, vector v2)
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    
    Returns
    -------
    Vector
        Difference v1 - v2
    
    Examples
    --------
    >>> v1 = Vector(5.0, 7.0, 9.0)
    >>> v2 = Vector(1.0, 2.0, 3.0)
    >>> v3 = vector_sub(v1, v2)
    >>> v3.x, v3.y, v3.z
    (4.0, 5.0, 6.0)
    """
    return v1 - v2


# ============================================================================
# Vector Products
# ============================================================================

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Compute dot product of two vectors.
    
    Equivalent to C function: double dot_product(vector v1, vector v2)
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    
    Returns
    -------
    float
        Dot product v1 · v2
    
    Examples
    --------
    >>> v1 = Vector(1.0, 0.0, 0.0)
    >>> v2 = Vector(0.0, 1.0, 0.0)
    >>> dot_product(v1, v2)
    0.0
    >>> v1 = Vector(1.0, 2.0, 3.0)
    >>> v2 = Vector(4.0, 5.0, 6.0)
    >>> dot_product(v1, v2)
    32.0
    """
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def cross_product(v1: Vector, v2: Vector) -> Vector:
    """
    Compute cross product of two vectors.
    
    Equivalent to C function: vector cross_product(vector v1, vector v2)
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    
    Returns
    -------
    Vector
        Cross product v1 × v2
    
    Examples
    --------
    >>> v1 = Vector(1.0, 0.0, 0.0)
    >>> v2 = Vector(0.0, 1.0, 0.0)
    >>> v3 = cross_product(v1, v2)
    >>> v3.x, v3.y, v3.z
    (0.0, 0.0, 1.0)
    """
    return Vector(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )


def volume_product(v1: Vector, v2: Vector, v3: Vector) -> float:
    """
    Compute scalar triple product (volume product) of three vectors.
    
    Equivalent to C function: double volume_product(vector v1, vector v2, vector v3)
    
    Returns v1 · (v2 × v3), which equals the volume of parallelepiped
    formed by the three vectors.
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    v3 : Vector
        Third vector
    
    Returns
    -------
    float
        Scalar triple product v1 · (v2 × v3)
    
    Examples
    --------
    >>> v1 = Vector(1.0, 0.0, 0.0)
    >>> v2 = Vector(0.0, 1.0, 0.0)
    >>> v3 = Vector(0.0, 0.0, 1.0)
    >>> volume_product(v1, v2, v3)
    1.0
    """
    vol = (v1.x * v2.y * v3.z + v1.y * v2.z * v3.x + v1.z * v2.x * v3.y -
           (v1.x * v2.z * v3.y + v1.y * v2.x * v3.z + v1.z * v2.y * v3.x))
    return vol


# ============================================================================
# Vector Norms and Normalization
# ============================================================================

def vector_norm(v: Vector) -> float:
    """
    Compute Euclidean norm (magnitude) of a vector.
    
    Equivalent to C function: double vector_norm(vector v)
    
    Parameters
    ----------
    v : Vector
        Input vector
    
    Returns
    -------
    float
        Euclidean norm ||v|| = sqrt(v·v)
    
    Examples
    --------
    >>> v = Vector(3.0, 4.0, 0.0)
    >>> vector_norm(v)
    5.0
    """
    return np.sqrt(dot_product(v, v))


def vector_normalization(v: Vector) -> Vector:
    """
    Normalize a vector to unit length.
    
    Equivalent to C function: vector vector_normalization(vector v)
    
    Parameters
    ----------
    v : Vector
        Input vector
    
    Returns
    -------
    Vector
        Unit vector in direction of v
    
    Examples
    --------
    >>> v = Vector(3.0, 4.0, 0.0)
    >>> v_norm = vector_normalization(v)
    >>> abs(vector_norm(v_norm) - 1.0) < 1e-10
    True
    """
    norm = vector_norm(v)
    return vector_scale(1.0 / norm, v)


def vector_round(v: Vector) -> Vector:
    """
    Round each component of vector to nearest integer.
    
    Equivalent to C function: vector vector_round(vector v)
    
    Parameters
    ----------
    v : Vector
        Input vector
    
    Returns
    -------
    Vector
        Vector with rounded components
    
    Examples
    --------
    >>> v = Vector(1.4, 2.6, 3.5)
    >>> v_round = vector_round(v)
    >>> v_round.x, v_round.y, v_round.z
    (1.0, 3.0, 4.0)
    """
    return Vector(round(v.x), round(v.y), round(v.z))


# ============================================================================
# Vector Rotation and Transformation
# ============================================================================

def matrix_dot(Tmat: List[Vector], v: Vector) -> Vector:
    """
    Multiply vector by matrix (v * T).
    
    Helper function used in distance calculation.
    Matrix is represented as list of 3 vectors (rows).
    
    Parameters
    ----------
    Tmat : List[Vector]
        Matrix as list of 3 row vectors
    v : Vector
        Vector to multiply
    
    Returns
    -------
    Vector
        Result of v * T
    """
    return Vector(
        v.x * Tmat[0].x + v.y * Tmat[1].x + v.z * Tmat[2].x,
        v.x * Tmat[0].y + v.y * Tmat[1].y + v.z * Tmat[2].y,
        v.x * Tmat[0].z + v.y * Tmat[1].z + v.z * Tmat[2].z
    )


def vector_rotate(v_in: Vector, rotation: npt.NDArray[np.float64]) -> Vector:
    """
    Rotate vector using rotation matrix.
    
    Equivalent to C function: vector vector_rotate(vector in, double rotation[3][3])
    
    Parameters
    ----------
    v_in : Vector
        Input vector
    rotation : np.ndarray
        3x3 rotation matrix
    
    Returns
    -------
    Vector
        Rotated vector
    
    Examples
    --------
    >>> v = Vector(1.0, 0.0, 0.0)
    >>> # 90 degree rotation around z-axis
    >>> rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    >>> v_rot = vector_rotate(v, rot)
    >>> abs(v_rot.x) < 1e-10 and abs(v_rot.y - 1.0) < 1e-10
    True
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    
    # Convert rotation matrix rows to vectors
    symm = [Vector(rotation[i, 0], rotation[i, 1], rotation[i, 2]) for i in range(3)]
    
    out = Vector(
        dot_product(symm[0], v_in),
        dot_product(symm[1], v_in),
        dot_product(symm[2], v_in)
    )
    
    return out


def vector_Rotate(v_in: Vector, Rot: List[Vector]) -> Vector:
    """
    Rotate vector using rotation matrix represented as list of vectors.
    
    Equivalent to C function: vector vector_Rotate(vector in, vector * Rot)
    
    Parameters
    ----------
    v_in : Vector
        Input vector
    Rot : List[Vector]
        List of 3 vectors representing rotation matrix rows
    
    Returns
    -------
    Vector
        Rotated vector
    
    Examples
    --------
    >>> v = Vector(1.0, 2.0, 3.0)
    >>> # Identity rotation
    >>> rot = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    >>> v_rot = vector_Rotate(v, rot)
    >>> v_rot.x, v_rot.y, v_rot.z
    (1.0, 2.0, 3.0)
    """
    out = Vector(
        dot_product(Rot[0], v_in),
        dot_product(Rot[1], v_in),
        dot_product(Rot[2], v_in)
    )
    return out


# ============================================================================
# Vector Comparison and Equality
# ============================================================================

def equal(v1: Vector, v2: Vector) -> bool:
    """
    Check if two vectors are equal within default tolerance (eps5).
    
    Equivalent to C function: int equal(vector v1, vector v2)
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    
    Returns
    -------
    bool
        True if all components differ by less than eps5
    
    Examples
    --------
    >>> v1 = Vector(1.0, 2.0, 3.0)
    >>> v2 = Vector(1.0 + 1e-6, 2.0, 3.0)
    >>> equal(v1, v2)
    True
    """
    return (abs(v1.x - v2.x) < eps5 and
            abs(v1.y - v2.y) < eps5 and
            abs(v1.z - v2.z) < eps5)


def equale(v1: Vector, v2: Vector, epsdiff: float) -> bool:
    """
    Check if two vectors are equal within specified tolerance.
    
    Equivalent to C function: int equale(vector v1, vector v2, double epsdiff)
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    epsdiff : float
        Tolerance for comparison
    
    Returns
    -------
    bool
        True if all components differ by less than epsdiff
    
    Examples
    --------
    >>> v1 = Vector(1.0, 2.0, 3.0)
    >>> v2 = Vector(1.001, 2.001, 3.001)
    >>> equale(v1, v2, 0.01)
    True
    >>> equale(v1, v2, 0.0001)
    False
    """
    return (abs(v1.x - v2.x) < epsdiff and
            abs(v1.y - v2.y) < epsdiff and
            abs(v1.z - v2.z) < epsdiff)


def vec_comp(v1: Vector, v2: Vector, tolerance: float) -> int:
    """
    Compare two vectors for ordering.
    
    Equivalent to C function: int vec_comp(vector v1, vector v2, double tolerance)
    
    Compares vectors component-wise (x, then y, then z).
    Returns positive if v1 > v2, negative if v1 < v2, zero if equal.
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    tolerance : float
        Tolerance for equality comparison
    
    Returns
    -------
    int
        Comparison result:
        - positive integer if v1 > v2
        - negative integer if v1 < v2
        - 0 if vectors are equal within tolerance
    
    Examples
    --------
    >>> v1 = Vector(1.0, 2.0, 3.0)
    >>> v2 = Vector(1.0, 2.0, 4.0)
    >>> vec_comp(v1, v2, 1e-6) < 0
    True
    """
    if abs(v1.x - v2.x) > tolerance:
        return int((v1.x - v2.x) / tolerance)
    elif abs(v1.y - v2.y) > tolerance:
        return int((v1.y - v2.y) / tolerance)
    elif abs(v1.z - v2.z) > tolerance:
        return int((v1.z - v2.z) / tolerance)
    else:
        return 0


# ============================================================================
# Distance and Geometric Operations
# ============================================================================

def distance(v1: Vector, v2: Vector, Tmat: Optional[List[Vector]] = None) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Equivalent to C function: double distance(vector v1, vector v2, vector * Tmat)
    
    If Tmat is provided, transforms vectors by matrix before computing distance.
    
    Parameters
    ----------
    v1 : Vector
        First vector
    v2 : Vector
        Second vector
    Tmat : List[Vector], optional
        Transformation matrix (3 row vectors), default None
    
    Returns
    -------
    float
        Euclidean distance ||v1 - v2||
    
    Examples
    --------
    >>> v1 = Vector(0.0, 0.0, 0.0)
    >>> v2 = Vector(3.0, 4.0, 0.0)
    >>> distance(v1, v2)
    5.0
    """
    if Tmat is not None:
        x1 = matrix_dot(Tmat, v1)
        x2 = matrix_dot(Tmat, v2)
    else:
        x1 = v1
        x2 = v2
    
    r = np.sqrt((x1.x - x2.x)**2 + (x1.y - x2.y)**2 + (x1.z - x2.z)**2)
    return r


def isenclosed(v1: Vector, v2: Vector) -> bool:
    """
    Check if v1 is enclosed within box defined by v2.
    
    Equivalent to C function: int isenclosed(vector v1, vector v2)
    
    Tests if |v1.x| <= v2.x and |v1.y| <= v2.y and |v1.z| <= v2.z
    
    Parameters
    ----------
    v1 : Vector
        Vector to test
    v2 : Vector
        Box dimensions
    
    Returns
    -------
    bool
        True if v1 is enclosed within box
    
    Examples
    --------
    >>> v1 = Vector(0.5, 0.5, 0.5)
    >>> v2 = Vector(1.0, 1.0, 1.0)
    >>> isenclosed(v1, v2)
    True
    """
    return (abs(v1.x) <= v2.x and
            abs(v1.y) <= v2.y and
            abs(v1.z) <= v2.z)


def translate_match(x1: Vector, x2: Vector) -> Tuple[Vector, bool]:
    """
    Find translation vector to match x1 to x2 within a lattice.
    
    Equivalent to C function: int translate_match(vector * rv, vector x1, vector x2)
    
    Computes integer translation rv such that x2 + rv ≈ x1 within eps5 tolerance.
    
    Parameters
    ----------
    x1 : Vector
        Target vector
    x2 : Vector
        Source vector
    
    Returns
    -------
    Tuple[Vector, bool]
        (translation vector, match_found)
        - translation vector: integer vector rv
        - match_found: True if translation brings x2 within eps5 of x1
    
    Examples
    --------
    >>> x1 = Vector(2.1, 3.1, 4.1)
    >>> x2 = Vector(0.1, 0.1, 0.1)
    >>> rv, match = translate_match(x1, x2)
    >>> match
    True
    >>> rv.x, rv.y, rv.z
    (2.0, 3.0, 4.0)
    """
    rv = Vector(
        int(round(x1.x - x2.x)),
        int(round(x1.y - x2.y)),
        int(round(x1.z - x2.z))
    )
    
    tmp = vector_add(x2, rv)
    match_found = distance(x1, tmp) < eps5
    
    return rv, match_found


def find_vector(v: Vector, vlist: List[Vector]) -> int:
    """
    Find vector in list using equality comparison.
    
    Equivalent to C function: int find_vector(vector v, vector * list, int nlist)
    
    Parameters
    ----------
    v : Vector
        Vector to find
    vlist : List[Vector]
        List of vectors to search
    
    Returns
    -------
    int
        Index of vector in list, or -1 if not found
    
    Examples
    --------
    >>> vlist = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    >>> find_vector(Vector(0, 1, 0), vlist)
    1
    >>> find_vector(Vector(1, 1, 1), vlist)
    -1
    """
    for i, vec in enumerate(vlist):
        if equal(v, vec):
            return i
    return -1


def kpt_equivalent(kpt1: Vector, kpt2: Vector, lattice: npt.NDArray[np.float64]) -> bool:
    """
    Check if two k-points are equivalent within lattice translations.
    
    Equivalent to C function: int kpt_equivalent(vector kpt1, vector kpt2, double lattice[3][3])
    
    Two k-points are equivalent if their difference is an integer vector
    (within tolerance eps5).
    
    Parameters
    ----------
    kpt1 : Vector
        First k-point
    kpt2 : Vector
        Second k-point
    lattice : np.ndarray
        3x3 lattice matrix (unused in current implementation)
    
    Returns
    -------
    bool
        True if k-points are equivalent
    
    Examples
    --------
    >>> kpt1 = Vector(0.1, 0.2, 0.3)
    >>> kpt2 = Vector(1.1, 2.2, 3.3)
    >>> lattice = np.eye(3)
    >>> kpt_equivalent(kpt1, kpt2, lattice)
    True
    """
    dk = vector_sub(kpt1, kpt2)
    
    if abs(dk.x - round(dk.x)) > eps5:
        return False
    if abs(dk.y - round(dk.y)) > eps5:
        return False
    if abs(dk.z - round(dk.z)) > eps5:
        return False
    
    return True
