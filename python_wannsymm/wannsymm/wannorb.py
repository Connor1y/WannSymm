"""
Wannier orbital module for WannSymm

Defines Wannier orbital structures and operations.
Translated from: src/wannorb.h and src/wannorb.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .vector import Vector


@dataclass
class WannOrb:
    """
    Wannier orbital structure.

    Equivalent to C struct __wannorb.

    Attributes
    ----------
    site : Vector
        Orbital site position
    axis : List[Vector]
        List of 3 axis vectors [axisx, axisy, axisz]
    r : int
        Main quantum number (radial quantum number, not used in symmetry)
    l : int
        Angular momentum quantum number:
        - 0: s orbital
        - 1: p orbital
        - 2: d orbital
        - 3: f orbital
    mr : int
        Indicator of cubic harmonic orbital function
    ms : int
        Spin quantum number:
        - 0: spin up
        - 1: spin down

    Examples
    --------
    >>> from wannsymm.vector import Vector
    >>> # Create an s orbital at origin with spin up
    >>> site = Vector(0.0, 0.0, 0.0)
    >>> axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    >>> orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
    >>> orb.l
    0
    >>> orb.ms
    0
    """

    site: Vector
    axis: List[Vector]
    r: int
    l: int
    mr: int
    ms: int

    def __post_init__(self):
        """
        Validate quantum numbers and spin values after initialization.

        Raises
        ------
        ValueError
            If quantum numbers or spin values are invalid
        """
        # Validate r (radial quantum number must be non-negative)
        if self.r < 0:
            raise ValueError(
                f"Radial quantum number r must be non-negative, got {self.r}"
            )

        # Validate l (angular momentum: 0=s, 1=p, 2=d, 3=f)
        if self.l < 0 or self.l > 3:
            raise ValueError(
                f"Angular momentum l must be 0 (s), 1 (p), 2 (d), "
                f"or 3 (f), got {self.l}"
            )

        # Validate ms (spin quantum number: 0=up, 1=down)
        if self.ms not in (0, 1):
            raise ValueError(
                f"Spin quantum number ms must be 0 (up) or 1 (down), "
                f"got {self.ms}"
            )

        # Validate axis list has exactly 3 vectors
        if len(self.axis) != 3:
            raise ValueError(
                f"axis must contain exactly 3 vectors, got {len(self.axis)}"
            )

        # Ensure all axis elements are Vector instances
        for i, ax in enumerate(self.axis):
            if not isinstance(ax, Vector):
                raise TypeError(
                    f"axis[{i}] must be a Vector instance, got {type(ax)}"
                )

    def __repr__(self) -> str:
        """
        String representation of WannOrb.

        Returns
        -------
        str
            String representation showing key attributes

        Examples
        --------
        >>> from wannsymm.vector import Vector
        >>> site = Vector(0.0, 0.0, 0.0)
        >>> axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        >>> orb = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        >>> repr(orb)  # doctest: +ELLIPSIS
        'WannOrb(site=Vector(0.0, 0.0, 0.0), r=1, l=0, mr=0, ms=0)'
        """
        orbital_type = {0: "s", 1: "p", 2: "d", 3: "f"}.get(self.l, "?")
        spin_type = "up" if self.ms == 0 else "down"
        return (
            f"WannOrb(site={self.site!r}, r={self.r}, "
            f"l={self.l}({orbital_type}), "
            f"mr={self.mr}, ms={self.ms}({spin_type}))"
        )

    def __eq__(self, other: object) -> bool:
        """
        Check equality of two WannOrb instances.

        Parameters
        ----------
        other : object
            Object to compare with

        Returns
        -------
        bool
            True if all attributes match

        Examples
        --------
        >>> from wannsymm.vector import Vector
        >>> site = Vector(0.0, 0.0, 0.0)
        >>> axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        >>> orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        >>> orb2 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        >>> orb1 == orb2
        True
        """
        if not isinstance(other, WannOrb):
            return False

        # Check scalar attributes
        if (
            self.r != other.r
            or self.l != other.l
            or self.mr != other.mr
            or self.ms != other.ms
        ):
            return False

        # Check site vector
        if self.site != other.site:
            return False

        # Check axis vectors
        if len(self.axis) != len(other.axis):
            return False

        for ax1, ax2 in zip(self.axis, other.axis):
            if ax1 != ax2:
                return False

        return True


def init_wannorb(
    site: Vector,
    l: int,
    mr: int,
    ms: int,
    r: int,
    axisz: Vector,
    axisx: Vector,
    axisy: Vector,
) -> WannOrb:
    """
    Initialize a Wannier orbital with given parameters.

    Equivalent to C function: void init_wannorb(wannorb * orb, vector site,
                                                 int l, int mr, int ms, int r,
                                                 vector axisz, vector axisx,
                                                 vector axisy)

    Parameters
    ----------
    site : Vector
        Orbital site position
    l : int
        Angular momentum quantum number (0=s, 1=p, 2=d, 3=f)
    mr : int
        Cubic harmonic orbital function indicator
    ms : int
        Spin quantum number (0=up, 1=down)
    r : int
        Main quantum number (radial)
    axisz : Vector
        Z-axis direction
    axisx : Vector
        X-axis direction
    axisy : Vector
        Y-axis direction

    Returns
    -------
    WannOrb
        New WannOrb instance

    Examples
    --------
    >>> from wannsymm.vector import Vector
    >>> site = Vector(0.0, 0.0, 0.0)
    >>> axisx = Vector(1.0, 0.0, 0.0)
    >>> axisy = Vector(0.0, 1.0, 0.0)
    >>> axisz = Vector(0.0, 0.0, 1.0)
    >>> orb = init_wannorb(site, l=1, mr=0, ms=0, r=2, axisz=axisz,
    ...                    axisx=axisx, axisy=axisy)
    >>> orb.l
    1
    >>> orb.r
    2
    """
    # Note: C code stores as [axisx, axisy, axisz] with indices [0, 1, 2]
    axis = [axisx, axisy, axisz]
    return WannOrb(site=site, axis=axis, r=r, l=l, mr=mr, ms=ms)


def find_index_of_wannorb(
    wann: List[WannOrb], site: Vector, r: int, l: int, mr: int, ms: int
) -> int:
    """
    Find index of Wannier orbital in array by properties.

    Equivalent to C function: int find_index_of_wannorb(wannorb * wann,
                                                         int num_wann,
                                                         vector site, int r,
                                                         int l, int mr, int ms)

    Note: Axis is ignored in version 0.01 as per original C code comment.

    Parameters
    ----------
    wann : List[WannOrb]
        List of Wannier orbitals to search
    site : Vector
        Orbital site position
    r : int
        Main quantum number
    l : int
        Angular momentum quantum number
    mr : int
        Cubic harmonic orbital function indicator
    ms : int
        Spin quantum number

    Returns
    -------
    int
        Index of matching orbital, or -1 if not found

    Examples
    --------
    >>> from wannsymm.vector import Vector, equal
    >>> site1 = Vector(0.0, 0.0, 0.0)
    >>> site2 = Vector(1.0, 0.0, 0.0)
    >>> axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    >>> orb1 = WannOrb(site=site1, axis=axis, r=1, l=0, mr=0, ms=0)
    >>> orb2 = WannOrb(site=site2, axis=axis, r=1, l=1, mr=0, ms=0)
    >>> wann_list = [orb1, orb2]
    >>> find_index_of_wannorb(wann_list, site1, r=1, l=0, mr=0, ms=0)
    0
    >>> find_index_of_wannorb(wann_list, site2, r=1, l=1, mr=0, ms=0)
    1
    >>> find_index_of_wannorb(wann_list, site1, r=2, l=0, mr=0, ms=0)
    -1
    """
    from .vector import equal

    for i, orb in enumerate(wann):
        # Match quantum numbers in order: ms, mr, l, r (as in C code)
        if orb.ms == ms and orb.mr == mr and orb.l == l and orb.r == r:
            # Check if site matches
            if equal(orb.site, site):
                return i

    return -1
