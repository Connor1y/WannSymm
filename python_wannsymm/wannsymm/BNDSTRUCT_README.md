# Band Structure Module (bndstruct.py)

## Overview

The `bndstruct` module provides functions for calculating electronic band structures from Wannier Hamiltonians. It implements Fourier transforms from real-space H(R) to reciprocal-space H(k), band interpolation, and utilities for analyzing symmetry properties.

## Features

- **Fourier Transform**: Convert real-space Hamiltonian H(R) to reciprocal-space H(k)
- **Band Diagonalization**: Calculate eigenvalues (band energies) and eigenvectors
- **Degeneracy Analysis**: Automatically identify degenerate energy levels
- **File Output**: Write band structures, characters, and eigenvalues to files
- **SmallGroup**: Track symmetry operations that preserve k-points

## Key Functions

### `diagonalize_hamiltonian(hr, kpt)`

Calculates eigenvalues and eigenvectors of the k-space Hamiltonian.

**Parameters:**
- `hr` (WannData): Real-space Hamiltonian data
- `kpt` (Vector): k-point in crystal coordinates

**Returns:**
- `eig_hk` (ndarray): Eigenvalues (energies) at k-point
- `vr_hk` (ndarray): Right eigenvectors

**Example:**
```python
from wannsymm.bndstruct import diagonalize_hamiltonian
from wannsymm.wanndata import WannData
from wannsymm.vector import Vector

hr = WannData(norb=2, nrpt=1)
# ... initialize Hamiltonian ...

kpt = Vector(0, 0, 0)  # Gamma point
eig, vec = diagonalize_hamiltonian(hr, kpt)
```

### `identify_degeneracies(eig_hk, degenerate_tolerance=1e-6)`

Determines degeneracy of each eigenstate.

**Parameters:**
- `eig_hk` (ndarray): Eigenvalues (must be sorted)
- `degenerate_tolerance` (float): Energy threshold for degeneracy

**Returns:**
- `ndegen` (ndarray): Degeneracy count for each state

**Example:**
```python
from wannsymm.bndstruct import identify_degeneracies
import numpy as np

eig = np.array([1.0, 1.0, 2.0, 3.0])
ndegen = identify_degeneracies(eig)
# Result: [2, 2, 1, 1] - first two states are degenerate
```

### `write_bands(fbands, norb, nkpath, nk_per_kpath, lattice, kvecs, ebands)`

Writes band structure data to file with cumulative k-path length.

**Parameters:**
- `fbands` (TextIO): Open file object
- `norb` (int): Number of orbitals
- `nkpath` (int): Number of k-path segments
- `nk_per_kpath` (int): Number of k-points per segment
- `lattice` (ndarray): Lattice vectors (3x3)
- `kvecs` (list): k-point coordinates
- `ebands` (ndarray): Band energies (nk x norb)

**Example:**
```python
from wannsymm.bndstruct import write_bands

with open('bands.dat', 'w') as f:
    write_bands(f, norb=2, nkpath=1, nk_per_kpath=10,
                lattice=np.eye(3), kvecs=kvecs, ebands=ebands)
```

## SmallGroup Data Structure

Represents the small group of symmetry operations that leave a k-point invariant.

**Attributes:**
- `order` (int): Number of symmetry operations in the group
- `element` (list): Indices of group elements

**Example:**
```python
from wannsymm.bndstruct import SmallGroup

sgrp = SmallGroup(order=2, element=[0, 3])
# This k-point is invariant under symmetry operations 0 and 3
```

## Fourier Transform

The Fourier transform from real space to reciprocal space is:

```
H(k) = Σ_R exp(i k·R) H(R) / weight(R)
```

Where:
- H(R) is the real-space Hamiltonian
- k is the crystal momentum
- R is a lattice vector
- weight(R) is the degeneracy weight of the R-point

## Testing

The module includes comprehensive tests in `tests/test_bndstruct.py`:

- Simple tight-binding model tests
- Fourier transform accuracy tests
- Degeneracy identification tests
- File I/O tests
- Integration tests

Run tests with:
```bash
pytest tests/test_bndstruct.py -v
```

## Implementation Notes

1. **Hermiticity**: The Hamiltonian H(k) is guaranteed to be Hermitian if H(R) satisfies the proper symmetry relations.

2. **Degeneracy Tolerance**: The default tolerance of 1e-6 eV is suitable for most purposes. Adjust if needed for high-precision calculations.

3. **Memory Efficiency**: Large Hamiltonians are handled efficiently using NumPy's vectorized operations.

4. **Eigenvector Convention**: Eigenvectors are returned in row-major format, matching the C implementation convention.

## Related Modules

- `wanndata`: Wannier Hamiltonian data structures
- `vector`: Vector operations and k-point utilities
- `matrix`: Linear algebra utilities
- `rotate_ham`: Hamiltonian symmetrization (for character analysis)

## References

- Wannier90 User Guide: http://www.wannier.org/
- Wannier interpolation: Marzari, Mostofi et al., Rev. Mod. Phys. 84, 1419 (2012)
