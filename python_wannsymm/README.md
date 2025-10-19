# WannSymm Python Implementation

Python implementation of WannSymm - A symmetry analysis code for Wannier orbitals.

## Translation Status

This is a one-to-one translation of the original C code to Python.

### Module Translation Progress

| Module | Status | Test Status | Notes |
|--------|--------|-------------|-------|
| constants.py | ⏳ Not Started | ⏳ Not Started | Foundation module |
| vector.py | ⏳ Not Started | ⏳ Not Started | Vector operations |
| matrix.py | ⏳ Not Started | ⏳ Not Started | Matrix utilities |
| wannorb.py | ⏳ Not Started | ⏳ Not Started | Orbital definitions |
| wanndata.py | ⏳ Not Started | ⏳ Not Started | Hamiltonian data |
| usefulio.py | ⏳ Not Started | ⏳ Not Started | I/O utilities |
| readinput.py | ⏳ Not Started | ⏳ Not Started | Input parsing |
| readsymm.py | ⏳ Not Started | ⏳ Not Started | Symmetry reading |
| rotate_orbital.py | ⏳ Not Started | ⏳ Not Started | Orbital rotation |
| rotate_spinor.py | ⏳ Not Started | ⏳ Not Started | Spinor rotation |
| rotate_basis.py | ⏳ Not Started | ⏳ Not Started | Basis rotation |
| rotate_ham.py | ⏳ Not Started | ⏳ Not Started | Hamiltonian rotation |
| bndstruct.py | ⏳ Not Started | ⏳ Not Started | Band structure |
| main.py | ⏳ Not Started | ⏳ Not Started | Main program |

Legend: ⏳ Not Started | 🚧 In Progress | ✅ Complete | ❌ Failed

## Installation (After Translation)

```bash
cd python_wannsymm
pip install -e .
```

## Dependencies

- Python >= 3.7
- numpy
- scipy
- spglib (Python bindings)

Optional:
- mpi4py (for parallel processing)
- pytest (for testing)

## Usage

See the [Original README](../README.md) for usage examples with the C version.
The Python version will have the same interface.

## Translation Approach

Each C module is translated one-to-one to a Python module:
- C structs → Python dataclasses or classes
- C functions → Python functions/methods
- MKL/BLAS → numpy/scipy
- MPI → mpi4py (optional)

See [PYTHON_TRANSLATION_ANALYSIS.md](../PYTHON_TRANSLATION_ANALYSIS.md) for detailed analysis.

## Testing

```bash
pytest tests/
```

## References

Original C implementation: https://github.com/Connor1y/WannSymm
Paper: G.-X. Zhi et al., Computer Physics Communications, 271 (2022) 108196
