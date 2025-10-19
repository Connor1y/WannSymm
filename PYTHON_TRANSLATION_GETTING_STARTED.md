# Python Translation - Getting Started

This document provides a roadmap for translating the WannSymm C project to Python.

## 📁 Project Structure Created

```
python_wannsymm/
├── wannsymm/                 # Python package
│   ├── __init__.py          # Package initialization
│   ├── constants.py         # Constants (TODO: translate)
│   ├── vector.py            # Vector operations (TODO: translate)
│   ├── matrix.py            # Matrix operations (TODO: translate)
│   ├── wannorb.py           # Orbital structures (TODO: translate)
│   ├── wanndata.py          # Hamiltonian data (TODO: translate)
│   ├── usefulio.py          # I/O utilities (TODO: translate)
│   ├── readinput.py         # Input parsing (TODO: translate)
│   ├── readsymm.py          # Symmetry reading (TODO: translate)
│   ├── rotate_orbital.py    # Orbital rotation (TODO: translate)
│   ├── rotate_spinor.py     # Spinor rotation (TODO: translate)
│   ├── rotate_basis.py      # Basis rotation (TODO: translate)
│   ├── rotate_ham.py        # Hamiltonian rotation (TODO: translate)
│   ├── bndstruct.py         # Band structure (TODO: translate)
│   └── main.py              # Main program (TODO: translate)
├── tests/                    # Test suite
│   ├── test_*.py            # Tests for each module
│   └── test_integration.py  # Integration tests
├── docs/                     # Documentation
│   ├── TRANSLATION_WORKFLOW.md    # Detailed translation instructions
│   ├── DEPENDENCY_GRAPH.md        # Module dependencies
│   └── QUICKSTART.md              # Quick start guide
├── README.md                 # Project README
├── pyproject.toml           # Python package configuration
├── requirements.txt         # Dependencies
└── .gitignore               # Git ignore rules
```

## 📚 Documentation Guide

### 1. **PYTHON_TRANSLATION_ANALYSIS.md** (Repository root)
   - Comprehensive analysis of the C codebase
   - Module complexity and effort estimates
   - Dependency relationships
   - Translation challenges and solutions
   - **Start here** to understand the big picture

### 2. **python_wannsymm/docs/TRANSLATION_WORKFLOW.md**
   - **Most important document for translators**
   - Step-by-step translation instructions for each module
   - Detailed prompts with code examples
   - Testing strategies
   - Week-by-week execution plan
   - **Use this as your primary guide**

### 3. **python_wannsymm/docs/DEPENDENCY_GRAPH.md**
   - Visual dependency graph
   - Translation order (bottom-up)
   - Parallelization opportunities
   - Success checkpoints
   - **Use this to understand module dependencies**

### 4. **python_wannsymm/docs/QUICKSTART.md**
   - Development environment setup
   - Translation workflow for each module
   - Code examples and best practices
   - Common C to Python mappings
   - Debugging tips
   - **Use this for day-to-day translation work**

## 🎯 Translation Order

Work through modules in this order:

### Phase 1: Foundation (Week 1)
1. ✅ **constants.py** - Define all constants (~0.5 hours)
2. ✅ **vector.py** - Vector operations (~4-6 hours)
3. ✅ **matrix.py** - Matrix utilities (~2-3 hours)
4. ✅ **wannorb.py** - Orbital definitions (~2-3 hours)
5. ✅ **wanndata.py** - Hamiltonian data (~6-8 hours)

### Phase 2: I/O (Week 2)
6. ✅ **usefulio.py** - I/O utilities (~2-3 hours)
7. ⚠️ **readinput.py** - Input parsing (~20-30 hours) ← **LARGEST MODULE**
8. ✅ **readsymm.py** - Symmetry reading (~3-4 hours)

### Phase 3: Rotation Operations (Weeks 3-4)
9. ⚠️ **rotate_orbital.py** - Orbital rotation (~10-15 hours) ← **COMPLEX PHYSICS**
10. ✅ **rotate_spinor.py** - Spinor rotation (~4-6 hours)
11. ✅ **rotate_basis.py** - Basis rotation (~6-8 hours)
12. ⚠️ **rotate_ham.py** - Hamiltonian rotation (~15-20 hours) ← **CORE ALGORITHM**

### Phase 4: Analysis & Integration (Week 5)
13. ✅ **bndstruct.py** - Band structure (~10-12 hours)
14. ✅ **main.py** - Main program (~8-10 hours)

**Total Estimated Time**: 93-138 hours (12-18 working days)

## 🚀 Quick Start

### For Your First Module (constants.py)

1. **Read the C code**:
   ```bash
   cat src/constants.h
   ```

2. **Read the translation prompt**:
   ```bash
   cat python_wannsymm/docs/TRANSLATION_WORKFLOW.md
   # Find the "Module 1: constants.py" section
   ```

3. **Implement the module**:
   ```bash
   # Edit python_wannsymm/wannsymm/constants.py
   # Remove TODO comments and implement
   ```

4. **Write tests**:
   ```bash
   # Edit python_wannsymm/tests/test_constants.py
   # Add comprehensive tests
   ```

5. **Run tests**:
   ```bash
   cd python_wannsymm
   pytest tests/test_constants.py -v
   ```

6. **Create PR for review**

## 📋 Workflow for Each Module

1. **Analyze** - Read C source code
2. **Plan** - Review translation prompt in TRANSLATION_WORKFLOW.md
3. **Implement** - Write Python code with type hints and docstrings
4. **Test** - Write comprehensive tests
5. **Validate** - Run tests, check code quality
6. **Review** - Create PR, get feedback
7. **Iterate** - Fix issues, update based on review

## 🎓 Key Concepts to Understand

### For All Modules:
- NumPy array operations
- Python type hints
- Docstring formatting (NumPy style)

### For Rotation Modules:
- Wigner D-matrices (angular momentum)
- Spherical and cubic harmonics
- Spinor rotations (SU(2) group)
- Symmetry operations in solid-state physics

### For Main Algorithm (rotate_ham.py):
- Real-space Hamiltonian representation
- Fourier transforms between R-space and k-space
- Symmetrization by averaging
- Time-reversal symmetry

## 📖 Additional Resources

- **Original Paper**: G.-X. Zhi et al., Computer Physics Communications, 271 (2022) 108196
- **Wannier90**: http://www.wannier.org/
- **spglib**: https://spglib.github.io/spglib/
- **NumPy**: https://numpy.org/doc/
- **SciPy**: https://docs.scipy.org/

## ✅ Success Criteria

For each module:
- [ ] All C functions translated
- [ ] Type hints on all functions
- [ ] NumPy-style docstrings
- [ ] Unit tests with >80% coverage
- [ ] All tests passing
- [ ] Code passes flake8 and mypy
- [ ] Formatted with black

## 🤝 Collaboration

### One Module per PR
Each module (or small group of related modules) should be in its own PR with:
- Module implementation
- Comprehensive tests
- Updated README.md status table

### PR Template
```markdown
## Module: [module_name]

### C Source Files
- src/[filename].h (X lines)
- src/[filename].c (Y lines)

### Changes
- Implemented [list of functions]
- Added tests for [list of test cases]
- [Any deviations from C code]

### Testing
- [ ] All tests passing
- [ ] Code coverage >80%
- [ ] flake8 clean
- [ ] mypy clean
- [ ] black formatted

### Notes
[Any implementation notes or questions]
```

## 🐛 Known Challenges

1. **Performance**: Python is slower than C
   - Solution: Use NumPy vectorization, consider numba for hot loops

2. **MKL Dependency**: C code uses Intel MKL
   - Solution: Use NumPy/SciPy (built on BLAS/LAPACK)

3. **MPI Parallelization**: C code uses MPI
   - Solution: Start single-process, optionally add mpi4py later

4. **Complex Physics**: Some modules require physics knowledge
   - Solution: Follow C code closely, verify with tests

## 📞 Getting Help

When you encounter issues:
1. Check the QUICKSTART.md for common patterns
2. Review the C code more carefully
3. Look for similar code in completed modules
4. Ask specific questions with context

## 🎉 Getting Started

**Your first task**: Translate `constants.py`

This is the simplest module and a good way to understand the workflow.

See `python_wannsymm/docs/QUICKSTART.md` for detailed instructions.

Good luck! 🚀
