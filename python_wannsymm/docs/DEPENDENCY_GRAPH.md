# WannSymm Module Dependency Graph

```
Level 0: Foundation (No dependencies)
┌──────────────┐
│ constants.py │ ← Start here
└──────────────┘

Level 1: Basic Operations
┌──────────────┐     ┌──────────────┐
│  vector.py   │     │  matrix.py   │
│ (depends on  │     │ (depends on  │
│ constants)   │     │ constants)   │
└──────────────┘     └──────────────┘

Level 2: Core Data Structures
┌──────────────┐     ┌──────────────┐
│  wannorb.py  │     │ usefulio.py  │
│ (depends on  │     │ (standard    │
│ vector)      │     │  library)    │
└──────────────┘     └──────────────┘
      │
      ▼
┌──────────────┐
│ wanndata.py  │
│ (depends on  │
│ vector,      │
│ wannorb)     │
└──────────────┘

Level 3: Input/Output
┌────────────────────────────────────┐
│         readinput.py               │
│ (depends on vector, matrix,        │
│  wannorb, wanndata)                │
│                                    │
│ ← LARGEST MODULE (1,830 lines)    │
└────────────────────────────────────┘
      │
      ▼
┌──────────────┐
│ readsymm.py  │
│ (depends on  │
│ readinput,   │
│ wanndata,    │
│ vector)      │
└──────────────┘

Level 4: Rotation Operations (Physics Core)
┌────────────────────┐     ┌────────────────────┐
│ rotate_orbital.py  │     │ rotate_spinor.py   │
│ (depends on        │     │ (depends on        │
│  constants)        │     │  constants)        │
│                    │     │                    │
│ ← Complex physics  │     │ ← Spin physics     │
└────────────────────┘     └────────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
         ┌────────────────────┐
         │  rotate_basis.py   │
         │  (depends on       │
         │   rotate_orbital,  │
         │   rotate_spinor,   │
         │   vector, wanndata)│
         └────────────────────┘
                    │
                    ▼
         ┌────────────────────────────────────┐
         │      rotate_ham.py                 │
         │  (depends on vector, matrix,       │
         │   wanndata, readinput,             │
         │   rotate_orbital, rotate_spinor)   │
         │                                    │
         │  ← SECOND LARGEST (890 lines)     │
         │  ← CORE ALGORITHM                 │
         └────────────────────────────────────┘

Level 5: Analysis
┌────────────────────────────────────┐
│         bndstruct.py               │
│ (depends on vector, matrix,        │
│  wanndata, readinput,              │
│  rotate_basis, rotate_ham)         │
│                                    │
│ ← Band structure calculation       │
└────────────────────────────────────┘

Level 6: Main Program
┌────────────────────────────────────┐
│           main.py                  │
│ (depends on ALL modules)           │
│                                    │
│ ← Entry point, orchestration      │
└────────────────────────────────────┘
```

## Translation Order (Bottom-Up)

### Phase 1: Foundation ⭐ START HERE
1. `constants.py` (0.5 hours)
2. `vector.py` (4-6 hours)  
3. `matrix.py` (2-3 hours)

**Why this order?** These are the foundation with no dependencies.

### Phase 2: Core Structures
4. `wannorb.py` (2-3 hours)
5. `wanndata.py` (6-8 hours)
6. `usefulio.py` (2-3 hours)

**Why this order?** Build core data structures on top of foundation.

### Phase 3: I/O
7. `readinput.py` (20-30 hours) ⚠️ LARGEST, most complex
8. `readsymm.py` (3-4 hours)

**Why this order?** Need data structures before parsing input files.

### Phase 4: Rotation Operations
9. `rotate_orbital.py` (10-15 hours) ⚠️ Complex physics
10. `rotate_spinor.py` (4-6 hours)
11. `rotate_basis.py` (6-8 hours)
12. `rotate_ham.py` (15-20 hours) ⚠️ CORE ALGORITHM

**Why this order?** Build up rotation capabilities incrementally.

### Phase 5: Analysis
13. `bndstruct.py` (10-12 hours)

**Why this order?** Needs all rotation operations complete.

### Phase 6: Integration
14. `main.py` (8-10 hours)

**Why this order?** Final integration after all modules ready.

## Critical Path

The critical modules that require the most attention:

1. **readinput.py** - Most complex input parsing
2. **rotate_ham.py** - Core symmetrization algorithm
3. **rotate_orbital.py** - Complex quantum mechanics
4. **main.py** - Integration and workflow

## Module Complexity Summary

| Module | Lines | Complexity | Priority | Effort |
|--------|-------|------------|----------|--------|
| constants | 26 | ⭐ Trivial | ⚡ Highest | 0.5h |
| vector | 412 | ⭐⭐ Low-Med | ⚡ High | 4-6h |
| matrix | 94 | ⭐ Low | ⚡ High | 2-3h |
| wannorb | 69 | ⭐ Low | ⚡ High | 2-3h |
| wanndata | 343 | ⭐⭐ Medium | ⚡ High | 6-8h |
| usefulio | 74 | ⭐ Low | Medium | 2-3h |
| readinput | 1,830 | ⭐⭐⭐⭐ High | ⚡ Critical | 20-30h |
| readsymm | 88 | ⭐⭐ Low-Med | ⚡ High | 3-4h |
| rotate_orbital | 337 | ⭐⭐⭐⭐ High | ⚡ Critical | 10-15h |
| rotate_spinor | 74 | ⭐⭐ Medium | ⚡ High | 4-6h |
| rotate_basis | 156 | ⭐⭐ Medium | ⚡ High | 6-8h |
| rotate_ham | 890 | ⭐⭐⭐⭐ High | ⚡ Critical | 15-20h |
| bndstruct | 401 | ⭐⭐⭐ Med-High | Medium | 10-12h |
| main | 789 | ⭐⭐ Medium | ⚡ High | 8-10h |

**Total Estimated Effort**: 93-138 hours (12-18 working days)

## Parallelization Opportunities

These modules can be worked on in parallel after Phase 1:

- `wannorb.py` and `usefulio.py` (independent)
- `rotate_orbital.py` and `rotate_spinor.py` (independent)
- Tests for each module (can be written in parallel with implementation)

## Key Dependencies to Note

- **Everything depends on** `constants.py` → Do first!
- **Most modules depend on** `vector.py` → Do early!
- `rotate_ham.py` depends on rotation modules → Do last in Phase 4
- `main.py` depends on everything → Do last overall

## Testing Strategy per Phase

### Phase 1-2: Unit Tests Only
Each module gets comprehensive unit tests.

### Phase 3: Integration Tests Start
Test input parsing with actual input files.

### Phase 4: Physics Validation
Verify rotations preserve physical properties.

### Phase 5-6: End-to-End Tests
Full workflow tests with example systems.

## Success Checkpoints

- ✅ **After Phase 1**: Can create and manipulate vectors/matrices
- ✅ **After Phase 2**: Can represent Wannier orbitals and Hamiltonians
- ✅ **After Phase 3**: Can read input files and structure files
- ✅ **After Phase 4**: Can symmetrize Hamiltonians (MAIN GOAL!)
- ✅ **After Phase 5**: Can calculate band structures
- ✅ **After Phase 6**: Complete working program

## References

- Detailed analysis: `../PYTHON_TRANSLATION_ANALYSIS.md`
- Translation workflow: `TRANSLATION_WORKFLOW.md`
- C source code: `../src/`
