/* Stub spglib.h for testing
 * This header provides minimal stubs for spglib functions
 * Only for compilation of tests, not for actual use
 */

#ifndef __SPGLIB_H__
#define __SPGLIB_H__

/* Minimal type definitions to allow compilation */
typedef struct {
    double lattice[3][3];
    double (*positions)[3];
    int *types;
    int num_atoms;
    double symprec;
} SpglibDataset;

typedef struct {
    int spacegroup_number;
    int hall_number;
    char international_symbol[11];
    char hall_symbol[17];
    char choice[6];
    double transformation_matrix[3][3];
    double origin_shift[3];
    int n_operations;
    int (*rotations)[3][3];
    double (*translations)[3];
} SpglibSpacegroupType;

#endif /* __SPGLIB_H__ */
