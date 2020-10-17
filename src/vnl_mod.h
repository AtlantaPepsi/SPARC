#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "isddft.h"
#include <mkl.h>

typedef struct _min_SPARC_OBJ {

    int n_atom; // total number of atoms
    int Ntypes; // number of atome types
    int *IP_displ;  // start index for storing nonlocal inner product, size: (n_atom + 1) x 1

    int *localPsd;          // respective local component of pseudopotential for
                            // each type of atom. 0 for s, 1 for p, 2 for d, 3 for f

    int *nAtomv;            // number of atoms of each type
    double dV;

    //from pSPARC->PSD_OBJ
    double **Gamma;
    int *l_sum;      //size of gamma, lengths of Ntypes
    int **ppl;       // number of nonlocal projectors per l
    int *lmax;       // maximum pseudopotential component

} min_SPARC_OBJ;

void free_min_SPARC(min_SPARC_OBJ* min_SPARC);

void interface(const SPARC_OBJ *pSPARC, min_SPARC_OBJ* min_SPARC);

void Vnl_mod(
    const min_SPARC_OBJ *pSPARC,
    int DMnd,
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
    NLOC_PROJ_OBJ *nlocProj,
    int ncol,
    double *x,
    double *Hx,
    MPI_Comm comm
);

void test_vnl(
    const SPARC_OBJ *pSPARC,
    int DMnd,
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
    NLOC_PROJ_OBJ *nlocProj,
    int ncol,
    double *x,
    double *Hx,
    MPI_Comm comm,
    double *hx
);