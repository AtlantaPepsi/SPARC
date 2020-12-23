#include "vnl_mod.h"
#include "gpu.h"
//#include "stencil_gpu.h"
//#include "memory.h"

#ifdef __cplusplus
extern "C" { //

#endif

typedef struct _GPU_GARBAGE_COLLECTOR {

    int n_atom;          // total number of atoms
    int Ntypes;          // number of atome types
    int *nAtomv;
  
/// atom_influence
    int **d_ndc;         // GPU pointers to _ndc at each one of _Ntypes Atom_Influence_obj
    int ***dd_cpy;       // GPU pointers to _grid_pos of each one of _nAtomv at each one of _Ntypes Atom_Influence_obj
    int ***tmp_ptr;      // host side memory of dd_cpy     #todo: free in interface code
  
/// nloc_chi
    double ***dd_chi;       // GPU pointers to _chi of each one of _nAtomv at each one of _Ntypes Nloc_Proj_obj 
    double ***tmp_ptr2;     // host side memory of dd_chi     #todo: free in interface code
  
/// sparc_obj
    int *d_local_psd, *d_nAtomv, *d_IP, *d_lmax; //yatayatayata
    int **dd_ppl;        // GPU pointers to _ppl for each one of _Ntypes
    double **dd_gamma;   // GPU pointers to _gamma for each one of _Ntypes
    int **ppls;          //host side memory of dd_cpy     #todo: free in interface code
    double **gamma;      //host side memory of dd_cpy     #todo: free in interface code

} GPU_GC;
  
  
void Vnl_gpu(const min_SPARC_OBJ *pSPARC, const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
             const NLOC_PROJ_OBJ *nlocProj,
             const min_SPARC_OBJ *d_SPARC, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
             const NLOC_PROJ_OBJ *d_locProj,
             const int DMnd, const int ncol, double *d_x, double *d_Hx, MPI_Comm comm, int GPUDirect);

  __global__
void x_rc(double *d_xrc, double *d_x, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
          int ncol, int type, int atom, int DMnd);
  
  __global__
void Vnl_gammaV(const min_SPARC_OBJ *d_SPARC, double *d_alpha, int ncol);
  
  __global__
void update(double *d_Hx, double *Vnlx, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
            int ncol, int type, int atom, int DMnd);
 
GPU_GC* interface_gpu(const SPARC_OBJ *pSPARC,                            min_SPARC_OBJ *min_SPARC,
                      const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
                      const NLOC_PROJ_OBJ *nlocProj,                      NLOC_PROJ_OBJ *d_locProj);
  
void free_gpu_SPARC(min_SPARC_OBJ *min_SPARC, ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
                    NLOC_PROJ_OBJ *d_locProj, GPU_GC *gc);

double test_gpu(const SPARC_OBJ *pSPARC, const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, const NLOC_PROJ_OBJ *nlocProj,
                const int DMnd, const int ncol, double *x, double *Hx, MPI_Comm comm);


#ifdef __cplusplus
}
#endif
