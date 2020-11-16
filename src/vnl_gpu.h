#ifdef __cplusplus
extern "C" { //

#endif


#include "vnl_mod.h"
#include "gpu.h"
//#include "stencil_gpu.h"
//#include "memory.h"


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
  




#ifdef __cplusplus
}
#endif
