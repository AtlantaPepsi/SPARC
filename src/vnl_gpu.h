//#ifdef __cplusplus
extern "C" { //

//#endif


#include "vnl_mod.h"
#include "gpu.h"
//#include "stencil_gpu.h"
//#include "memory.h"


void Vnl_gpu(const min_SPARC_OBJ *pSPARC, const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
             const NLOC_PROJ_OBJ *nlocProj,
             const min_SPARC_OBJ *d_SPARC, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
             const NLOC_PROJ_OBJ *d_locProj,
             const int DMnd, const int ncol, double *d_x, double *d_Hx, int GPUDirect);



//#ifdef __cplusplus
}
//#endif
