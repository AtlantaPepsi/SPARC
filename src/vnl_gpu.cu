#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu.h"
#include "mpi.h"
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" { //}
#endif
#include "vnl_gpu.h"
void Vnl_gpu(const min_SPARC_OBJ *pSPARC, const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
             const NLOC_PROJ_OBJ *nlocProj,
             const min_SPARC_OBJ *d_SPARC, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
             const NLOC_PROJ_OBJ *d_locProj,
             const int DMnd, const int ncol, double *d_x, double *d_Hx, MPI_Comm comm, int GPUDirect)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    dim3 blockDims(16,16);

    stat = cublasCreate(&handle);

    /* compute nonlocal operator times vector(s) */
    double one = 1.0, zero = 0.0;
    double *d_alpha;
    cudaMalloc((void **)&d_alpha,  pSPARC->IP_displ[pSPARC->n_atom] * ncol * sizeof(double));
    // |pSPARC->IP_displ[pSPARC->n_atom]|  = n_atom * nproj(of each atom)

    /*first find inner product*/
    //double Start = MPI_Wtime();
    for (int type = 0; type < pSPARC->Ntypes; type++) {

        if (! nlocProj[type].nproj) continue; // this is typical for hydrogen

        for (int atom = 0; atom < Atom_Influence_nloc[type].n_atom; atom++) {

            int ndc = Atom_Influence_nloc[type].ndc[atom];
            printf("\n\natom: %d\n",atom);

            dim3 gridDims( (ndc-1)/blockDims.x + 1, (ncol-1)/blockDims.y + 1 );
            const size_t shmem = 16 * sizeof(int);//ndc * sizeof(double);

            double *d_xrc; // = (double *)malloc( ndc * ncol * sizeof(double));
            cudaMalloc((void **)&d_xrc,  ndc * ncol * sizeof(double));

            x_rc<<<gridDims, blockDims, shmem>>>(d_xrc, d_x, d_Atom_Influence_nloc, ncol, type, atom, DMnd);

            double *xrc = (double *)calloc( ndc * ncol, sizeof(double));
            cudaMemcpy(xrc, d_xrc, ndc * ncol * sizeof(double), cudaMemcpyDeviceToHost);

            printf("gpu xrc: %f\n",xrc[ndc-1]);
            int atom_index = Atom_Influence_nloc[type].atom_index[atom];

            /*double *chi;
            cudaMalloc((void **)&chi,  ndc * nlocProj[type].nproj * sizeof(double));
            cudaMemcpy(chi, nlocProj[type].Chi[atom], ndc * nlocProj[type].nproj * sizeof(double), cudaMemcpyHostToDevice);*/

            stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               nlocProj[type].nproj, ncol, ndc,
                               &(pSPARC->dV), nlocProj[type].Chi[atom], ndc,
                               d_xrc, ndc, &one,
                               d_alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[type].nproj);

            if(stat!=CUBLAS_STATUS_SUCCESS) printf("error: %d\n",stat);
            gpuErrchk(cudaPeekAtLastError() );    
            /*printf("displ: %d\n",pSPARC->IP_displ[atom_index]*ncol);
            printf("ndc: %d\n",ndc);
            printf("nnproj: %d\n",nlocProj[type].nproj);
            printf("chi: %f\n",nlocProj[type].Chi[atom][ndc* nlocProj[type].nproj-2]);
            printf("dv: %f\n",pSPARC->dV);
            printf("chi: %f\n",chi[ndc-1]);*/

            cudaFree(d_xrc);
        }
    } 
    //printf("1st total: %f\n",(MPI_Wtime()-Start)*1e3);

    //printf("xv exited safely\n\n");

    /*double *af = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
    cudaMemcpy(af, d_alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    printf("alpha copy:%f \n", af[0]);*/

    /* if there are domain parallelization over each band,
     * we need to sum over all processes over domain comm */
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        printf("size %d\n",commsize);
    if (GPUDirect){
            MPI_Allreduce(MPI_IN_PLACE, d_alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, comm);
        } else {
            double *alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
            cudaMemcpy(alpha, d_alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, comm);
            cudaMemcpy(d_alpha, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * sizeof(double), cudaMemcpyHostToDevice);
            free(alpha);
        }

    }



    printf("reduce exited safely\n\n");
    /* go over all atoms and multiply gamma_Jl to the inner product */
    //Start = MPI_Wtime();
    dim3 gridDim( (pSPARC->IP_displ[pSPARC->n_atom]-1)/blockDims.x + 1, (ncol-1)/blockDims.y + 1 );
    Vnl_gammaV<<<gridDim, blockDims>>>(d_SPARC, d_alpha, ncol);
    //printf("2nd total: %f\n",(MPI_Wtime()-Start)*1e3);

    /*double *af = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
    cudaMemcpy(af, d_alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    printf("alpha copy:%f \n", af[0]);*/

    gpuErrchk(cudaPeekAtLastError() );    
    //printf("gamma exited safely\n\n");
            

    /* multiply the inner product and the nonlocal projector */
    //Start = MPI_Wtime();
    for (int type = 0; type < pSPARC->Ntypes; type++) {
        if (! nlocProj[type].nproj) continue; // this is typical for hydrogen

        for (int atom = 0; atom < Atom_Influence_nloc[type].n_atom; atom++) {

            int ndc = Atom_Influence_nloc[type].ndc[atom];

            dim3 gridDims( (ndc-1)/blockDims.x + 1, (ncol-1)/blockDims.y + 1);
            const size_t shmem = 16 * sizeof(int);//ndc * sizeof(double);

            double *Vnlx;// = (double *)malloc( ndc * ncol * sizeof(double));
            cudaMalloc((void **)&Vnlx,  ndc * ncol * sizeof(double));
            gpuErrchk(cudaPeekAtLastError() );    

            int atom_index = Atom_Influence_nloc[type].atom_index[atom];
            
            /*double *chi;
            cudaMalloc((void **)&chi,  ndc * nlocProj[type].nproj * sizeof(double));
            cudaMemcpy(chi, nlocProj[type].Chi[atom], ndc * nlocProj[type].nproj * sizeof(double), cudaMemcpyHostToDevice);*/
  
            stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,                ///!
                               ndc, ncol, nlocProj[type].nproj,
                               &one, nlocProj[type].Chi[atom], ndc,
                               d_alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[type].nproj, &zero,
                               Vnlx, ndc);

            gpuErrchk(cudaPeekAtLastError() );    
            update<<<gridDims, blockDims, shmem>>>(d_Hx, Vnlx, d_Atom_Influence_nloc, ncol, type, atom, DMnd);

            cudaFree(Vnlx);                                       /////! persistence issue
        }
    }
    //printf("3rd total: %f\n",(MPI_Wtime()-Start)*1e3);

    cudaFree(d_alpha);
    cublasDestroy(handle);
}

__global__
void x_rc(double *d_xrc, double *d_x, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
          int ncol, int type, int atom, int DMnd)
{
    extern __shared__ int shared_grid_pose[];


    int index = blockIdx.x*blockDim.x + threadIdx.x;
    
    int n = blockIdx.y*blockDim.y + threadIdx.y;
    int ndc = d_Atom_Influence_nloc[type].ndc[atom];

    if (threadIdx.y == 0) {
        if (index < ndc)
            shared_grid_pose[threadIdx.x] = d_Atom_Influence_nloc[type].grid_pos[atom][index];  //?
    }
    __syncthreads();

    if (index < ndc && n < ncol)
        d_xrc[n*ndc+index] = d_x[n*DMnd + shared_grid_pose[threadIdx.x]];                       ///?
    
    //if (index==0){//&&n==0)
    /*printf("test1: %d\n",d_Atom_Influence_nloc->atom_index[0]);
    printf("test2: %d\n",d_Atom_Influence_nloc->grid_pos[0][3]);
    printf("test3: %d\n",ndc);
    printf("test4: %f\n",d_xrc[n*ndc+index]); */
    //}

}

__global__
void Vnl_gammaV(const min_SPARC_OBJ *d_SPARC, double *d_alpha, int ncol)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;

    if (index >= d_SPARC->IP_displ[d_SPARC->n_atom] || n >= ncol)
        return;


    int i = 1;                                                   //index of atom
    int type = 0;
    int temp = d_SPARC->nAtomv[0];
    while (index >= d_SPARC->IP_displ[i]) {     ///<=?
        i++;                                        ///i + 1
        if (i > temp) {
            temp += d_SPARC->nAtomv[type];
            type++;
        }
    }

    //printf("type: %d, atom: %d, ndc: %d, n: %d\n",type,i,index,n);        

    int leftover = index - d_SPARC->IP_displ[i-1];
    
    //if (index ==  1)  {
    //printf("type: %d, atom: %d, ndc: %d, n: %d\n",type,i,index,n);        
    //printf("bool: %p\n", /*leftover); - ( revotfel +*/ (d_SPARC->ppl));//*(2*l+1) ) > 0);
    //printf("bool: %p\n", /*leftover); - ( revotfel +*/ (d_SPARC->ppl[1]));//*(2*l+1) ) > 0);
    //printf("bllo: %p\n", /*leftover); - ( revotfel +*/ (d_SPARC->Gamma[1]));//*(2*l+1) ) > 0);
    //}
        
    //int nproj = d_SPARC->IP_displ[i+1] - d_SPARC->IP_displ[i];
    //leftover %= nproj;


    int lmax = d_SPARC->lmax[type];
    int lloc = d_SPARC->localPsd[type];
    int l = 0;
    int ldispl = 0;
    int revotfel = 0;

    for (l = 0; l <= lmax; l++) {
        // __syncthreads();
        if (l == lloc) {
            ldispl += (d_SPARC->ppl[type])[l];
            continue;
        }

        
        if (leftover - ( revotfel + (d_SPARC->ppl[type])[l]*(2*l+1) ) > 0) {
            ldispl += (d_SPARC->ppl[type])[l];
            revotfel += (d_SPARC->ppl[type])[l] * (2 * l + 1);
        } else {
            leftover -= revotfel;
            int np = leftover / (2*l+1);
            int actual = d_SPARC->IP_displ[i-1] * ncol 
                       + (d_SPARC->IP_displ[i] - d_SPARC->IP_displ[i-1]) * n
                       + (index - d_SPARC->IP_displ[i-1]);
                //printf("actual %d\n", actual);
    
            d_alpha[actual] *=  (d_SPARC->Gamma[type])[ldispl+np];

            return;
        }
    //printf("gammav: %d, %d\n",ldispl, l);
    }
    //printf("2nd total: %f\n",(MPI_Wtime()-Start)*1e3);
}

__global__
void update(double *d_Hx, double *Vnlx, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
            int ncol, int type, int atom, int DMnd)
{

    extern __shared__ int shared_grid_pose[];

    int ndc = d_Atom_Influence_nloc[type].ndc[atom];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;

    if (threadIdx.y == 0) {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < ndc)
            shared_grid_pose[threadIdx.x] = d_Atom_Influence_nloc[type].grid_pos[atom][index];  //?
    }
    __syncthreads();

    if (index < ndc && n < ncol) {
        d_Hx[n*DMnd + shared_grid_pose[threadIdx.x]] += Vnlx[n*ndc+index];
    }
}

GPU_GC* interface_gpu(const SPARC_OBJ *pSPARC,                            min_SPARC_OBJ *d_SPARC,
                      const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
                      const NLOC_PROJ_OBJ *nlocProj,                      NLOC_PROJ_OBJ *d_locProj)
{
    GPU_GC *gc = (GPU_GC*) malloc(sizeof(GPU_GC));

    min_SPARC_OBJ *min_SPARC = (min_SPARC_OBJ*) malloc(sizeof(min_SPARC_OBJ));
    interface(pSPARC, min_SPARC);
    
    int Ntypes = pSPARC->Ntypes;
    int n_atom = pSPARC->n_atom;
    gc->Ntypes = pSPARC->Ntypes;
    gc->n_atom = pSPARC->n_atom;

    gc->nAtomv = (int*) malloc(sizeof(int) * gc->Ntypes);
    memcpy(gc->nAtomv, min_SPARC->nAtomv, sizeof(int) * gc->Ntypes);
/*
    d_SPARC
        = (min_SPARC_OBJ*) malloc(sizeof(min_SPARC_OBJ));
    d_Atom_Influence_nloc
        = (ATOM_NLOC_INFLUENCE_OBJ*) malloc(sizeof(ATOM_NLOC_INFLUENCE_OBJ) * Ntypes);
    d_locProj
        = (NLOC_PROJ_OBJ*) malloc(sizeof(NLOC_PROJ_OBJ) * Ntypes);
*/

    cudaMalloc((void **)&d_SPARC,                sizeof(min_SPARC_OBJ));
    cudaMalloc((void **)&d_Atom_Influence_nloc,  sizeof(ATOM_NLOC_INFLUENCE_OBJ) * Ntypes);

    cudaMemcpy(d_SPARC, min_SPARC, sizeof(min_SPARC_OBJ), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError() );

    cudaMemcpy(d_Atom_Influence_nloc, Atom_Influence_nloc, sizeof(ATOM_NLOC_INFLUENCE_OBJ) * Ntypes, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError() );

    cudaMemcpy(d_locProj, nlocProj, sizeof(NLOC_PROJ_OBJ) * Ntypes, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError() );


    //deep copy of atom_influ obj
    gc->d_ndc   = (int**) malloc(sizeof(int*) * Ntypes);
    gc->dd_cpy  = (int***) malloc(sizeof(int**) * Ntypes);
    gc->tmp_ptr = (int***) malloc(sizeof(int**) * Ntypes);
    for (int i = 0; i < Ntypes; i++) {
        int *d_ndc, **dd_cpy;

        cudaMalloc((void**)&d_ndc, sizeof(int) * pSPARC->nAtomv[i] );
        cudaMemcpy(d_ndc, Atom_Influence_nloc[i].ndc, sizeof(int)*pSPARC->nAtomv[i], cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_Atom_Influence_nloc[i].ndc), &d_ndc, sizeof(int*), cudaMemcpyHostToDevice);
        gc->d_ndc[i] = d_ndc;

        cudaMalloc((void**)&dd_cpy, sizeof(int*) * pSPARC->nAtomv[i] );
        gc->dd_cpy[i] = dd_cpy;

        int** tmp_ptr = (int **)malloc( sizeof(int*) * pSPARC->nAtomv[i] );
        gc->tmp_ptr[i] = tmp_ptr;

        for(int j = 0; j < pSPARC->nAtomv[i]; j++) {
            int ndc = Atom_Influence_nloc[i].ndc[j];
            cudaMalloc( (void**)&tmp_ptr[j], sizeof(int) * ndc );
            cudaMemcpy(tmp_ptr[j], Atom_Influence_nloc[i].grid_pos[j], sizeof(int)*ndc, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(dd_cpy, tmp_ptr, sizeof(tmp_ptr), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_Atom_Influence_nloc[i].grid_pos), &dd_cpy, sizeof(int**), cudaMemcpyHostToDevice);
    }
    gpuErrchk(cudaPeekAtLastError() );


    //deep copy of sparc obj

    int *d_local_psd, *d_nAtomv, *d_IP, *d_lmax, **dd_ppl;
    double **dd_gamma;
    cudaMalloc((void**)&d_local_psd,sizeof(int) * Ntypes  );
    cudaMalloc((void**)&d_nAtomv   ,sizeof(int) * Ntypes  );
    cudaMalloc((void**)&d_IP       ,sizeof(int) * (n_atom+1)  );
    cudaMalloc((void**)&d_lmax     ,sizeof(int) * Ntypes  );
    cudaMalloc((void**)&dd_ppl     ,sizeof(int*)* Ntypes  );
    cudaMalloc((void**)&dd_gamma   ,sizeof(double*)* Ntypes  );

    gc->d_local_psd = d_local_psd;
    gc->d_nAtomv    = d_nAtomv;
    gc->d_IP        = d_IP;
    gc->d_lmax      = d_lmax;

    cudaMemcpy(d_local_psd, min_SPARC->localPsd, sizeof(int) * Ntypes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_SPARC->localPsd), &d_local_psd, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nAtomv, min_SPARC->nAtomv, sizeof(int)*Ntypes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_SPARC->nAtomv), &d_nAtomv, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IP, min_SPARC->IP_displ, sizeof(int)*(n_atom+1), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_SPARC->IP_displ), &d_IP, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lmax, min_SPARC->lmax, sizeof(int)*Ntypes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_SPARC->lmax), &d_lmax, sizeof(int*), cudaMemcpyHostToDevice);

    int** ppls     = (int **)malloc( sizeof(int*) * Ntypes );
    double** gamma = (double **)malloc( sizeof(double*) * Ntypes );
    gc->ppls   = ppls;
    gc->gamma  = gamma;

    for(int i=0; i<Ntypes; i++) {
        cudaMalloc( (void**)&ppls[i], sizeof(int)*(min_SPARC->lmax[i]+1) );
        cudaMemcpy(ppls[i], min_SPARC->ppl[i], sizeof(int)*(min_SPARC->lmax[i]+1), cudaMemcpyHostToDevice);
        cudaMalloc( (void**)&gamma[i], sizeof(double)*10 ); //line 139
        cudaMemcpy(gamma[i], min_SPARC->Gamma[i], sizeof(double)*10, cudaMemcpyHostToDevice);

        cudaMemcpy(dd_ppl + i, ppls + i, sizeof(sizeof(int*)/*ppls*/), cudaMemcpyHostToDevice);
        cudaMemcpy(dd_gamma + i, gamma + i, sizeof(sizeof(double*)/*ppls*/), cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError() );
    }
 

    cudaMemcpy(&(d_SPARC->ppl), &dd_ppl, sizeof(int**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_SPARC->Gamma), &dd_gamma, sizeof(double**), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError() );


    //deep copy of nonlocal chi's
    gc->dd_chi   = (double***) malloc(sizeof(double**) * Ntypes);
    gc->tmp_ptr2 = (double***) malloc(sizeof(double**) * Ntypes);

    for(int i=0; i<Ntypes; i++) {
        double** dd_chi;
        double** tmp_ptr2 = (double **)malloc( sizeof(double*) * pSPARC->nAtomv[i] );
        cudaMalloc((void**)&dd_chi, sizeof(double*) * pSPARC->nAtomv[i] );
      
        for(int j = 0; j < pSPARC->nAtomv[i]; j++) {
            int ndc = Atom_Influence_nloc[i].ndc[j]; 
            cudaMalloc((void **)&tmp_ptr2[j],  ndc * nlocProj[i].nproj * sizeof(double));
            cudaMemcpy(tmp_ptr2[j], nlocProj[i].Chi[j], ndc * nlocProj[i].nproj * sizeof(double), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(dd_chi, tmp_ptr2, sizeof(tmp_ptr2), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_locProj[i].Chi), &dd_chi, sizeof(double**), cudaMemcpyHostToDevice);
        gc->dd_chi[i]   = dd_chi;
        gc->tmp_ptr2[i] = tmp_ptr2;
    }



    free_min_SPARC(min_SPARC);
    return gc;
}
  
void free_gpu_SPARC(min_SPARC_OBJ *d_SPARC, ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
                    NLOC_PROJ_OBJ *d_locProj, GPU_GC *gc)
{
    //free sparc
    cudaFree(gc->d_local_psd);
    cudaFree(gc->d_nAtomv);
    cudaFree(gc->d_IP);
    cudaFree(gc->d_lmax);

    for(int i = 0; i < gc->Ntypes; i++) {
        cudaFree(gc->d_ndc[i]);
        int **tmp_ptr = gc->tmp_ptr[i];
        cudaFree(gc->dd_cpy);

        cudaFree(gc->ppls[i]);
        cudaFree(gc->gamma[i]);

        double** tmp_ptr2 = gc->tmp_ptr2[i];
        for (int j = 0; j < gc->nAtomv[i]; j++) {
            cudaFree(tmp_ptr[j]);
            cudaFree(tmp_ptr2[j]);
        }
        free(tmp_ptr2);
        cudaFree(gc->dd_chi[i]);

        cudaFree(d_Atom_Influence_nloc+i);
        cudaFree(d_locProj+i);
    }
    free(gc->d_ndc);
    free(gc->dd_cpy);
    free(gc->tmp_ptr);

    free(gc->tmp_ptr2);
    free(gc->dd_chi);

    free(gc->ppls);
    free(gc->gamma);
 
    cudaFree(d_SPARC);

    free(gc->nAtomv);
    free(gc);
}

double test_gpu(const SPARC_OBJ *pSPARC, const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, const NLOC_PROJ_OBJ *nlocProj,
                const int DMnd, const int ncol, double *x, double *Hx, MPI_Comm comm, double *hx)
{
    //hx is before hx, Hx is truth
    min_SPARC_OBJ *d_SPARC;
    ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc;
    NLOC_PROJ_OBJ *d_locProj;
  
    min_SPARC_OBJ *min_SPARC = (min_SPARC_OBJ*) malloc(sizeof(min_SPARC_OBJ));
    interface(pSPARC, min_SPARC);
  
    GPU_GC *gc = interface_gpu(pSPARC,              d_SPARC,
                               Atom_Influence_nloc, d_Atom_Influence_nloc,
                               nlocProj,            d_locProj);      //#todo: modify psparc to min sparc
  
    double *d_x, *d_Hx;
    cudaMalloc((void **)&d_x,  DMnd * ncol * sizeof(double));
    cudaMalloc((void **)&d_Hx,  DMnd * ncol * sizeof(double));
    cudaMemcpy(d_x, x, DMnd * ncol * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Hx, hx, DMnd * ncol * sizeof(double), cudaMemcpyHostToDevice);
    
  
    double t1 = MPI_Wtime();
    Vnl_gpu(min_SPARC, Atom_Influence_nloc, nlocProj,
            d_SPARC, d_Atom_Influence_nloc, d_locProj,
            DMnd, ncol, d_x, d_Hx, comm, 0);
    double t2 = MPI_Wtime();
    
    //copy answer back
    double *d_hx = (double *)malloc(DMnd * ncol * sizeof(double));
    cudaMemcpy(d_hx, d_Hx, DMnd * ncol * sizeof(double), cudaMemcpyDeviceToHost);
  
    free_gpu_SPARC(d_SPARC, d_Atom_Influence_nloc, d_locProj, gc);
    free_min_SPARC(min_SPARC);
    cudaFree(d_x);
    cudaFree(d_Hx);
  
  
    double local_err;
    int err_count = 0;
    for (int ix = 0; ix < DMnd*ncol; ix++)
    {
        local_err = fabs(d_hx[ix] - Hx[ix]) / fabs(Hx[ix]);
        // Consider a relative error of 1e-10 to guard against floating point rounding
        if ((local_err > 1e-10) || isnan(local_err))
        {
            //printf("At index %d: %.15f vs. %.15f\n", ix, fabs(psi_new[ix]), fabs(psi_chk[ix]));
            err_count = err_count + 1;
        }
    }

    if (err_count > 1) {
        printf("There are %d errors out of %d entries!\n", err_count, ncol*DMnd);
        exit(0);
    }
    free(d_hx);
  
    return (t2-t1)*1e3;
    
}

#ifdef __cplusplus
}
#endif
