

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu.h"
#include "mpi.h"
#include "vnl_gpu.h"
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" { //}
#endif
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

            dim3 gridDims( (ndc-1)/blockDims.x + 1, (ncol-1)/blockDims.y + 1);
            const size_t shmem = 16 * sizeof(double);//ndc * sizeof(double);

            double *d_xrc; // = (double *)malloc( ndc * ncol * sizeof(double));
            cudaMalloc((void **)&d_xrc,  ndc * ncol * sizeof(double));

            x_rc<<<gridDims, blockDims, shmem>>>(d_xrc, d_x, d_Atom_Influence_nloc, ncol, type, atom, DMnd);

            int atom_index = Atom_Influence_nloc[type].atom_index[atom];

            stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               nlocProj[type].nproj, ncol, ndc,
                               &(pSPARC->dV), d_locProj[type].Chi[atom], ndc,
                               d_xrc, ndc, &one,
                               d_alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[type].nproj);

            cudaFree(d_xrc);
        }
    }
    //printf("1st total: %f\n",(MPI_Wtime()-Start)*1e3);




    /* if there are domain parallelization over each band,
     * we need to sum over all processes over domain comm */
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
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



    /* go over all atoms and multiply gamma_Jl to the inner product */
    //Start = MPI_Wtime();
    Vnl_gammaV(pSPARC, alpha, ncol);
    //printf("2nd total: %f\n",(MPI_Wtime()-Start)*1e3);


    /* multiply the inner product and the nonlocal projector */
    //Start = MPI_Wtime();
    for (int type = 0; type < pSPARC->Ntypes; type++) {
        if (! nlocProj[type].nproj) continue; // this is typical for hydrogen

        for (int atom = 0; atom < Atom_Influence_nloc[type].n_atom; atom++) {

            int ndc = Atom_Influence_nloc[type].ndc[atom];

            dim3 gridDims( (ndc-1)/blockDims.x + 1, (ncol-1)/blockDims.y + 1);
            const size_t shmem = 16 * sizeof(double);//ndc * sizeof(double);

            double *Vnlx;// = (double *)malloc( ndc * ncol * sizeof(double));
            cudaMalloc((void **)&Vnlx,  ndc * ncol * sizeof(double));

            int atom_index = Atom_Influence_nloc[type].atom_index[atom];

            stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,                ///!
                               ndc, ncol, nlocProj[type].nproj,
                               &one, d_locProj[type].Chi[atom], ndc,
                               d_alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[type].nproj, &zero,
                               Vnlx, ndc);

            update<<<gridDims, blockDims, shmem>>>(d_Hx, Vnlx, d_Atom_Influence_nloc, ncol, type, atom, DMnd);

            cudaFree(Vnlx);
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
    extern __shared__ double shared_grid_pose[];

    int ndc = d_Atom_Influence_nloc[type].ndc[atom];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;

    if (threadIdx.y == 0) {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < ndc)
            shared_grid_pose[threadIdx.x] = d_Atom_Influence_nloc[type].grid_pos[atom][index];  //?
    }
    __syncthreads();

    if (index < ndc && n < ncol)
        d_xrc[n*ndc+index] = d_x[n*DMnd + shared_grid_pose[threadIdx.x]];                       ///?
}

__global__
void Vnl_gammaV(const min_SPARC_OBJ *d_SPARC, double *d_alpha, int ncol)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;

    if (index >= d_SPARC->IP_displ[d_SPARC->n_atom] || n >= ncol)
        return;

    /*if (index >= ncol * d_SPARC->IP_displ[d_SPARC->n_atom])
        return;*/

    int i = 1;                                                   //index of atom
    int type = 0;
    int temp = d_SPARC->nAtomv[0];
    while (index >= d_SPARC->IP_displ[i]) {     ///<=?
        i++;                                        ///i + 1
        if (i >= temp) {
            temp += d_SPARC->nAtomv[type];
            type++;
        }
    }


    int leftover = index - d_SPARC->IP_displ[i-1];
    //int nproj = d_SPARC->IP_displ[i+1] - d_SPARC->IP_displ[i];
    //leftover %= nproj;

    int lmax = d_SPARC->lmax[type];
    int lloc = d_SPARC->localPsd[type];

    int l = 0;
    int ldispl = 0;
    int revotfel = 0;
    for (l = 0; l <= lmax; l++) {
        if (l == lloc) {
            ldispl += (d_SPARC->ppl[type])[l];
            continue;
        }

        if (leftover - ( revotfel + d_SPARC->psd[type].ppl[l]*(2*l+1) ) > 0) {
            ldispl += (d_SPARC->ppl[type])[l];
            revotfel += (d_SPARC->ppl[type])[l] * (2 * l + 1);
        } else {
            leftover -= revotfel;
            int np = leftover / (2*l+1);
            d_alpha[index] *=  (d_SPARC->Gamma[type])[ldispl+np];
            return;
        }
    }
    //printf("2nd total: %f\n",(MPI_Wtime()-Start)*1e3);
}

__global__
void update(double *d_Hx, double *Vnlx, const ATOM_NLOC_INFLUENCE_OBJ *d_Atom_Influence_nloc,
            int ncol, int type, int atom, int DMnd)
{

    extern __shared__ double shared_grid_pose[];

    int ndc = d_Atom_Influence_nloc[type].ndc[atom];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;

    if (threadIdx.y == 0) {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < ndc)
            shared_grid_pose[threadIdx.x] = d_Atom_Influence_nloc[type].grid_pos[atom][index];  //?
    }
    __syncthreads();

    if (index < ndc && n < ncol)
        d_Hx[n*DMnd + shared_grid_pose[threadIdx.x]] += Vnlx[n*ndc+index];

}


#ifdef __cplusplus
}
#endif
