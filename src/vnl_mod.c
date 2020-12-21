#include "vnl_mod.h"
#include <assert.h>
#include <math.h>
#include <omp.h>
//#include "finalization.h"

void interface(const SPARC_OBJ *pSPARC, min_SPARC_OBJ* min_SPARC)
{
    min_SPARC->n_atom = pSPARC->n_atom;
    min_SPARC->dV = pSPARC->dV;
    min_SPARC->Ntypes = pSPARC->Ntypes;

    min_SPARC->localPsd = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->nAtomv   = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->IP_displ = (int *)malloc( sizeof(int) * (pSPARC->n_atom+1));
    //copy
    memcpy(min_SPARC->localPsd,pSPARC->localPsd, sizeof(int)*pSPARC->Ntypes);
    memcpy(min_SPARC->nAtomv,pSPARC->nAtomv, sizeof(int)*pSPARC->Ntypes);
    memcpy(min_SPARC->IP_displ,pSPARC->IP_displ, sizeof(int)*(pSPARC->n_atom+1));


    min_SPARC->lmax = (int *) malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->partial_sum = (int *) malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->ppl = (int **)malloc( sizeof(int*) * pSPARC->Ntypes );
    min_SPARC->Gamma = (double **)malloc( sizeof(double*) * pSPARC->Ntypes );

    for(int i = 0; i < pSPARC->Ntypes; i++)
    {
        min_SPARC->lmax[i] = pSPARC->psd[i].lmax;
        min_SPARC->ppl[i] = (int*) malloc( sizeof(int) * (pSPARC->psd[i].lmax+1) );
        int ppl_sum = 0;
        for (int j = 0; j <= pSPARC->psd[i].lmax; j++)
        {
            (min_SPARC->ppl[i])[j] = pSPARC->psd[i].ppl[j];
            ppl_sum += pSPARC->psd[i].ppl[j];
	    min_SPARC->partial_sum[j] = ppl_sum;
        }

        min_SPARC->Gamma[i] = (double*) malloc( sizeof(double) * ppl_sum );
        memcpy(min_SPARC->Gamma[i],pSPARC->psd[i].Gamma,sizeof(double) * ppl_sum);
    }
}

void free_min_SPARC(min_SPARC_OBJ* min_SPARC)
{
    free(min_SPARC->localPsd);
    free(min_SPARC->nAtomv);
    free(min_SPARC->IP_displ);

    free(min_SPARC->lmax);
    free(min_SPARC->partial_sum);
    for (int i = 0; i < min_SPARC->Ntypes; i++) 
    {
 	free(min_SPARC->ppl[i]);
	free(min_SPARC->Gamma[i]);
    }
    free(min_SPARC->ppl);
    free(min_SPARC->Gamma);
	
    free(min_SPARC);
}

void Vnl_mod(const min_SPARC_OBJ *pSPARC, const int DMnd, const ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
                  const NLOC_PROJ_OBJ *nlocProj, const int ncol, double *x, double *Hx, MPI_Comm comm)
{
    /* compute nonlocal operator times vector(s) */   

    double *alpha;
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
    // |pSPARC->IP_displ[pSPARC->n_atom]|  = n_atom * nproj(of each atom)
	 
    /*first find inner product*/
    //double Start = MPI_Wtime();
    //#pragma omp parallel for collapse(2)
    for (int type = 0; type < pSPARC->Ntypes; type++) {

        if (! nlocProj[type].nproj) continue; // this is typical for hydrogen
	
	#pragma omp parallel for
        for (int atom = 0; atom < Atom_Influence_nloc[type].n_atom; atom++) {
            int ndc = Atom_Influence_nloc[type].ndc[atom];
            int atom_index = Atom_Influence_nloc[type].atom_index[atom];
            double *x_rc = (double *)malloc( ndc * ncol * sizeof(double));

            for (int n = 0; n < ncol; n++) {
                for (int i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = x[n*DMnd + Atom_Influence_nloc[type].grid_pos[atom][i]];
                }
            }
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nlocProj[type].nproj, ncol, ndc,
                pSPARC->dV, nlocProj[type].Chi[atom], ndc, x_rc, ndc, 1.0,
	     	alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[type].nproj);

            free(x_rc);
        }
    }
    //printf("1st total: %f\n",(MPI_Wtime()-Start)*1e3);


    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, comm);
    }


/*
    // go over all atoms and multiply gamma_Jl to the inner product
    int natom = 0;
    //Start = MPI_Wtime();
    for (int type = 0; type < pSPARC->Ntypes; type++) {
        int lloc = pSPARC->localPsd[type];
        int lmax = pSPARC->lmax[type]; 
	//#pragma omp parallel for
        for (int atom = 0; atom < pSPARC->nAtomv[type]; atom++) {     
	    int count = 0;
            int start_index = pSPARC->IP_displ[natom];
	    for (int n = 0; n < ncol; n++) {
                int ldispl = 0;
                for (int l = 0; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += (pSPARC->ppl[type])[l];                         
                        continue;
                    }
                    for (int np = 0; np < (pSPARC->ppl[type])[l]; np++) {
                        for (int m = -l; m <= l; m++) {
                            alpha[count+start_index] *= (pSPARC->Gamma[type])[ldispl+np];
                        }
                    }
                    ldispl += (pSPARC->ppl[type])[l];
                }
            }
        }
	natom += pSPARC->nAtomv[type];
    }
    //printf("2nd total: %f\n",(MPI_Wtime()-Start)*1e3);
*/
   
    int count = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int lloc = pSPARC->localPsd[ityp];
        int lmax = pSPARC->lmax[ityp];
        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (int n = 0; n < ncol; n++) {
                int ldispl = 0;
                for (int l = 0; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += (pSPARC->ppl[ityp])[l];
                        continue;
                    }
                    for (int np = 0; np < (pSPARC->ppl[ityp])[l]; np++) {
                        for (int m = -l; m <= l; m++) {
                            alpha[count++] *= (pSPARC->Gamma[ityp])[ldispl+np];
                        }
                    }
                    ldispl += (pSPARC->ppl[ityp])[l];
                }
            }
        }
    }






    // multiply the inner product and the nonlocal projector
    //Start = MPI_Wtime();
    for (int type = 0; type < pSPARC->Ntypes; type++) {
        if (! nlocProj[type].nproj) continue; // this is typical for hydrogen
	    
	//#pragma omp parallel for
        for (int atom = 0; atom < Atom_Influence_nloc[type].n_atom; atom++) {
            int ndc = Atom_Influence_nloc[type].ndc[atom];
            int atom_index = Atom_Influence_nloc[type].atom_index[atom];
            double *Vnlx = (double *)malloc( ndc * ncol * sizeof(double));
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[type].nproj, 1.0, nlocProj[type].Chi[atom], ndc,
                        alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[type].nproj, 0.0, Vnlx, ndc);
            for (int n = 0; n < ncol; n++) {
                for (int i = 0; i < ndc; i++) {
                    Hx[n*DMnd + Atom_Influence_nloc[type].grid_pos[atom][i]] += Vnlx[n*ndc+i];
                }
            }
            free(Vnlx);
        }
    }
    //printf("3rd total: %f\n",(MPI_Wtime()-Start)*1e3);
    
    
    free(alpha);
}






void test_vnl(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
                  NLOC_PROJ_OBJ *nlocProj, int ncol, double *x, double *Hx, MPI_Comm comm, double*hx)
{

    min_SPARC_OBJ *min_SPARC = (min_SPARC_OBJ*) malloc(sizeof(min_SPARC_OBJ));
    interface(pSPARC, min_SPARC);
    double t1 = MPI_Wtime();
    Vnl_mod(min_SPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, x, hx, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    double final = (t2-t1)*1e3;
    printf("total time :%f\n", final);

    double local_err;
    int err_count = 0;
    for (int ix = 0; ix < DMnd*ncol; ix++)
    {
        local_err = fabs(hx[ix] - Hx[ix]) / fabs(Hx[ix]);
        // Consider a relative error of 1e-10 to guard against floating point rounding
        if ((local_err > 1e-10) || isnan(local_err))
        {
            //printf("At index %d: %.15f vs. %.15f\n", ix, fabs(psi_new[ix]), fabs(psi_chk[ix]));
            err_count = err_count + 1;
        }
    }


    printf("There are %d errors out of %d entries!\n", err_count, ncol*DMnd);

    //Free_SPARC(pSPARC);
    free_min_SPARC(min_SPARC);

    exit(0);


}
