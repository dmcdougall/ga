#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#if defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

/*
#define BLOCK1 1024*1024
#define BLOCK1 65536
*/
#define BLOCK1 65530
#define DIMSIZE 2048
#define SMLDIM 256
#define MAXCOUNT 10000
#define MAX_FACTOR 256
#define NLOOP 10

void set_device(int* devid) {
  #if defined(ENABLE_CUDA)
  cudaSetDevice(*devid);
  #elif defined(ENABLE_HIP)
  hipSetDevice(*devid);
  #endif    
}

void device_malloc(void **buf, size_t bsize){
  #if defined(ENABLE_CUDA)
  cudaMalloc(buf, bsize);
  #elif defined(ENABLE_HIP)
  hipMalloc(buf, bsize);
  #endif    
}

void memcpyH2D(void* dbuf, void* sbuf, size_t bsize){
  #if defined(ENABLE_CUDA)
  cudaMemcpy(dbuf, sbuf, bsize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  #elif defined(ENABLE_HIP)
  hipMemcpy(dbuf, sbuf, bsize, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  #endif    
}

void memcpyD2H(void* dbuf, void* sbuf, size_t bsize){
  #if defined(ENABLE_CUDA)
  cudaMemcpy(dbuf, sbuf, bsize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  #elif defined(ENABLE_HIP)
  hipMemcpy(dbuf, sbuf, bsize, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  #endif    
}

void device_free(void *buf) {
  #if defined(ENABLE_CUDA)
  cudaFree(buf);
  #elif defined(ENABLE_HIP)
  hipFree(buf);
  #endif   
}


void factor(int p, int *idx, int *idy) {
  int i, j;                              
  int ip, ifac, pmax;                    
  int prime[MAX_FACTOR];                 
  int fac[MAX_FACTOR];                   
  int ix, iy;                            
  int ichk;                              

  i = 1;

 //find all prime numbers, besides 1, less than or equal to the square root of p
  ip = (int)(sqrt((double)p))+1;

  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }

 //find all prime factors of p
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }

 //when p itself is prime
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }


 //find two factors of p of approximately the same size
  *idx = 1;
  *idy = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = *idx;
    iy = *idy;
    if (ix <= iy) {
      *idx = fac[i]*(*idx);
    } else {
      *idy = fac[i]*(*idy);
    }
  }
}

//int nprocs, rank, wrank;
int nprocs, rank;
int pdx, pdy;
int wrank;

int *list;
int *devIDs;
int ndev;
int my_dev;

double tget, tacc, tinc;
int get_cnt,put_cnt,acc_cnt;
double put_bw, get_bw, acc_bw;
double t_vput;

void test_int_array(int on_device, int local_buf_on_device)
{
  int g_a;
  int i, j, ii, jj, n, idx;
  int ipx, ipy;
  int isx, isy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  int *buf;
  int *tbuf;
  int nelem;
  int ld;
  int g_ok, p_ok, a_ok;
  int *ptr;
  int one;
  double tbeg;
  double zero = 0.0;
  int ok;

  tget = 0.0;
  tacc = 0.0;
  put_cnt = 0;
  get_cnt = 0;
  acc_cnt = 0;
  p_ok = 1;
  g_ok = 1;
  a_ok = 1;

  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/pdx;
  yinc = DIMSIZE/pdy;
  ipx = rank%pdx;
  ipy = (rank-ipx)/pdx;
  isx = (ipx+1)%pdx;
  isy = (ipy+1)%pdy;
  /* Guarantee some data exchange between nodes */
  lo[0] = isx*xinc;
  lo[1] = isy*yinc;
  if (isx<pdx-1) {
    hi[0] = (isx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  if (isy<pdy-1) {
    hi[1] = (isy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);

  /* create a global array and initialize it to zero */
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  if (local_buf_on_device) {
    void *tbuf;
    set_device(&my_dev);
    device_malloc(&tbuf,(int)(nsize*sizeof(int)));
    buf = (int*)tbuf;
  } else {
    buf = (int*)malloc(nsize*sizeof(int));
  }

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    GA_Fill(g_a,&zero);
    ld = (hi[1]-lo[1]+1);
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj = lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            tbuf[idx] = ii*DIMSIZE+jj;
          }
        }
        memcpyH2D(buf, tbuf, tnelem*sizeof(int));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = ii*DIMSIZE+jj;
        }
      }
    }
    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, lo, hi, buf, &ld);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    tbeg = GA_Wtime();
    NGA_Distribution(g_a,rank,tlo,thi);
#if 1
    if (rank == 0 && n == 0) printf("Completed NGA_Distribution\n");
    if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
      int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
      if (on_device) {
        tbuf = (int*)malloc(tnelem*sizeof(int));
        NGA_Access(g_a,tlo,thi,&ptr,&tld);
        for (i=0; i<tnelem; i++) tbuf[i] = 0;
        memcpyD2H(tbuf, ptr, tnelem*sizeof(int));
      } else {
        NGA_Access(g_a,tlo,thi,&tbuf,&tld);
      }
      ok = 1;
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        for (jj=tlo[1]; jj<=thi[1]; jj++) {
          j = jj-tlo[1];
          idx = i*tld+j;
          if (tbuf[idx] != ii*DIMSIZE+jj) {
            if (ok) printf("p[%d] (%d,%d) (put) expected: %d actual[%d]: %d\n",
                rank,ii,jj,ii*DIMSIZE+jj,idx,tbuf[idx]);
            ok = 0;
          }
        }
      }
      if (!ok) {
        printf("Mismatch found for put on process %d after Put\n",rank);
      } else if (n==0 && rank==0 && ok) {
        printf("Access function is okay\n");
      }
      NGA_Release(g_a,tlo,thi);
      if (on_device) {
        free(tbuf);
      }
    }
    tbeg = GA_Wtime();
    GA_Sync();
#endif

    /* zero out local buffer */
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      for (i=0; i<nsize; i++) tbuf[i] = 0;
      memcpyH2D(buf, tbuf, nsize*sizeof(int));
      free(tbuf);
    } else {
      for (i=0; i<nsize; i++) buf[i] = 0;
    }

    /* copy data from global array to local buffer */
    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();

    tbeg = GA_Wtime();
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (i=0; i<tnelem; i++) tbuf[i] = 0;
        memcpyD2H(tbuf, buf, tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj=lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            if (tbuf[idx] != ii*DIMSIZE+jj) {
              if (g_ok) printf("p[%d] (%d,%d) (get) expected: %d"
                  " actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,idx,tbuf[idx]);
              g_ok = 0;
            }
          }
        }
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (buf[idx] != ii*DIMSIZE+jj) {
            if (g_ok) printf("p[%d] (%d,%d) (get) expected: %d"
                " actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,idx,buf[idx]);
            g_ok = 0;
          }
        }
      }
    }

    /* reset values in buf */
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj = lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            tbuf[idx] = ii*DIMSIZE+jj;
          }
        }
        memcpyH2D(buf, tbuf, tnelem*sizeof(int));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = ii*DIMSIZE+jj;
        }
      }
    }

    /* accumulate data to global array */
    one = 1;
    tbeg = GA_Wtime();
    NGA_Acc(g_a, lo, hi, buf, &ld, &one);
    tacc += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    acc_cnt += nsize;
    GA_Sync();
    tbeg = GA_Wtime();
    /* reset values in buf */
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (i=0; i<nelem; i++) tbuf[i] = 0;
        memcpyH2D(buf, tbuf, tnelem*sizeof(int));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = 0;
        }
      }
    }

    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    tbeg = GA_Wtime();
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (i=0; i<tnelem; i++) tbuf[i] = 0;
        memcpyD2H(tbuf, buf, tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj=lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            if (tbuf[idx] != 2*(ii*DIMSIZE+jj)) {
              if (a_ok) printf("p[%d] (%d,%d) (acc) expected: %d"
                  " actual[%d]: %d device: %d\n",rank,ii,jj,
                  2*(ii*DIMSIZE+jj),idx,tbuf[idx],on_device);
              a_ok = 0;
            }
          }
        }
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (buf[idx] != 2*(ii*DIMSIZE+jj)) {
            if (a_ok) printf("p[%d] (%d,%d) (acc) expected: %d"
                " actual[%d]: %d device: %d\n",rank,ii,jj,
                2*(ii*DIMSIZE+jj),idx,buf[idx],on_device);
            a_ok = 0;
          }
        }
      }
    }
  }

  if (local_buf_on_device) {
    device_free(buf);
  } else {
    free(buf);
  }
  tbeg = GA_Wtime();
  GA_Destroy(g_a);

  if (!g_ok) {
    printf("Mismatch found for get on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }
  if (!a_ok) {
    printf("Mismatch found on process %d after Acc\n",rank);
  } else {
    if (rank == 0) printf("Acc is okay\n");
  }

  GA_Igop(&put_cnt, 1, "+");
  GA_Igop(&get_cnt, 1, "+");
  GA_Igop(&acc_cnt, 1, "+");
  GA_Dgop(&tget, 1, "+");
  GA_Dgop(&tacc, 1, "+");
  get_bw = (double)(get_cnt*sizeof(int))/tget;
  acc_bw = (double)(acc_cnt*sizeof(int))/tacc;
}

int main(int argc, char **argv) {

  int g_a;
  double *rbuf;
  double one_r;
  double t_sum;
  int zero = 0;
  int icnt;
  double tbeg;
  int local_buf_on_device;
  int i;
  
  MPI_Init(&argc, &argv);

  int myrank, mysize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &mysize);

  MA_init(C_DBL, 2000000, 2000000);
  GA_Initialize();

  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   
  // wrank = rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);

  int nodeid = GA_Cluster_nodeid();

  list = (int*)malloc(nprocs*sizeof(int));
  devIDs = (int*)malloc(nprocs*sizeof(int));
  NGA_Device_host_list(list, devIDs, &ndev, NGA_Pgroup_get_default());

  local_buf_on_device = 0;
  for (i=0; i<ndev; i++) {
     if (rank == list[i]) {
       local_buf_on_device = 1;
       my_dev = devIDs[i];
       break;
     }
  }

  /* Divide matrix up into pieces that are owned by each processor */
  factor(nprocs, &pdx, &pdy);
  if (rank == 0) printf("  Testing integer array on device, local buffer on device\n");
  test_int_array(1,local_buf_on_device);

  free(list);
  free(devIDs);
  
  GA_Terminate();
  if (rank == 0) printf("Completed GA terminate\n");
  MPI_Finalize();
}
