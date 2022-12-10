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


//int nprocs, rank, wrank;
int nprocs, rank;
int pdx, pdy;
int wrank;

int *list;
int *devIDs;
int ndev;
int my_dev;

double tinc;
double put_bw, get_bw, acc_bw;
double t_vput;

void test_int_array()
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
  int *ptr;
  int one;
  double zero = 0.0;

  pdx = 2;
  pdy = 2;

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
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Set_device(g_a, 1);
  NGA_Allocate(g_a);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  set_device(&my_dev);
  device_malloc((void**)&tbuf,(int)(nsize*sizeof(int)));
  buf = (int*)tbuf;

  for (n=0; n<NLOOP; n++) {
    GA_Zero(g_a);
    GA_Fill(g_a,&zero);
    ld = (hi[1]-lo[1]+1);
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
    /* copy data to global array */
    NGA_Put(g_a, lo, hi, buf, &ld);
    GA_Sync();
    NGA_Distribution(g_a,rank,tlo,thi);

    if (rank == 0 && n == 0) printf("Completed NGA_Distribution\n");
    if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
      int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
      tbuf = (int*)malloc(tnelem*sizeof(int));
      NGA_Access(g_a,tlo,thi,&ptr,&tld);
      for (i=0; i<tnelem; i++) tbuf[i] = 0;
      memcpyD2H(tbuf, ptr, tnelem*sizeof(int));
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        for (jj=tlo[1]; jj<=thi[1]; jj++) {
          j = jj-tlo[1];
          idx = i*tld+j;
        }
      }
      NGA_Release(g_a,tlo,thi);
      free(tbuf);
    }
    GA_Sync();

    /* zero out local buffer */
    int *tbuf = (int*)malloc(nsize*sizeof(int));
    for (i=0; i<nsize; i++) tbuf[i] = 0;
    memcpyH2D(buf, tbuf, nsize*sizeof(int));
    free(tbuf);

    /* copy data from global array to local buffer */
    NGA_Get(g_a, lo, hi, buf, &ld);
    GA_Sync();

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
        }
      }
      free(tbuf);
    }

    /* reset values in buf */
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

    /* accumulate data to global array */
    one = 1;
    NGA_Acc(g_a, lo, hi, buf, &ld, &one);
    GA_Sync();
    /* reset values in buf */
    if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
      int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
      tbuf = (int*)malloc(tnelem*sizeof(int));
      for (i=0; i<nelem; i++) tbuf[i] = 0;
      memcpyH2D(buf, tbuf, tnelem*sizeof(int));
      free(tbuf);
    }

    NGA_Get(g_a, lo, hi, buf, &ld);
    GA_Sync();
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
        }
      }
      free(tbuf);
    }
  }

  device_free(buf);
  GA_Destroy(g_a);
}

int main(int argc, char **argv) {
  int i;
  
  MPI_Init(&argc, &argv);
  MA_init(C_DBL, 2000000, 2000000);
  GA_Initialize();

  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   
  // wrank = rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);

  list = (int*)malloc(nprocs*sizeof(int));
  devIDs = (int*)malloc(nprocs*sizeof(int));
  NGA_Device_host_list(list, devIDs, &ndev, NGA_Pgroup_get_default());

  for (i=0; i<ndev; i++) {
     if (rank == list[i]) {
       my_dev = devIDs[i];
       break;
     }
  }

  test_int_array();

  free(list);
  free(devIDs);
  
  GA_Terminate();
  if (rank == 0) printf("Completed GA terminate\n");
  MPI_Finalize();
}
