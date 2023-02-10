#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#include <hip/hip_runtime.h>

#define BLOCK1 65530
#define DIMSIZE 2048
#define SMLDIM 256
#define MAXCOUNT 10000
#define MAX_FACTOR 256
#define NLOOP 3

int nprocs, rank;
int pdx, pdy;
int wrank;
int my_dev;

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
  int *ptr;
  int one;
  double zero = 0.0;
  int ok;

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
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (int*)malloc(nsize*sizeof(int));

  for (n=0; n<NLOOP; n++) {
    GA_Zero(g_a);
    GA_Fill(g_a,&zero);
    ld = (hi[1]-lo[1]+1);
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = ii*DIMSIZE+jj;
        }
      }
    /* copy data to global array */
    NGA_Put(g_a, lo, hi, buf, &ld);
    GA_Sync();
    NGA_Distribution(g_a,rank,tlo,thi);
    if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
      int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
      tbuf = (int*)malloc(tnelem*sizeof(int));
      NGA_Access(g_a,tlo,thi,&ptr,&tld);
      for (i=0; i<tnelem; i++) tbuf[i] = 0;
      hipMemcpy(tbuf, ptr, tnelem*sizeof(int), hipMemcpyDeviceToHost);
      hipDeviceSynchronize();
      ok = 1;
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        for (jj=tlo[1]; jj<=thi[1]; jj++) {
          j = jj-tlo[1];
          idx = i*tld+j;
          if (tbuf[idx] != ii*DIMSIZE+jj) {
            if (ok && (rank == 0) && (n == 2)) printf("p[%d] (%d,%d) (put) expected: %d actual[%d]: %d, n: %d\n", rank,ii,jj,ii*DIMSIZE+jj,idx,tbuf[idx],n);
            ok = 0;
          }
        }
      }
      NGA_Release(g_a,tlo,thi);
      free(tbuf);
    }
    GA_Sync();
  }

  free(buf);
  GA_Destroy(g_a);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MA_init(C_DBL, 2000000, 2000000);
  GA_Initialize();

  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  my_dev = rank;

  pdx = 2;
  pdy = 2;
  test_int_array(1,0);

  GA_Terminate();
  MPI_Finalize();
}
