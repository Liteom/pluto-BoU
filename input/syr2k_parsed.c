#include "../include/syr2k.h"
#include "../include/polybench.h"
#include <math.h>

void kernel_syr2k(int ni, int nj,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni),
		  DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		  DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
{
  int i, j, k;

#pragma scop
  /*    C := alpha*A*B' + alpha*B*A' + beta*C */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NI; j++)
      C[i][j] *= beta;
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NI; j++)
      for (k = 0; k < _PB_NJ; k++)
	{
	  C[i][j] += alpha * A[i][k] * B[j][k];
	  C[i][j] += alpha * B[i][k] * A[j][k];
	}
#pragma endscop

}

