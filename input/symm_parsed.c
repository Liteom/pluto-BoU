#include "../include/polybench.h"
#include "../include/symm.h"
#include <math.h>

void kernel_symm(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
                 DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                 DATA_TYPE POLYBENCH_2D(A, NJ, NJ, nj, nj),
                 DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj)) {
  int i, j, k;
  DATA_TYPE acc;

#pragma scop
  /*  C := alpha*A*B + beta*C, A is symetric */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++) {
      acc = 0;
      for (k = 0; k < j - 1; k++) {
        C[k][j] += alpha * A[k][i] * B[i][j];
        acc += B[k][j] * A[k][i];
      }
      C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc;
    }
#pragma endscop
}
