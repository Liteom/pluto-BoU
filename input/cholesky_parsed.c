#include "../include/cholesky.h"
#include "../include/polybench.h"
#include <math.h>

void kernel_cholesky(int n, DATA_TYPE POLYBENCH_1D(p, N, n),
                     DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
  int i, j, k;

  DATA_TYPE x;

#pragma scop
  for (i = 0; i < _PB_N; ++i) {
    x = A[i][i];                          // S1
    for (j = 0; j <= i - 1; ++j)
      x = x - A[i][j] * A[i][j];       // S2
    p[i] = 1.0 / sqrt(x);                 // S3
    for (j = i + 1; j < _PB_N; ++j) {
      x = A[i][j];                        // S4
      for (k = 0; k <= i - 1; ++k)
        x = x - A[j][k] * A[i][k];     // S5
      A[j][i] = x * p[i];                 // S6
    }
  }
#pragma endscop
}
