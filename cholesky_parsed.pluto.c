#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#include "../include/cholesky.h"
#include "../include/polybench.h"
#include <math.h>

void kernel_cholesky(int n, DATA_TYPE POLYBENCH_1D(p, N, n),
                     DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
  int i, j, k;

  DATA_TYPE x[N];

  int t1, t2, t3, t4, t5;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if (_PB_N >= 1) {
  lbp=0;
  ubp=_PB_N-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5)
  for (t2=lbp;t2<=ubp;t2++) {
    x[t2] = A[t2][t2];;
  }
  p[0] = 1.0 / sqrt(x[0]);;
  if (_PB_N >= 2) {
    x[0] = A[0][1];;
    A[1][0] = x[0] * p[0];;
    x[1] = x[1] - A[1][0] * A[1][0];;
  }
  if (_PB_N >= 3) {
    p[1] = 1.0 / sqrt(x[1]);;
    x[0] = A[0][2];;
    A[2][0] = x[0] * p[0];;
    x[2] = x[2] - A[2][0] * A[2][0];;
  }
  for (t2=3;t2<=_PB_N-1;t2++) {
    if (t2%2 == 0) {
      p[(t2/2)] = 1.0 / sqrt(x[(t2/2)]);;
    }
    lbp=ceild(t2+1,2);
    ubp=t2-1;
#pragma omp parallel for private(lbv,ubv,t4,t5)
    for (t3=lbp;t3<=ubp;t3++) {
      x[(t2-t3)] = A[(t2-t3)][t3];;
      for (t4=0;t4<=t2-t3-1;t4++) {
        x[(t2-t3)] = x[(t2-t3)] - A[t3][t4] * A[(t2-t3)][t4];;
      }
      A[t3][(t2-t3)] = x[(t2-t3)] * p[(t2-t3)];;
      x[t3] = x[t3] - A[t3][(t2-t3)] * A[t3][(t2-t3)];;
    }
    x[0] = A[0][t2];;
    A[t2][0] = x[0] * p[0];;
    x[t2] = x[t2] - A[t2][0] * A[t2][0];;
  }
  for (t2=_PB_N;t2<=2*_PB_N-3;t2++) {
    if (t2%2 == 0) {
      p[(t2/2)] = 1.0 / sqrt(x[(t2/2)]);;
    }
    lbp=ceild(t2+1,2);
    ubp=_PB_N-1;
#pragma omp parallel for private(lbv,ubv,t4,t5)
    for (t3=lbp;t3<=ubp;t3++) {
      x[(t2-t3)] = A[(t2-t3)][t3];;
      for (t4=0;t4<=t2-t3-1;t4++) {
        x[(t2-t3)] = x[(t2-t3)] - A[t3][t4] * A[(t2-t3)][t4];;
      }
      A[t3][(t2-t3)] = x[(t2-t3)] * p[(t2-t3)];;
      x[t3] = x[t3] - A[t3][(t2-t3)] * A[t3][(t2-t3)];;
    }
  }
  if (_PB_N >= 2) {
    p[(_PB_N-1)] = 1.0 / sqrt(x[(_PB_N-1)]);;
  }
}
}
