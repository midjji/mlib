#pragma once

namespace hartley {

typedef double Ematrix[3][3];
typedef double Matches[][3];
typedef double Matches_5[5][3];

void compute_E_matrices(Matches q, Matches qp, Ematrix Ematrices[10], int &nroots, bool optimized);

}
