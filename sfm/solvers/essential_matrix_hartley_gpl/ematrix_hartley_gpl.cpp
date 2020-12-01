#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#include "ematrix_hartley_gpl.h"

namespace hartley {
// Copyright Richard Hartley, 2010

//--------------------------------------------------------------------------
// LICENSE INFORMATION
//
// 1.  For academic/research users:
//
// This program is free for academic/research purpose:   you can redistribute
// it and/or modify  it under the terms of the GNU General Public License as 
// published by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Under this academic/research condition,  this program is distributed in 
// the hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program. If not, see <http://www.gnu.org/licenses/>.
//
// 2.  For commercial OEMs, ISVs and VARs:
// 
// For OEMs, ISVs, and VARs who distribute/modify/use this software 
// (binaries or source code) with their products, and do not license and 
// distribute their source code under the GPL, please contact NICTA 
// (www.nicta.com.au), and NICTA will provide a flexible OEM Commercial 
// License. 
//
//---------------------------------------------------------------------------

// #define RH_DEBUG

// hidden6.h

// Dimensions of the matrices that we will be using
const int Nrows = 10;
const int Ncols = 10;
const int Maxdegree = 20;

// For holding polynomials of matrices
typedef double PolyMatrix [Nrows][Ncols][Maxdegree+1];
typedef int PolyDegree    [Nrows][Ncols];
typedef double Ematrix[3][3];
typedef double Matches[][3];
typedef double Matches_5[5][3];

// We need to be able to solve matrix equations up to this dimension
typedef double BMatrix[Maxdegree+1][Maxdegree+1];

// Forward declarations
// void print_equation_set (EquationSet A, int maxdegree = 3);
void print_polymatrix   (PolyMatrix A, int maxdegree = 3);

void polyquotient (
	double *a, int sa, 
	double *b, double *t, int sb, 
	double *q, int &sq,
	BMatrix B, int &current_size
	);

void find_polynomial_determinant (
        PolyMatrix &Q,
        PolyDegree deg,
        int rows[Nrows],  // This keeps the order of rows pivoted on.
	int dim = Nrows
        );

void det_preprocess_6pt (
	PolyMatrix &Q, 
	PolyDegree degree, 
        int n_zero_roots	// Number of roots known to be zero
	);

void do_scale (
	PolyMatrix &Q, 
	PolyDegree degree, 
        double &scale_factor,	// Factor that x is multiplied by
        bool degree_by_row,
	int dim = Nrows
	);
	
// Ematrix_5pt.cc //////////////////////////////


// Degree of the polynomial
const int PolynomialDegree = 10;

// Some forward declarations
class poly4_1;
class poly4_2;
class poly4_3;
class EmatrixSet;

typedef double EquationSet[5][10][10];
typedef double Polynomial[PolynomialDegree+1];

void print_equation_set (EquationSet A, int maxdegree = 3);
void test_E_matrix ( const double E[3][3]);

class poly4_3
   {
   protected :
      double A_[4][4][4];

   public :
      poly4_3 operator + (poly4_3);
      void operator += (poly4_3);
      poly4_3 operator - (poly4_3);
      poly4_3 operator * (double);
      double &operator () (int i, int j, int k) {return A_[i][j][k]; }

      void print ()
         {
         for (int i=0; i<4; i++) 
            {
            for (int j=0; j<4; j++) 
               {
               for (int k=0; k<4; k++) 
	          printf ("%12.3e ", A_[i][j][k]);
               printf ("\n");
               }
            printf ("\n");
            }
         }

      void clear() 
         { 
         for (int i=0; i<4; i++) 
            for (int j=0; j<4; j++) 
               for (int k=0; k<4; k++) 
               A_[i][j][k] = 0.0;
         }
   };

class poly4_2 
   {
   protected :
      double A_[4][4];

   public :
      poly4_3 operator * (poly4_1);
      poly4_2 operator + (poly4_2);
      void operator += (poly4_2);
      poly4_2 operator - (poly4_2);
      double &operator () (int i, int j) { return A_[i][j]; }

      void clear() 
         { 
         for (int i=0; i<4; i++) 
            for (int j=0; j<4; j++) 
               A_[i][j] = 0.0;
         }

      void print ()
         {
         for (int i=0; i<4; i++) 
            {
            for (int j=0; j<4; j++) 
	       printf ("%12.3e ", A_[i][j]);
            printf ("\n");
            }
         }
   };

class poly4_1
   {
   protected :
      double A_[4];

   public :

      // Constructors
      poly4_1(){};
      poly4_1 (double w, double x, double y, double z) 
	 { A_[0] = w; A_[1] = x; A_[2] = y; A_[3] = z; }
      ~poly4_1 () {};

      // Operators
      poly4_2 operator * (poly4_1);
      poly4_1 operator + (poly4_1);
      poly4_1 operator - (poly4_1);
      double &operator () (int i) { return A_[i]; }

      void print ()
         {
         for (int i=0; i<4; i++) 
	    printf ("%12.3e ", A_[i]);
         printf ("\n");
         }
   };

class EmatrixSet
   {
   protected :
      poly4_1 E_[3][3];

   public :

      EmatrixSet () {};
      ~EmatrixSet() {};

      poly4_1 &operator () (int i, int j) { return E_[i][j]; }

      void print ()
         {
         for (int i=0; i<4; i++)
            {
            for (int j=0; j<3; j++)
               {
               for (int k=0; k<3; k++)
	          printf ("%12.3e ", E_[j][k](i));
               printf ("\n");
               }
            printf ("\n");
            }
         }
   };

//=============================================================================
//           Various operators on the polynomial classes
//=============================================================================

poly4_2 poly4_1::operator * (poly4_1 p2)
   {
   poly4_1 &p1 = *this;
   poly4_2 prod;

   prod(0,0)  = p1(0)*p2(0);
   prod(0,1)  = p1(0)*p2(1);
   prod(0,2)  = p1(0)*p2(2);
   prod(0,3)  = p1(0)*p2(3);

   prod(0,1) += p1(1)*p2(0);
   prod(1,1)  = p1(1)*p2(1);
   prod(1,2)  = p1(1)*p2(2);
   prod(1,3)  = p1(1)*p2(3);

   prod(0,2) += p1(2)*p2(0);
   prod(1,2) += p1(2)*p2(1);
   prod(2,2)  = p1(2)*p2(2);
   prod(2,3)  = p1(2)*p2(3);

   prod(0,3) += p1(3)*p2(0);
   prod(1,3) += p1(3)*p2(1);
   prod(2,3) += p1(3)*p2(2);
   prod(3,3)  = p1(3)*p2(3);

   return prod;
   }

poly4_3 poly4_2::operator * (poly4_1 p2)
   {
   poly4_2 &p1 = *this;
   poly4_3 prod;

   prod(0,0,0)  = p1(0,0)*p2(0);
   prod(0,0,1)  = p1(0,0)*p2(1);
   prod(0,0,2)  = p1(0,0)*p2(2);
   prod(0,0,3)  = p1(0,0)*p2(3);

   prod(0,0,1) += p1(0,1)*p2(0);
   prod(0,1,1)  = p1(0,1)*p2(1);
   prod(0,1,2)  = p1(0,1)*p2(2);
   prod(0,1,3)  = p1(0,1)*p2(3);

   prod(0,0,2) += p1(0,2)*p2(0);
   prod(0,1,2) += p1(0,2)*p2(1);
   prod(0,2,2)  = p1(0,2)*p2(2);
   prod(0,2,3)  = p1(0,2)*p2(3);

   prod(0,0,3) += p1(0,3)*p2(0);
   prod(0,1,3) += p1(0,3)*p2(1);
   prod(0,2,3) += p1(0,3)*p2(2);
   prod(0,3,3)  = p1(0,3)*p2(3);

   prod(0,1,1) += p1(1,1)*p2(0);
   prod(1,1,1)  = p1(1,1)*p2(1);
   prod(1,1,2)  = p1(1,1)*p2(2);
   prod(1,1,3)  = p1(1,1)*p2(3);

   prod(0,1,2) += p1(1,2)*p2(0);
   prod(1,1,2) += p1(1,2)*p2(1);
   prod(1,2,2)  = p1(1,2)*p2(2);
   prod(1,2,3)  = p1(1,2)*p2(3);

   prod(0,1,3) += p1(1,3)*p2(0);
   prod(1,1,3) += p1(1,3)*p2(1);
   prod(1,2,3) += p1(1,3)*p2(2);
   prod(1,3,3)  = p1(1,3)*p2(3);

   prod(0,2,2) += p1(2,2)*p2(0);
   prod(1,2,2) += p1(2,2)*p2(1);
   prod(2,2,2)  = p1(2,2)*p2(2);
   prod(2,2,3)  = p1(2,2)*p2(3);

   prod(0,2,3) += p1(2,3)*p2(0);
   prod(1,2,3) += p1(2,3)*p2(1);
   prod(2,2,3) += p1(2,3)*p2(2);
   prod(2,3,3)  = p1(2,3)*p2(3);

   prod(0,3,3) += p1(3,3)*p2(0);
   prod(1,3,3) += p1(3,3)*p2(1);
   prod(2,3,3) += p1(3,3)*p2(2);
   prod(3,3,3)  = p1(3,3)*p2(3);

#ifdef RH_DEBUG
   printf ("In poly4_2 * poly4_1\n");
   printf ("poly4_2 = \n");
   p1.print();
   printf ("poly4_1 = \n");
   p2.print();
   printf ("poly4_2 * poly4_2 = \n");
   prod.print();
#endif

   return prod;
   }

poly4_3 poly4_3::operator * (double k)
   {
   poly4_3 &p1 = *this;
   poly4_3 prod;

   prod(0,0,0) = p1(0,0,0) * k;
   prod(0,0,1) = p1(0,0,1) * k;
   prod(0,0,2) = p1(0,0,2) * k;
   prod(0,0,3) = p1(0,0,3) * k;
   prod(0,1,1) = p1(0,1,1) * k;
   prod(0,1,2) = p1(0,1,2) * k;
   prod(0,1,3) = p1(0,1,3) * k;
   prod(0,2,2) = p1(0,2,2) * k;
   prod(0,2,3) = p1(0,2,3) * k;
   prod(0,3,3) = p1(0,3,3) * k;
   prod(1,1,1) = p1(1,1,1) * k;
   prod(1,1,2) = p1(1,1,2) * k;
   prod(1,1,3) = p1(1,1,3) * k;
   prod(1,2,2) = p1(1,2,2) * k;
   prod(1,2,3) = p1(1,2,3) * k;
   prod(1,3,3) = p1(1,3,3) * k;
   prod(2,2,2) = p1(2,2,2) * k;
   prod(2,2,3) = p1(2,2,3) * k;
   prod(2,3,3) = p1(2,3,3) * k;
   prod(3,3,3) = p1(3,3,3) * k;

   return prod;
   }

poly4_3 poly4_3::operator + (poly4_3 p2)
   {
   poly4_3 &p1 = *this;
   poly4_3 sum;

   sum(0,0,0) = p1(0,0,0) + p2(0,0,0);
   sum(0,0,1) = p1(0,0,1) + p2(0,0,1);
   sum(0,0,2) = p1(0,0,2) + p2(0,0,2);
   sum(0,0,3) = p1(0,0,3) + p2(0,0,3);
   sum(0,1,1) = p1(0,1,1) + p2(0,1,1);
   sum(0,1,2) = p1(0,1,2) + p2(0,1,2);
   sum(0,1,3) = p1(0,1,3) + p2(0,1,3);
   sum(0,2,2) = p1(0,2,2) + p2(0,2,2);
   sum(0,2,3) = p1(0,2,3) + p2(0,2,3);
   sum(0,3,3) = p1(0,3,3) + p2(0,3,3);
   sum(1,1,1) = p1(1,1,1) + p2(1,1,1);
   sum(1,1,2) = p1(1,1,2) + p2(1,1,2);
   sum(1,1,3) = p1(1,1,3) + p2(1,1,3);
   sum(1,2,2) = p1(1,2,2) + p2(1,2,2);
   sum(1,2,3) = p1(1,2,3) + p2(1,2,3);
   sum(1,3,3) = p1(1,3,3) + p2(1,3,3);
   sum(2,2,2) = p1(2,2,2) + p2(2,2,2);
   sum(2,2,3) = p1(2,2,3) + p2(2,2,3);
   sum(2,3,3) = p1(2,3,3) + p2(2,3,3);
   sum(3,3,3) = p1(3,3,3) + p2(3,3,3);

   return sum;
   }

void poly4_3::operator += (poly4_3 p2)
   {
   poly4_3 &p1 = *this;

   p1(0,0,0) += p2(0,0,0);
   p1(0,0,1) += p2(0,0,1);
   p1(0,0,2) += p2(0,0,2);
   p1(0,0,3) += p2(0,0,3);
   p1(0,1,1) += p2(0,1,1);
   p1(0,1,2) += p2(0,1,2);
   p1(0,1,3) += p2(0,1,3);
   p1(0,2,2) += p2(0,2,2);
   p1(0,2,3) += p2(0,2,3);
   p1(0,3,3) += p2(0,3,3);
   p1(1,1,1) += p2(1,1,1);
   p1(1,1,2) += p2(1,1,2);
   p1(1,1,3) += p2(1,1,3);
   p1(1,2,2) += p2(1,2,2);
   p1(1,2,3) += p2(1,2,3);
   p1(1,3,3) += p2(1,3,3);
   p1(2,2,2) += p2(2,2,2);
   p1(2,2,3) += p2(2,2,3);
   p1(2,3,3) += p2(2,3,3);
   p1(3,3,3) += p2(3,3,3);
   }

#if 0
void poly4_3::operator *= (double p2)
   {
   poly4_3 &p1 = *this;

   p1(0,0,0) *= p2;
   p1(0,0,1) *= p2;
   p1(0,0,2) *= p2;
   p1(0,0,3) *= p2;
   p1(0,1,1) *= p2;
   p1(0,1,2) *= p2;
   p1(0,1,3) *= p2;
   p1(0,2,2) *= p2;
   p1(0,2,3) *= p2;
   p1(0,3,3) *= p2;
   p1(1,1,1) *= p2;
   p1(1,1,2) *= p2;
   p1(1,1,3) *= p2;
   p1(1,2,2) *= p2;
   p1(1,2,3) *= p2;
   p1(1,3,3) *= p2;
   p1(2,2,2) *= p2;
   p1(2,2,3) *= p2;
   p1(2,3,3) *= p2;
   p1(3,3,3) *= p2;
   }
#endif

poly4_3 poly4_3::operator - (poly4_3 p2)
   {
   poly4_3 &p1 = *this;
   poly4_3 dif;

   dif(0,0,0) = p1(0,0,0) - p2(0,0,0);
   dif(0,0,1) = p1(0,0,1) - p2(0,0,1);
   dif(0,0,2) = p1(0,0,2) - p2(0,0,2);
   dif(0,0,3) = p1(0,0,3) - p2(0,0,3);
   dif(0,1,1) = p1(0,1,1) - p2(0,1,1);
   dif(0,1,2) = p1(0,1,2) - p2(0,1,2);
   dif(0,1,3) = p1(0,1,3) - p2(0,1,3);
   dif(0,2,2) = p1(0,2,2) - p2(0,2,2);
   dif(0,2,3) = p1(0,2,3) - p2(0,2,3);
   dif(0,3,3) = p1(0,3,3) - p2(0,3,3);
   dif(1,1,1) = p1(1,1,1) - p2(1,1,1);
   dif(1,1,2) = p1(1,1,2) - p2(1,1,2);
   dif(1,1,3) = p1(1,1,3) - p2(1,1,3);
   dif(1,2,2) = p1(1,2,2) - p2(1,2,2);
   dif(1,2,3) = p1(1,2,3) - p2(1,2,3);
   dif(1,3,3) = p1(1,3,3) - p2(1,3,3);
   dif(2,2,2) = p1(2,2,2) - p2(2,2,2);
   dif(2,2,3) = p1(2,2,3) - p2(2,2,3);
   dif(2,3,3) = p1(2,3,3) - p2(2,3,3);
   dif(3,3,3) = p1(3,3,3) - p2(3,3,3);

   return dif;
   }

poly4_2 poly4_2::operator + (poly4_2 p2)
   {
   poly4_2 &p1 = *this;
   poly4_2 sum;

   sum(0,0) = p1(0,0) + p2(0,0);
   sum(0,1) = p1(0,1) + p2(0,1);
   sum(0,2) = p1(0,2) + p2(0,2);
   sum(0,3) = p1(0,3) + p2(0,3);
   sum(1,1) = p1(1,1) + p2(1,1);
   sum(1,2) = p1(1,2) + p2(1,2);
   sum(1,3) = p1(1,3) + p2(1,3);
   sum(2,2) = p1(2,2) + p2(2,2);
   sum(2,3) = p1(2,3) + p2(2,3);
   sum(3,3) = p1(3,3) + p2(3,3);

   return sum;
   }

void poly4_2::operator += (poly4_2 p2)
   {
   poly4_2 &p1 = *this;

   p1(0,0) += p2(0,0);
   p1(0,1) += p2(0,1);
   p1(0,2) += p2(0,2);
   p1(0,3) += p2(0,3);
   p1(1,1) += p2(1,1);
   p1(1,2) += p2(1,2);
   p1(1,3) += p2(1,3);
   p1(2,2) += p2(2,2);
   p1(2,3) += p2(2,3);
   p1(3,3) += p2(3,3);
   }

poly4_2 poly4_2::operator - (poly4_2 p2)
   {
   poly4_2 &p1 = *this;
   poly4_2 dif;

   dif(0,0) = p1(0,0) - p2(0,0);
   dif(0,1) = p1(0,1) - p2(0,1);
   dif(0,2) = p1(0,2) - p2(0,2);
   dif(0,3) = p1(0,3) - p2(0,3);
   dif(1,1) = p1(1,1) - p2(1,1);
   dif(1,2) = p1(1,2) - p2(1,2);
   dif(1,3) = p1(1,3) - p2(1,3);
   dif(2,2) = p1(2,2) - p2(2,2);
   dif(2,3) = p1(2,3) - p2(2,3);
   dif(3,3) = p1(3,3) - p2(3,3);

   return dif;
   }

poly4_1 poly4_1::operator + (poly4_1 p2)
   {
   poly4_1 &p1 = *this;
   poly4_1 sum;

   sum(0) = p1(0) + p2(0);
   sum(1) = p1(1) + p2(1);
   sum(2) = p1(2) + p2(2);
   sum(3) = p1(3) + p2(3);

   return sum;
   }

poly4_1 poly4_1::operator - (poly4_1 p2)
   {
   poly4_1 &p1 = *this;
   poly4_1 dif;

   dif(0) = p1(0) - p2(0);
   dif(1) = p1(1) - p2(1);
   dif(2) = p1(2) - p2(2);
   dif(3) = p1(3) - p2(3);

   return dif;
   }

//=============================================================================

poly4_3 polydet4 (EmatrixSet E)
   {
   // Takes the determinant of a polynomial
   poly4_3 det = 
      (E(1,1)*E(2,2) - E(2,1)*E(1,2)) * E(0,0) +
      (E(2,1)*E(0,2) - E(0,1)*E(2,2)) * E(1,0) +
      (E(0,1)*E(1,2) - E(1,1)*E(0,2)) * E(2,0);

#ifdef RH_DEBUG
   printf ("Det =\n");
   det.print();
#endif

   return det;
   }

#define FULL_TRACE
#ifdef  FULL_TRACE
poly4_2 traceEEt (EmatrixSet E)
   {
   // Takes the trace of E E' -- returns a quadratic polynomial
   // Trace of product is the elementwise product of the elements

   poly4_2 tr = E(0,0) * E(0, 0) + E(0,1) * E(0, 1) + E(0,2) * E(0, 2) 
            + E(1,0) * E(1, 0) + E(1,1) * E(1, 1) + E(1,2) * E(1, 2) 
            + E(2,0) * E(2, 0) + E(2,1) * E(2, 1) + E(2,2) * E(2, 2);

#ifdef RH_DEBUG
   printf ("Trace is:\n");
   tr.print();
#endif

   return tr;
   }

#else

poly4_2 traceEEt (EmatrixSet E)
   {
   // We know that the trace has a simple form, provided that the
   // E-matrix basis is orthogonal.

   poly4_2 tr;  
   tr.clear();
   tr(0,0) = 1.0;
   tr(1,1) = 1.0;
   tr(2,2) = 1.0;
   tr(3,3) = 1.0;

   return tr;
   }

#endif

void mono_coeff (poly4_3 B, EquationSet A, int n)
   {
   // Extracts the monomial coefficients in x and y (with z = 1) from
   // a cubic homogeneous polynomial. Returns 4 vectors (degrees 0 to 3 in w)

   // Make some constants to make the code easier to read

   // Degrees of terms in w
   const int w0 = 0;
   const int w1 = 1;
   const int w2 = 2;
   const int w3 = 3;

   // Linear variables
   const int w = 0;
   const int x = 1;
   const int y = 2;
   const int z = 3;

   // Monomials
   const int xx  = 3;
   const int xy  = 4;
   const int yy  = 5;
   const int xxx = 6;
   const int xxy = 7;
   const int xyy = 8;
   const int yyy = 9;

   // Terms in w^0
   A[w0][n][ 0  ] = B(z, z, z);
   A[w0][n][ x  ] = B(x, z, z);
   A[w0][n][ y  ] = B(y, z, z);
   A[w0][n][ xx ] = B(x, x, z);
   A[w0][n][ yy ] = B(y, y, z);
   A[w0][n][ xy ] = B(x, y, z);
   A[w0][n][ xxx] = B(x, x, x);
   A[w0][n][ xxy] = B(x, x, y);
   A[w0][n][ xyy] = B(x, y, y);
   A[w0][n][ yyy] = B(y, y, y);

   // Terms in w^1
   A[w1][n][ 0  ] = B(w, z, z);
   A[w1][n][ x  ] = B(w, x, z);
   A[w1][n][ y  ] = B(w, y, z);
   A[w1][n][ xx ] = B(w, x, x);
   A[w1][n][ yy ] = B(w, y, y);
   A[w1][n][ xy ] = B(w, x, y);

   // Terms in w^2
   A[w2][n][ 0  ] = B(w, w, z);
   A[w2][n][ x  ] = B(w, w, x);
   A[w2][n][ y  ] = B(w, w, y);

   // Terms in w^3
   A[w3][n][ 0  ] = B(w, w, w);
   }

void EEeqns_5pt (EmatrixSet E, EquationSet A)
   {
   //
   // Computes the equations that will be used to input to polyeig.
   //    void EEeqns_5pt(E, A)
   // where E has dimensions E(3, 3, 4).  The output is a matrix
   // of dimension A(4, 10, 10), where A(i, :, :) is the coeffient of w^{i-1}
   //

   // Makes all the equations from the essential matrix E

   // First of all, set the equations to zero
   memset (&(A[0][0][0]), 0, sizeof(EquationSet));

   // Find the trace - this is a quadratic polynomial
   poly4_2 tr = traceEEt(E);

   // First equation is from the determinant
   mono_coeff (polydet4(E), A, 0);

   // Other equations from the equation 2 E*E'*E - tr(E*E') E = 0
   // In the following loop, we compute EE'E(i,j) = sum_pq E(i,p)*E(q,p)*E(q,j)
   // The way this is done is optimized for speed.  We compute first the matrix
   // EE'(i, q) and then use this to accumulate EE'E(i, j)

   int eqn = 1;  // Count on the next equation
   for (int i=0; i<3; i++)
      {
      // An array of cubic polynomials, one for each j = 0 ... 2
      poly4_3 EEE_i[3];  // Will hold (EE'E)(i,j)
      for (int j=0; j<3; j++) EEE_i[j].clear();

      // Compute each EE'(i,q) = sum_p E(i,p) E(q,p)
      for (int q=0; q<3; q++)
         {
         // Accumulate EE(i, q)
         poly4_2 EE_iq; EE_iq.clear();
         for (int p=0; p<3; p++)
            EE_iq += E(i,p) * E(q,p);

         // Now, accumulate EEE(ij) = sum_q  EE'(i,q) * E(q, j)
         for (int j=0; j<3; j++)
            EEE_i[j] += EE_iq * E(q,j);
         }

      // Now, EE'E(i,j) is computed for this i and all j
      // We can complete the computation of the coefficients from EE'E(i, j)
      for (int j=0; j<3; j++)
         mono_coeff(EEE_i[j]*2.0 - tr* E(i,j), A, eqn++);
      }
   }

void null_space_solve (double A[3][3], double &x, double &y)
   {
   // Solve for the null-space of the matrix
   
   // This time we will do pivoting
   int p1;
   double f0 = fabs(A[0][2]), f1 = fabs(A[1][2]), f2 = fabs(A[2][2]);
   if (f0 > f1) p1 = (f0>f2)? 0 : 2;
   else p1 = (f1>f2) ? 1 : 2;

   // The other two rows
   int r1 = (p1+1)%3, r2 = (p1+2)%3;

   // Now, use this to pivot
   double fac = A[r1][2] / A[p1][2];
   A[r1][0] -= fac * A[p1][0];
   A[r1][1] -= fac * A[p1][1];

   fac = A[r2][2] / A[p1][2];
   A[r2][0] -= fac * A[p1][0];
   A[r2][1] -= fac * A[p1][1];
  
   // Second pivot - largest element in column 1
   int p2 = fabs(A[r1][1]) > fabs(A[r2][1]) ? r1 : r2;
   
   // Now, read off the values - back substitution
   x = - A[p2][0]               / A[p2][1];
   y = -(A[p1][0] + A[p1][1]*x) / A[p1][2];
   }

void null_space_solve (double A[9][9], EmatrixSet &E)
   {
   // This will compute the set of solutions for the equations
   // Sweep out one column at a time, starting with highest column number

   // We do Gaussian elimination to convert M to the form M = [X | I]
   // Then the null space will be [-I | X].

   // For present, this is done without pivoting.  
   // Mostly, do not need to actually change right hand part (that becomes I)

   const int lastrow  = 4;
   const int firstcol = 4; // First column to do elimination to make I
   const int lastcol  = 8; 

   // First sweep is to get rid of the above diagonal parts
   for (int col=lastcol; col>firstcol; col--)  // No need to do first col
      {
      // Remove column col
      const int row = col-firstcol;	// Row to pivot around
      const double pivot = A[row][col];

      // Sweep out all rows up to the current one 
      for (int i=0; i<row; i++)
         {
         // This factor of the pivot row is to subtract from row i
         const double fac = A[i][col] / pivot;

         // Constant terms
         for (int j=0; j<col; j++)
            A[i][j] -= fac * A[row][j];
         }
      }

   // Now, do backward sweep to clear below the diagonal
   for (int col=firstcol; col<lastcol; col++) // No need to do lastcol
      {
      // Remove column col
      const int row = col-firstcol;	// Row to pivot around
      const double pivot = A[row][col];

      // Sweep out all rows up to the current one 
      for (int i=row+1; i<=lastrow; i++)
         {
         // This factor of the pivot row is to subtract from row i
         const double fac = A[i][col] / pivot;

         // Constant terms
         for (int j=0; j<firstcol; j++)
            A[i][j] -= fac * A[row][j];
         }
      }

   // Make this into a matrix of solutions
   double fac;
   E(0, 0) = poly4_1(1.0, 0.0, 0.0, 0.0);
   E(0, 1) = poly4_1(0.0, 1.0, 0.0, 0.0);
   E(0, 2) = poly4_1(0.0, 0.0, 1.0, 0.0);
   E(1, 0) = poly4_1(0.0, 0.0, 0.0, 1.0);
   fac = -1.0/A[0][4];
   E(1, 1) = poly4_1(fac*A[0][0], fac*A[0][1], fac*A[0][2], fac*A[0][3]);
   fac = -1.0/A[1][5];
   E(1, 2) = poly4_1(fac*A[1][0], fac*A[1][1], fac*A[1][2], fac*A[1][3]);
   fac = -1.0/A[2][6];
   E(2, 0) = poly4_1(fac*A[2][0], fac*A[2][1], fac*A[2][2], fac*A[2][3]);
   fac = -1.0/A[3][7];
   E(2, 1) = poly4_1(fac*A[3][0], fac*A[3][1], fac*A[3][2], fac*A[3][3]);
   fac = -1.0/A[4][8];
   E(2, 2) = poly4_1(fac*A[4][0], fac*A[4][1], fac*A[4][2], fac*A[4][3]);

// #define USE_TEST_VALUES
#ifdef  USE_TEST_VALUES

   // Put an artificial value in 
   E(0,0)(0) =  2; E(0,1)(0) =   4; E(0,2)(0) = -1;
   E(1,0)(0) =  4; E(1,1)(0) =   5; E(1,2)(0) = -8;
   E(2,0)(0) =  2; E(2,1)(0) = -11; E(2,2)(0) =  8;

   E(0,0)(1) =  0; E(0,1)(1) =  -1; E(0,2)(1) =  2;
   E(1,0)(1) =  1; E(1,1)(1) =   7; E(1,2)(1) =  1;
   E(2,0)(1) = -2; E(2,1)(1) =   6; E(2,2)(1) =  7;

   E(0,0)(2) =  2; E(0,1)(2) =  -3; E(0,2)(2) =  7;
   E(1,0)(2) =  1; E(1,1)(2) =  -3; E(1,2)(2) = -9;
   E(2,0)(2) =  4; E(2,1)(2) =   1; E(2,2)(2) = -9;

   E(0,0)(3) =  5; E(0,1)(3) =   2; E(0,2)(3) =  7;
   E(1,0)(3) =  1; E(1,1)(3) =  -2; E(1,2)(3) = -4;
   E(2,0)(3) =  5; E(2,1)(3) =  -1; E(2,2)(3) =  8;

#endif
   }

void Ematrix_5pt(Matches q, Matches qp, EmatrixSet &E, EquationSet &A)
   {
   // Computes the E-matrix from match inputs

   // A matrix to solve linearly for the ematrix
   double M[9][9];
   memset (&(M[0][0]), 0, sizeof (M));

   for (int i=0; i<5; i++)
      {
      M[i][0] = qp[i][0]*q[i][0];
      M[i][1] = qp[i][0]*q[i][1]; 
      M[i][2] = qp[i][0]*q[i][2];
      M[i][3] = qp[i][1]*q[i][0];
      M[i][4] = qp[i][1]*q[i][1]; 
      M[i][5] = qp[i][1]*q[i][2]; 
      M[i][6] = qp[i][2]*q[i][0];
      M[i][7] = qp[i][2]*q[i][1];
      M[i][8] = qp[i][2]*q[i][2]; 
      }

   // Solve using null_space_solve
   null_space_solve (M, E);

#  ifdef RH_DEBUG
      printf ("E = \n");
      E.print();
#  endif

   // Now, get the equations
   EEeqns_5pt(E, A);

#  ifdef RH_DEBUG
   print_equation_set (A, 3); 
#endif
   }

void sweep_up (EquationSet A, int row, int col, int degree)
   {
   // Use the given pivot point to sweep out above the pivot
   const int num1 = 6; // number of nonzero columns of A in degree 1
   const int num2 = 3; // number of nonzero columns of A in degree 2
   const int num3 = 1; // number of nonzero columns of A in degree 3

   // Find the pivot value
   const double pivot = A[degree][row][col];

   // Sweep out all rows up to the current one 
   for (int i=0; i<row; i++)
      {
      // This factor of the pivot row is to subtract from row i
      const double fac = A[degree][i][col] / pivot;

      // Constant terms
      for (int j=0; j<=col; j++)
         A[0][i][j] -= fac * A[0][row][j];

      // Degree 1 terms
      for (int j=0; j<num1; j++)
         A[1][i][j] -= fac * A[1][row][j];

      // Degree 2 terms
      for (int j=0; j<num2; j++)
         A[2][i][j] -= fac * A[2][row][j];

      // Degree 3 terms
      for (int j=0; j<num3; j++)
         A[3][i][j] -= fac * A[3][row][j];
      }
   }

void sweep_down (EquationSet A, int row, int col, int degree, int lastrow)
   {
   // Use the given pivot point to sweep out below the pivot
   const int num1 = 6; // number of nonzero columns of A in degree 1
   const int num2 = 3; // number of nonzero columns of A in degree 2
   const int num3 = 1; // number of nonzero columns of A in degree 3

   // The value of the pivot point
   const double pivot = A[degree][row][col];

   // Sweep out all rows up to the current one 
   for (int i=row+1; i<=lastrow; i++)
      {
      // This factor of the pivot row is to subtract from row i
      const double fac = A[degree][i][col] / pivot;

      // Constant terms
      for (int j=0; j<=col; j++)
         A[0][i][j] -= fac * A[0][row][j];

      // Degree 1 terms
      for (int j=0; j<num1; j++)
         A[1][i][j] -= fac * A[1][row][j];

      // Degree 2 terms
      for (int j=0; j<num2; j++)
         A[2][i][j] -= fac * A[2][row][j];

      // Degree 3 terms
      for (int j=0; j<num3; j++)
         A[3][i][j] -= fac * A[3][row][j];
      }
   }

void print_equation_set (EquationSet A, int maxdegree)
   {
   // Print out the matrix
   printf ("Equation matrix\n");
   for (int degree=0; degree<=maxdegree; degree++)
      {
      for (int i=0; i<10; i++)
         {
         for (int j=0; j<10; j++)
            printf ("%7.1f ", A[degree][i][j]);
         printf ("\n");
         }
      printf ("\n");
      } 
   }

void pivot (EquationSet A, int col, int deg, int lastrow)
   {
   // Pivot so that the largest element in the column is in the diagonal

   // Use the given pivot point to sweep out below the pivot
   const int num1 = 6; // number of nonzero columns of A in degree 1
   const int num2 = 3; // number of nonzero columns of A in degree 2
   const int num3 = 1; // number of nonzero columns of A in degree 3

   // Find the maximum value in the column
   double maxval = -1.0;
   int row = -1;
   for (int i=0; i<=lastrow; i++)
      {
      if (i != col && fabs(A[deg][i][col]) > maxval)
         {
         row = i;
         maxval = fabs(A[deg][i][col]);
         }
      }

   // We should add or subtract depending on sign
   double fac;
   if (A[deg][row][col] * A[deg][col][col] < 0.0)
      fac = -1.0;
   else fac = 1.0;

   // Next, add row to the pivot row
   // Constant terms
   for (int j=0; j<=col; j++)
      A[0][col][j] += fac * A[0][row][j];

   // Degree 1 terms
   for (int j=0; j<num1; j++)
      A[1][col][j] += fac * A[1][row][j];

   // Degree 2 terms
   for (int j=0; j<num2; j++)
      A[2][col][j] += fac * A[2][row][j];

   // Degree 3 terms
   for (int j=0; j<num3; j++)
      A[3][col][j] += fac * A[3][row][j];
   }

void reduce_Ematrix (EquationSet A)
   {
   // This reduces the equation set to 3 x 3.  In this version there is
   // no pivoting, which relies on the pivots to be non-zero.

   // Relies on the particular form of the A matrix to reduce it
   // That means that there are several rows of zero elements in different
   // degrees, as given below.

   // Sweeping out the constant terms to reduce to 6 x 6
   pivot (A, 9, 0, 8); sweep_up (A, 9, 9, 0);
   pivot (A, 8, 0, 7); sweep_up (A, 8, 8, 0);
   pivot (A, 7, 0, 6); sweep_up (A, 7, 7, 0);
   pivot (A, 6, 0, 5); sweep_up (A, 6, 6, 0);

   // Now, the matrix is 6 x 6.  Next we need to handle linear terms
   pivot (A, 5, 0, 4); sweep_up (A, 5, 5, 0);
   pivot (A, 4, 0, 3); sweep_up (A, 4, 4, 0);
   pivot (A, 3, 0, 2); sweep_up (A, 3, 3, 0);

   int lastrow = 5;
   sweep_down (A, 3, 3, 0, lastrow);
   sweep_down (A, 4, 4, 0, lastrow);

   // Also sweep out the first-order terms
   sweep_up   (A, 2, 5, 1);
   sweep_up   (A, 1, 4, 1);

   sweep_down (A, 0, 3, 1, lastrow);
   sweep_down (A, 1, 4, 1, lastrow);
   sweep_down (A, 2, 5, 1, lastrow);

   // Now, sweep out the x terms by increasing the degree
   for (int i=0; i<3; i++)
      {
      double fac = A[1][i][3+i] / A[0][3+i][3+i];

      // Introduces 4-th degree term
      A[4][i][0] = -A[3][i+3][0] * fac;

      // Transfer terms of degree 0 to 3
      for (int j=0; j<3; j++)
         {
         A[3][i][j] -= A[2][i+3][j] * fac;
         A[2][i][j] -= A[1][i+3][j] * fac;
         A[1][i][j] -= A[0][i+3][j] * fac;
         }
      }
   }

void reduce_constant_terms (EquationSet A)
   {
   // This reduces the equation set to 6 x 6 by eliminating the
   // constant terms at the end.  In this
   // no pivoting, which relies on the pivots to be non-zero.

   // Sweeping out the constant terms to reduce to 6 x 6
   pivot (A, 9, 0, 8); sweep_up (A, 9, 9, 0);
   pivot (A, 8, 0, 7); sweep_up (A, 8, 8, 0);
   pivot (A, 7, 0, 6); sweep_up (A, 7, 7, 0);
   pivot (A, 6, 0, 5); sweep_up (A, 6, 6, 0);
   }

inline void one_cofactor (EquationSet A, Polynomial poly, 
	int r0, int r1, int r2)
   {
   // Computes one term of the 3x3 cofactor expansion

   // Get a polynomial to hold a 2x2 determinant
   double two[7];
   memset (&(two[0]), 0, 7*sizeof(double));

   // Compute the 2x2 determinant - results in a 6x6 determinant
   for (int i=0; i<=3; i++)
      for (int j=0; j<=3; j++)
         two [i+j] += A[i][r1][1]*A[j][r2][2] - A[i][r2][1]*A[j][r1][2];

   // Now, multiply by degree 4 polynomial
   for (int i=0; i<=6; i++)
      for (int j=0; j<=4; j++)
         poly [i+j] += A[j][r0][0]*two[i];
   }

void compute_determinant (EquationSet A, Polynomial poly)
   {
   // Does the final determinant computation to return the determinant

   // Clear out the polynomial
   memset (&(poly[0]), 0, (PolynomialDegree+1)*sizeof(double));

   // Now, the three cofactors
   one_cofactor (A, poly, 0, 1, 2);
   one_cofactor (A, poly, 1, 2, 0);
   one_cofactor (A, poly, 2, 0, 1);
   }


// Declaration of the function to find roots
int find_real_roots_sturm( 
   double *p, int order, double *roots, int *nroots, bool non_neg = false);



void compute_E_matrix (EmatrixSet &Es, EquationSet &A, double w, Ematrix &E)
   {
   // Compute the essential matrix corresponding to this root
   double w2 = w*w;
   double w3 = w2*w;
   double w4 = w3*w;
 
   // Form equations to solve
   double M[3][3];
   for (int i=0; i<3; i++)
      {
      for (int j=0; j<3; j++)
         {
         M[i][j] = A[0][i][j] + w*A[1][i][j] + w2*A[2][i][j] + w3*A[3][i][j];
         }

      // Only the first row has degree 4 terms
      M[i][0] += w4*A[4][i][0];
      }

   // Now, find the solution
   double x, y;
   null_space_solve (M, x, y);

   // Multiply out the solution to get the essential matrix
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         {
         poly4_1 &p = Es(i, j);
         E[i][j] = w*p(0) + x*p(1) + y*p(2) + p(3);
         }
   }

void compute_E_A_poly (
     	Matches q, Matches qp, 
	double EE[4][3][3], 
	double AA[5][3][3], 
	Polynomial poly)
   {
   // This is used by the Matlab interface.
   // It takes the matches and returns the basis for the E-matrices (EE)
   // along with a 3x3 matrix of polynomials, which allows us to solve
   // for w.  It also returns the polynomial to solve

   // Get the matrix set
   EquationSet A;
   EmatrixSet E;
   Ematrix_5pt(q, qp, E, A);

   // Now, reduce its dimension to 3 x 3
   reduce_Ematrix (A);

   // Finally, get the 10-th degree polynomial out of this
   if (poly) compute_determinant (A, poly);

   // Now, copy to the simple arrays
   if (EE)
      for (int d=0; d<4; d++) for (int i=0; i<3; i++) for (int j=0; j<3; j++)
      EE[d][i][j] = E(i,j)(d);	// Do not transpose - we want Ematrices thus

   if (AA)
      for (int d=0; d<5; d++) for (int i=0; i<3; i++) for (int j=0; j<3; j++)
         AA[d][i][j] = A[d][j][i]; // Transpose
   }

static inline double pval (double *p, int deg, double x)
   {
   // Evaluates a polynomial at a given point x.  Assumes deg >= 0
   double val = p[deg];
   for (int i=deg-1; i>=0; i--)
      val = x*val + p[i];
   return val;
   }

static void compute_E_matrix_generic (
        EmatrixSet &Es,
        PolyMatrix A,
        PolyDegree deg,         // Degree of each entry in A
        int rows[Nrows],
        double w,
	double scale,
        Ematrix &E
        )
   {
   // Compute the essential matrix corresponding to this root from
   // the matrix of equations A, assumed to be in row-echelon form
   // as defined by the array rows.

   double a10 = pval(A[rows[1]][0], deg[rows[1]][0], w);
   double a11 = pval(A[rows[1]][1], deg[rows[1]][1], w);
   double a20 = pval(A[rows[2]][0], deg[rows[2]][0], w);
   double a21 = pval(A[rows[2]][1], deg[rows[2]][1], w);
   double a22 = pval(A[rows[2]][2], deg[rows[2]][2], w);

   double x = -a10/a11;
   double y = -(a20 + x*a21) / a22;

   // Multiply out the solution to get the essential matrix
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         {
         poly4_1 &p = Es(i, j);
         E[i][j] = scale*w*p(0) + x*p(1) + y*p(2) + p(3);
         }
   }

void compute_E_matrices (
     Matches q, Matches qp, 
     Ematrix Ematrices[10], 
     int &nroots,
     bool optimized
     )
   {
   // Declare and clear the matrix of equations

   // Get the matrix set
   EquationSet A;
   EmatrixSet E;
   Ematrix_5pt(q, qp, E, A);

   // print_equation_set (A, 3); 

   if (!optimized)
      {
      //------------------------------------------------------------------------
      // This is the generic version of the solver as in our paper
      //------------------------------------------------------------------------
   
      int dim = Nrows;

      // First of all, reduce to 6 x 6 by eliminating constant columns
      reduce_constant_terms (A);
      dim = 6;

      // Set up array of degrees
      PolyDegree degrees;
      for (int i=0; i<dim; i++)
         {
         degrees[i][0] = 3;
         degrees[i][1] = 2;
         degrees[i][2] = 2;
         degrees[i][3] = 1;
         degrees[i][4] = 1;
         degrees[i][5] = 1;
         degrees[i][6] = 0;
         degrees[i][7] = 0;
         degrees[i][8] = 0;
         degrees[i][9] = 0;
         }

      // Unfortunately, we need to rearrange the data since it is incompatible
      PolyMatrix P;
      for (int i=0; i<dim; i++)
         for (int j=0; j<dim; j++)
            for (int d=0; d<=degrees[i][j]; d++)
                P[i][j][d] = A[d][i][j];

      // print_polymatrix (P, 3);

      // Go ahead and find the polynomial determinant
      double scale_factor = 1.0;
      do_scale (P, degrees, scale_factor, false, dim);

      int rows[Nrows];
      find_polynomial_determinant (P, degrees, rows, dim);
      double *poly = P[rows[0]][0];	
      int poly_degree = degrees[rows[0]][0];

      // Find the positive real roots
      double roots[Maxdegree];
      find_real_roots_sturm(poly, poly_degree, roots, &nroots);

      // Now, get the ematrices
      for (int i=0; i<nroots; i++) 
         compute_E_matrix_generic (E, P, degrees, rows, 
	   roots[i], scale_factor, Ematrices[i]);
      }
   
   else
      {
      //------------------------------------------------------------------------
      // This is the highly optimized version of the code -- essentiall Nister's
      //------------------------------------------------------------------------

      // Now, reduce its dimension to 3 x 3
      reduce_Ematrix (A);

      // Finally, get the 10-th degree polynomial out of this
      Polynomial poly;
      compute_determinant (A, poly);

      // Find the roots
      double roots[PolynomialDegree];
      find_real_roots_sturm(poly, PolynomialDegree, roots, &nroots);

      // Now, get the ematrices
      for (int i=0; i<nroots; i++) 
         compute_E_matrix (E, A, roots[i], Ematrices[i]);
      }

// #define PRINT_RESULTS
#ifdef PRINT_RESULTS
#undef PRINT_RESULTS
   printf ("Polynomial\n");
   for (int i=0; i<=PolynomialDegree; i++)
      printf ("\t%14.6f\n", poly[i]/poly[0]);
#endif

// #define PRINT_RESULTS
#ifdef PRINT_RESULTS
#undef PRINT_RESULTS
   // Print out the roots
   printf ("Roots\n");
   for (int i=0; i<nroots; i++)
      printf ("\t%14.6f\n", roots[i]);
#endif

// #define PRINT_RESULTS
#ifdef PRINT_RESULTS
#undef PRINT_RESULTS
   // Print out the essential matrices
   printf ("Ematrices\n");
   for (int m=0; m<nroots; m++)
      {
      const Ematrix &E = Ematrices[m];
      for (int i=0; i<3; i++)
         printf ("\t%12.5f  %12.5f  %12.5f\n", E[i][0], E[i][1], E[i][2]);
      printf ("\n");

      // Now, compute to see if it has worked
      printf ("Verify: ");
      for (int pt=0; pt<5; pt++) 
         {
         double sum = 0.0;
         for (int i=0; i<3; i++) for (int j=0; j<3; j++)
            sum += qp[pt][i] * E[i][j] * q[pt][j];
         printf ("%11.3e ", sum);
         }
      printf ("\n\n");
      }
#endif
   }

// These are for stand-alone applications
#ifndef BUILD_MEX

#ifdef MAIN_5PT

   double urandom()
   {
	   // Returns a real random between -1 and 1
	   const int MAXRAND = 65000;
	   return 2.0*((rand()%MAXRAND)/((double) MAXRAND) - 0.5);
   }

int main (int argc, char *argv[])
   {
   // Declare the data structure for the point matches
   const int NRepetitions = 7500;

   // Flag for whether to run generic solver or optimized -- 
   // default, use optimized version
   bool run_optimized = false;

   // Get the parameters
   // Skip over the program name
   char *program_name = argv[0];
   argv++; argc--;

   // Read the parameters
   while (argc > 0)
      {
      if (argv[0][0] != '-') break;
  
      // parse the option
      switch (argv[0][1])
         {
         case 'o' :
            {
            run_optimized = true;
            break;
            }
         default :
            {
            fprintf (stderr, "%s : Unknown option \"%s\"\n",
                program_name, argv[0]);
            exit (1);
            break;
            }
         }

      // Skip to the next argument
      argv++; argc--;
      }

   // Set up a histogram
   int histogram[20];
   for (int i=0; i<20; i++) histogram[i] = 0;

   int nhistogram[11];
   for (int i=0; i<11; i++) nhistogram[i] = 0;

   double maxerr = 0.0; // Holds the maximum error, for verification

   for (int rep=0; rep<NRepetitions; rep++)
      {
      Matches_5 q, qp;

      // Fill the matches
      for (int i=0; i<5; i++) for (int j=0; j<3; j++)
         q[i][j] = urandom();

      for (int i=0; i<5; i++) for (int j=0; j<3; j++)
         qp[i][j] = urandom();

      Ematrix Ematrices[10];
      int nroots;
      compute_E_matrices (q, qp, Ematrices, nroots, run_optimized);

      // Keep histogram
      nhistogram[nroots] += 1;

      // Now, compute to see if it has worked
      for (int m=0; m<nroots; m++)
         {
         Ematrix &E = Ematrices[m];

         // Test using SVD and write out Singular values
         test_E_matrix (E);

         // Normalize the E matrix
         double sumsq = 0.0;
         for (int i=0; i<3; i++) for (int j=0; j<3; j++)
            sumsq += E[i][j]*E[i][j];
         double fac = 1.0 / sqrt(sumsq);
         for (int i=0; i<3; i++) for (int j=0; j<3; j++)
            E[i][j] *= fac;

         // Normalize the matrix
         for (int pt=0; pt<5; pt++) 
            {
            double sum = 0.0;
            for (int i=0; i<3; i++) for (int j=0; j<3; j++)
               sum += qp[pt][i] * E[i][j] * q[pt][j];

            if (fabs(sum) > maxerr) maxerr = fabs(sum);

            // Get the logarithm
            int llog = (int) (-log10(fabs(sum) + 1.0e-100));
            if (llog >= 20) llog = 19;
            if (llog < 0) llog = 0;
            histogram[llog] += 1;

            if (llog == 0) printf ("Sum = %12.3e\n", sum);
            }
         }
      }

   printf ("Maximum error = %13.5e\n", maxerr);
   for (int i=0; i<20; i++)
      printf ("%2d: %d\n", i, histogram[i]);

   printf ("Number of solutions\n");
   for (int i=0; i<11; i++)
      printf ("%2d: %d\n", i, nhistogram[i]);

   return 0;
   }

#endif // MAIN_5PT
#endif // BUILD_MEX

// polydet.cc ////////////////////


// For generating synthetic polynomials
const int ColumnDegree[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

int stuffit () { return 1; }

void eval_poly (PolyMatrix &Q, PolyDegree &deg, double x)
   {
   // Evaluates the polynomial at a given value, overwriting it
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         {
         // Evaluate the poly
         for (int k=deg[i][j]-1; k>=0; k--)
            Q[i][j][k] += x*Q[i][j][k+1];

         // Set degree to zero
         deg[i][j] = 0;
         }
   }

void cross_prod (
                 double a11[], int deg11, 
                 double a12[], int deg12,
                 double a21[], int deg21,
                 double a22[], int deg22,
                 double a00[], double toep[], int deg00,
                 double res[], int &dres,
		 BMatrix B, int &current_size
		 )
   {
   // Does a single 2x2 cross multiplication

   // Do the multiplcation in temporary storage
   double temp[2*Maxdegree + 1];

   // Work out the actual degree
   int deg1 = deg11 + deg22;
   int deg2 = deg12 + deg21;
   int deg = (deg1 > deg2) ? deg1 : deg2;

   // Clear out the temporary
   memset (temp, 0, sizeof(temp));

   // Now, start multiplying
   for (int i=0; i<=deg11; i++)
      for (int j=0; j<=deg22; j++)
         temp[i+j] += a11[i]*a22[j];

   for (int i=0; i<=deg12; i++)
      for (int j=0; j<=deg21; j++)
         temp[i+j] -= a12[i]*a21[j];

   // Clear out the result -- not really necessary
   memset (res, 0, (Maxdegree+1)*sizeof(double));

   //-----------------------------------------------------
   // This is the most tricky part of the code, to divide
   // one polynomial into the other.  By theory, the division
   // should be exact, but it is not, because of roundoff error.
   // we need to find a way to do this efficiently and accurately.
   //-----------------------------------------------------

#define USE_TOEPLITZ
#ifdef USE_TOEPLITZ

   // Now, divide by a00 - there should be no remainder
   int sres;
   polyquotient (temp, deg+1, a00, toep, deg00+1, res, sres, B, current_size);
   dres = sres-1;

#else

   // Now, divide by a00 - there should be no remainder
   double *pres = &(res[deg-deg00]);
   for (int d=deg; d>=deg00; d--)
      {
      // Work out the divisor
      int td = d - deg00;	// Degree of current term
      double val = temp[d] / a00[deg00];
      *(pres--) = val;

      // Do the subtraction involved in the division
      for (int j=0; j<deg00; j++)
         temp[j+td] -= val * a00[j];
      }
#endif

#ifdef RH_DEBUG
   // Print the remainder
   printf ("Remainder\n");
   for (int i=0; i<deg00; i++)
      printf ("\t%.5e\n", temp[i]);

#endif

   // Set the degree of the term
   dres = deg - deg00;
   }

void det_preprocess_6pt (
	PolyMatrix &Q, 
	PolyDegree degree, 
        int n_zero_roots	// Number of roots known to be zero
	)
   {
   // We do row-echelon form decomposition on the matrix to eliminate the
   // trivial known roots.
   // What is assumed here is the following.
   //   - the first row of the matrix consists of constants
   //   - the nullity of the matrix of constant terms is n_zero_roots,
   //     so when it is put in row-echelon form, the last n_zero_roots are zero.

   // Initialize the list of and columns.  We will do complete pivoting
   const int nrows = Nrows - 1;
   const int ncols = Nrows;

   int rows[Nrows], cols[Nrows];
   for (int i=0; i<nrows; i++) rows[i] = i+1;	// Miss the first row
   for (int i=0; i<ncols; i++) cols[i] = i;

   // Eliminate one row at a time
   for (int nr=nrows-1, nc=ncols-1; nr>=n_zero_roots; nr--,nc--)
      {
      // We must take the first row first to pivot around
      double bestval = 0.0;
      int bestrow = 0, bestcol = 0;

      // Find the highest value to pivot around
      for (int i=0; i<=nr; i++)
         for (int j=0; j<=nc; j++)
            {
            double val=Q[rows[i]][cols[j]][0];
            if (fabs(val) > bestval) 
               {
               bestval = fabs(val);
               bestrow = i;   // Actually rows[i]
               bestcol = j;
               }
            }

// #define RH_DEBUG
#ifdef RH_DEBUG
#undef RH_DEBUG
      // Print out the best value
      printf ("Pivot %d = %e at position %d %d\n",nr, 
	bestval, rows[bestrow], cols[bestcol]);
#endif

      // Now, select this row as a pivot.  Also keep track of rows pivoted
      int prow = rows[bestrow];
      rows[bestrow] = rows[nr];   // Replace pivot row by last row
      rows[nr] = prow;

      int pcol = cols[bestcol];
      cols[bestcol] = cols[nc];
      cols[nc] = pcol;

      // Clear out all the values above and to the right
      for (int i=0; i<nr; i++)
         {
         int iii = rows[i];
         double fac = Q[iii][pcol][0] / Q[prow][pcol][0];
         
         // Must do this to all the columns
         for (int j=0; j<ncols; j++)
            {
            int jjj = cols[j];
            int deg = degree[prow][jjj];
            int dij = degree[iii][jjj];
            if (deg>dij) degree[iii][jjj] = deg;
            for (int d=0; d<=deg; d++)
               {
               if (d <= dij)
                  Q[iii][jjj][d] -= Q[prow][jjj][d] * fac;
               else
                  Q[iii][jjj][d] = -Q[prow][jjj][d] * fac;
	       }
            }
         }
      }

   // Decrease the degree of the remaining rows
   for (int i=0; i<n_zero_roots; i++)
      {
      int ii = rows[i];
      for (int jj=0; jj<ncols; jj++)
         {
         // Decrease the degree of this element by one
         for (int d=1; d<=degree[ii][jj]; d++)
            Q[ii][jj][d-1] = Q[ii][jj][d];

         degree[ii][jj] -= 1;
         }
      }

// #define RH_DEBUG
#ifdef RH_DEBUG
#undef RH_DEBUG
   printf ("Degrees\n");
   for (int i=0; i<Nrows; i++)
      {
      for (int j=0; j<Nrows; j++)
         printf ("%1d ", degree[i][j]);
      printf ("\n");
      }
   printf("\n");

   printf ("Equation matrix\n");
   for (int i=0; i<nrows; i++)
       {
       for (int j=0; j<ncols; j++)
           printf ("%7.4f ", Q[rows[i]][cols[j]][0]);
       printf ("\n");
       }
   printf ("\n");
#endif

   }

double quick_compute_determinant (double A[Nrows][Nrows], int dim)
   {
   // Do row reduction on A to find the determinant (up to sign)

   // Initialize the list of rows
   int rows[Nrows];
   for (int i=0; i<dim; i++) rows[i] = i;

   // To accumulate the determinant
   double sign = 1.0;

   // Sweep out one row at a time
   for (int p = dim-1; p>=0; p--)
      {
      // Find the highest value to pivot around, in column p
      double bestval = 0.0;
      int bestrow = 0;
      for (int i=0; i<=p; i++)
         {
         double val=A[rows[i]][p];
         if (fabs(val) > bestval) 
            {
            bestval = fabs(val);
            bestrow = i;   // Actually rows[i]
            }
         }

      // Return early if the determinant is zero
      if (bestval == 0.0) return 0.0;

      // Now, select this row as a pivot.  Swap this row with row p
      if (bestrow != p)
         {
         int prow = rows[bestrow];
         rows[bestrow] = rows[p];   // Replace pivot row by last row
         rows[p] = prow;
         sign = -sign;		    // Keep track of sign 
         }

      // Clear out all the values above and to the right
      for (int i=0; i<p; i++)
         {
         int ii = rows[i];
         double fac = A[ii][p] / A[rows[p]][p];
         
         // Must do this to all the columns
         for (int j=0; j<dim; j++)
            A[ii][j] -= A[rows[p]][j] * fac;
         }
      }

   // Now compute the determinant
   double det = sign;
   for (int i=0; i<dim; i++)
      det *= A[rows[i]][i];
   return det;
   }

void do_scale (
	PolyMatrix &Q, 
	PolyDegree degree, 
        double &scale_factor,	// Value that x is multiplied by
        bool degree_by_row,	// Estimate degree from row degrees
        int dim			// Actual dimension of the matrix
	)
   {
   // Scale the variable so that coefficients of low and high order are equal
   // There is an assumption made here that the high order term of the
   // determinant can be computed from the high-order values of each term,
   // which is not in general true, but is so in the cases that we consider.

   // First step is to compute these values
   double low_order, high_order;
   int total_degree;

   // Find the coefficient of minimum degree term
   double A[Nrows][Nrows];
   for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
         A[i][j] = Q[i][j][0];

   low_order = quick_compute_determinant (A, dim);
   // printf ("Low order = %.7e\n", low_order);

   // Find the coefficient of maximum degree term
   total_degree = 0;
   for (int i=0; i<dim; i++)
      {
      // Find what the degree of this row is
      int rowdegree = -1;
      if (degree_by_row)
         {
         for (int j=0; j<dim; j++)
            if (degree[i][j] > rowdegree) rowdegree = degree[i][j];

         for (int j=0; j<dim; j++)
            if (degree[i][j] < rowdegree) A[i][j] = 0.0;
            else A[i][j] = Q[i][j][rowdegree];
         }
      else
         {
         for (int j=0; j<dim; j++)
            if (degree[j][i] > rowdegree) rowdegree = degree[j][i];

         for (int j=0; j<dim; j++)
            if (degree[j][i] < rowdegree) A[j][i] = 0.0;
            else A[j][i] = Q[j][i][rowdegree];
         }

      // Accumulate the row degree
      total_degree += rowdegree;
      }

   high_order = quick_compute_determinant (A, dim);
   // printf ("High order = %.7e\n", high_order);

   // Now, work out what the scale factor should be, and scale
   scale_factor = pow(fabs(low_order/high_order), 1.0 / total_degree);
   // printf ("Scale factor = %e\n", scale_factor);
   for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
         {
         double fac = scale_factor;
         for (int d=1; d<=degree[i][j]; d++)
            {
            Q[i][j][d] *= fac;
            fac *= scale_factor;
            }
         }
   }

void find_polynomial_determinant (
	PolyMatrix &Q, 
	PolyDegree deg, 
	int rows[Nrows], // This keeps the order of rows pivoted on. 
	int dim		// Actual dimension of the matrix
	)
   {
   // Compute the polynomial determinant - we work backwards from
   // the end of the matrix.  Do not bother with pivoting

   // Polynomial to start with
   double aa = 1.0;
   double *a00 = &aa;
   int deg00 = 0;

   // Initialize the list of rows
   for (int i=0; i<dim; i++)
      rows[i] = dim-1-i;

   // The row to pivot around.  At end of the loop, this will be 
   // the row containing the result.
   int piv;

   for (int p = dim-1; p>=1; p--)
      {
      // We want to find the element with the biggest high order term to
      // pivot around

#define DO_PARTIAL_PIVOT
#ifdef  DO_PARTIAL_PIVOT
      double bestval = 0.0;
      int bestrow = 0;
      for (int i=0; i<=p; i++)
         {
         double val=Q[rows[i]][p][deg[rows[i]][p]];
         if (fabs(val) > bestval) 
            {
            bestval = fabs(val);
            bestrow = i;   // Actually rows[i]
            }
         }

      // Now, select this row as a pivot.  Also keep track of rows pivoted
      piv = rows[bestrow];
      rows[bestrow] = rows[p];   // Replace pivot row by last row
      rows[p] = piv;
#else

      piv = rows[p];

#endif

// #define RH_DEBUG
#ifdef RH_DEBUG
#undef RH_DEBUG
      // Print out the pivot
      printf ("Pivot %d = \n", p);
      for (int i=0; i<=deg[piv][p]; i++)
         printf ("\t%16.5e\n", Q[piv][p][i]);
#endif

      // Set up a matrix for Toeplitz
      BMatrix B;
      int current_size = 0;

      // Also the Toeplitz vector
      double toep[Maxdegree+1];
      for (int i=0; i<=deg00; i++)
        {
        toep[i] = 0.0;
        for (int j=0; j+i<=deg00; j++)
           toep[i] += a00[j] * a00[j+i];
        }

      // Clear out all the values above and to the right
      for (int i=0; i<p; i++)
         {
         int iii = rows[i];
         for (int j=0; j<p; j++)
            cross_prod (
               Q[piv][p], deg[piv][p],
               Q[piv][j], deg[piv][j],
               Q[iii][p], deg[iii][p],
               Q[iii][j], deg[iii][j],
               a00, toep, deg00,
               Q[iii][j], deg[iii][j],	// Replace original value
               B, current_size
               );
         }

      // Now, update to the next
      a00 = &(Q[piv][p][0]);
      deg00 = deg[piv][p];
      }

   // Now, the polynomial in the position Q(0,0) is the solution
   }

//=========================================================================
//  The rest of this code is for stand-alone testing
//=========================================================================

#ifndef BUILD_MEX
#ifdef POLYDET_HAS_MAIN

	static double urandom()
	{
		// Returns a real random between -1 and 1
		const int MAXRAND = 65000;
		return 4.0*((rand()%MAXRAND)/((double) MAXRAND) - 0.5);
		// return rand() % 20 - 10.0;
	}

void copy_poly (
      PolyMatrix pin, PolyDegree din, PolyMatrix pout, PolyDegree dout)
   {
   memcpy (pout, pin, sizeof(PolyMatrix));
   memcpy (dout, din, sizeof(PolyDegree));
   }

int accuracy_test_main (int argc, char *argv[])
   {
   // Try this out

   // To hold the matrix and its degrees
   PolyMatrix p;
   PolyDegree degrees;
   int pivotrows[Nrows];

   //--------------------------------------------------------

   // Generate some data
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         degrees[i][j] = ColumnDegree[j];

   // Now, fill out the polynomials
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         {
         for (int k=0; k<=ColumnDegree[j]; k++)
            p[i][j][k] = urandom();
         for (int k=ColumnDegree[j]+1; k<=Maxdegree; k++)
            p[i][j][k] = 0.0;
         }

   //--------------------------------------------------------

   // Back up the matrix
   PolyMatrix pbak;
   PolyDegree degbak;
   copy_poly (p, degrees, pbak, degbak);

   //---------------------
   // Find determinant, then evaluate
   //---------------------

   // Now, compute the determinant
   copy_poly (pbak, degbak, p, degrees);

   // Preprocess
   double scale_factor = 1.0;
   det_preprocess_6pt (p, degrees, 3);
   do_scale (p, degrees, scale_factor, true);

   // Find the determinant
   find_polynomial_determinant (p, degrees, pivotrows);

   // Print out the solution
   const double print_solution = 0;
   if (print_solution)
      {
      printf ("Solution is\n");
      for (int i=0; i<=degrees[pivotrows[0]][0]; i++)
         printf ("\t%16.5e\n", p[pivotrows[0]][0][i]);
      }

   // Now, evaluate and print out
   double x = 1.0;
   eval_poly (p, degrees, x);

   double val1 = p[pivotrows[0]][0][0];

   //---------------------
   // Now, evaluate first
   //---------------------

   copy_poly (pbak, degbak, p, degrees);

   // Now, evaluate and print out
   eval_poly (p, degrees, x);
   find_polynomial_determinant (p, degrees, pivotrows);

   double val2 = p[pivotrows[0]][0][0];
   double diff = fabs((fabs(val1) - fabs(val2))) / fabs(val1);

   printf ("%18.9e\t%18.9e\t%10.3e\n", val1, val2, diff);

   return 0;
   }

int timing_test_main (int argc, char *argv[])
   {

	   const int NRepetitions = 10000;

   // Try this out

   // To hold the matrix and its degrees
   PolyMatrix p;
   PolyDegree degrees;
   int pivotrows[Nrows];

   //--------------------------------------------------------

   // Generate some data
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         degrees[i][j] = ColumnDegree[j];

   // Now, fill out the polynomials
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         {
         for (int k=0; k<=ColumnDegree[j]; k++)
            p[i][j][k] = urandom();
         for (int k=ColumnDegree[j]+1; k<=Maxdegree; k++)
            p[i][j][k] = 0.0;
         }

   //--------------------------------------------------------

   // Back up the matrix
   PolyMatrix pbak;
   PolyDegree degbak;
   copy_poly (p, degrees, pbak, degbak);

   // Now, compute the determinant
   for (int rep=0; rep<NRepetitions; rep++)
      {
      copy_poly (pbak, degbak, p, degrees);
      find_polynomial_determinant (p, degrees, pivotrows);
      }

   return 0;
   }

//===========================================================================

int main (int argc, char *argv[])
   {
   // Now, compute the determinant
   // for (int rep=0; rep<NRepetitions; rep++)
   //  accuracy_test_main (argc, argv);

   timing_test_main (argc, argv);

   return 0;
   }

#endif
#endif  // BUILD_MEX

// polyquotient.cc //////////////////////////////////////////

static void toeplitz_get_b_vector (
	BMatrix B, 
	double *t, int st,	
	int &current_size, int required_size
	)
   {
   // Incrementally computes the matrix of back-vectors used in the Levinson
   // algorithm. The back-vectors bn are stored as rows in B.

   // Initialize B
   if (current_size <= 0)
      {
      B[0][0] = 1.0/t[0];
      current_size = 1;
      }

   // Build up the back vectors one by one
   for (int n=current_size; n<required_size; n++)
      {
      // Fill out row n of the matrix B

      // Compute out each vector at once
      double e = 0.0;
      for (int i=1; i<st && i<=n; i++)
         e += t[i] * B[n-1][i-1];

      // Write the next row of the matrix
      double cb = 1.0 / (1.0 - e*e);
      double cf = -(e*cb);

      // Addresses into the arrays for the addition
      double *b0 = &B[n-1][0];
      double *f0 = &B[n-1][n];
      double *bn = &(B[n][0]);

      // First term does not include b, last does not have f
      *(bn++) = *(--f0) * cf;
      while (f0 != &B[n-1][0])
         *(bn++) = *(b0++) * cb + *(--f0) * cf;
      *bn = *b0 * cb;
      }

   // Update the current dimension
   current_size = required_size;
   }

void polyquotient (
	double *a, int sa, 
	double *b, double *t, int sb, 
	double *q, int &sq,
	BMatrix B, int &current_size
	)
   {
   // Computes the quotient of one polynomial with respect to another
   // in a least-squares sense, using Toeplitz matrices
   
   // First, get the sizes of the vectors
   sq = sa - sb + 1;  // Degree of the quotient

   // Next get the back-vectors for the Levinson algorithm
   if (sq > current_size)
      toeplitz_get_b_vector (B, t, sb, current_size, sq);

#ifdef RH_DEBUG
   for (int i=0; i<sq; i++)
      {
      for (int j=0; j<sq; j++)
          printf ("%9.3f ", B[i][j]);
      printf ("\n");
      }
#endif

   // Initially no values
   memset(q, 0, sq*sizeof(double));

   // Next, compute the quotient, one at a time
   for (int n=0; n<sq; n++)
      {
      // Inner product of a and b
      double yn = 0.0;
      for (int i=0; i<sb; i++)
         yn += b[i] * a[i+n];

      // The error value
      double e = 0.0;
      for (int i=1; i<sb && i<=n; i++)
         e += t[i] * q[n-i];

#ifdef RH_DEBUG
      printf ("yn = %12.6f, e = %12.6f\n", yn, e);
#endif

      // Now, update the value of q
      double fac = yn - e;
      q[n] = 0.0;
      for (int i=0; i<=n; i++)
         q[i] += fac * B[n][i];
      }
   }

#ifndef BUILD_MEX
#ifdef POLYQUOTIENT_HAS_MAIN

const int NRepetitions = 100000;


int main (int argc, char *argv[])
   {
   // Try the thing out
   const int qsize = 21;
   const int bsize = 21;
   const int asize = qsize+bsize-1;

   int da = asize;
   int db = bsize;
   int dq = qsize;

   double a[asize];
   double b[bsize];
   double q[qsize];
   double t[bsize];

   // Fill out the polynomials with random values
   for (int i=0; i<dq; i++) q[i] = (double) i+1;
   for (int i=0; i<db; i++) b[i] = (double) i+1;

   // The matrix of back vectors
   BMatrix B;

   // Now, try the thing
   for (int row=0; row<9; row++)
      {
      // Do the test for pivoting on row "row"
      printf ("Pivoting on row %d\n", row);
      fflush(stdout);

      db = 2*row+1;
      da = 4*row+5;
      dq = 2*row+5;
      int reps = (9-row)*(9-row);

      // Multiply out to get a
      for (int i=0; i<da; i++) a[i] = 0.0;
      for (int i=0; i<db; i++)
         for (int j=0; j<dq; j++)
            a[i+j] += b[i]*q[j];

      // Also, multiply out to get the toeplitz vector
      for (int i=0; i<db; i++)
         {
         t[i] = 0.0;
         for (int j=0; j+i<db; j++)
            t[i] += b[j] * b[j+i];
         }

#ifdef RH_DEBUG
      printf ("a = \n");
      for (int i=0; i<da; i++) printf ("%7.2f\n", a[i]);

      printf ("b = \n");
      for (int i=0; i<db; i++) printf ("%7.2f\n", b[i]);

      printf ("q = \n");
      for (int i=0; i<dq; i++) printf ("%7.2f\n", q[i]);

      printf ("t = \n");
      for (int i=0; i<db; i++) printf ("%7.2f\n", t[i]);
#endif

      for (int rep=0; rep<NRepetitions; rep++)
         {
         int current_size = 0;
         for (int m=0; m<reps; m++)
            {
            polyquotient (a, da, b, t, db, q, dq, B, current_size);
            }
         }
      }

   printf ("Finished\n");
   fflush(stdout);

#ifdef RH_DEBUG
   // Now, print out the result
   for (int i=0; i<dq; i++)
      printf("%9.3f\n", q[i]);
#endif

   return 0;
   } 
   

#endif
#endif	// BUILD_MEX

// sturm.cc ////////////////////////////////////////

/*
 * sturm.c
 *
 * the functions to build and evaluate the Sturm sequence
 */

// #define RH_DEBUG

#define RELERROR      1.0e-12   /* smallest relative error we want */
#define MAXPOW        32        /* max power of 10 we wish to search to */
#define MAXIT         800       /* max number of iterations */
#define SMALL_ENOUGH  1.0e-12   /* a coefficient smaller than SMALL_ENOUGH 
                                 * is considered to be zero (0.0). */

/* structure type for representing a polynomial */
typedef struct p {
   int ord;
   double coef[Maxdegree+1];
   } poly;

/*---------------------------------------------------------------------------
 * evalpoly
 *
 * evaluate polynomial defined in coef returning its value.
 *--------------------------------------------------------------------------*/

double evalpoly (int ord, double *coef, double x)
   {
   double *fp = &coef[ord];
   double f = *fp;

   for (fp--; fp >= coef; fp--)
      f = x * f + *fp;

   return(f);
   }

int modrf_pos( int ord, double *coef, double a, double b, 
	double *val, int invert)
   {
   int  its;
   double fx, lfx;
   double *fp;
   double *scoef = coef;
   double *ecoef = &coef[ord];
   double fa, fb;

   // Invert the interval if required
   if (invert)
      {
      double temp = a;
      a = 1.0 / b;
      b = 1.0 / temp;
      }

   // Evaluate the polynomial at the end points
   if (invert)
      {
      fb = fa = *scoef;
      for (fp = scoef + 1; fp <= ecoef; fp++) 
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }
   else
      {
      fb = fa = *ecoef;
      for (fp = ecoef - 1; fp >= scoef; fp--) 
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }

   // if there is no sign difference the method won't work
   if (fa * fb > 0.0)
      return(0);

   // Return if the values are close to zero already
   if (fabs(fa) < RELERROR) 
      {
      *val = invert ? 1.0/a : a;
      return(1);
      }

   if (fabs(fb) < RELERROR) 
      {
      *val = invert ? 1.0/b : b;
      return(1);
      }

   lfx = fa;

   for (its = 0; its < MAXIT; its++) 
      {
      // Assuming straight line from a to b, find zero
      double x = (fb * a - fa * b) / (fb - fa);

      // Evaluate the polynomial at x
      if (invert)
         {
         fx = *scoef;
         for (fp = scoef + 1; fp <= ecoef; fp++)
            fx = x * fx + *fp;
	 }
      else
         {
         fx = *ecoef;
         for (fp = ecoef - 1; fp >= scoef; fp--)
            fx = x * fx + *fp;
         }

      // Evaluate two stopping conditions
      if (fabs(x) > RELERROR && fabs(fx/x) < RELERROR) 
         {
         *val = invert ? 1.0/x : x;
         return(1);
         }
      else if (fabs(fx) < RELERROR) 
         {
         *val = invert ? 1.0/x : x;
         return(1);
         }

      // Subdivide region, depending on whether fx has same sign as fa or fb
      if ((fa * fx) < 0) 
         {
         b = x;
         fb = fx;
         if ((lfx * fx) > 0)
            fa /= 2;
         } 
      else 
         {
         a = x;
         fa = fx;
         if ((lfx * fx) > 0)
            fb /= 2;
         }

   
      // Return if the difference between a and b is very small
      if (fabs(b-a) < fabs(RELERROR * a))
         {
         *val = invert ? 1.0/a : a;
         return(1);
         }

      lfx = fx;
      }

   //==================================================================
   // This is debugging in case something goes wrong.
   // If we reach here, we have not converged -- give some diagnostics
   //==================================================================

   fprintf(stderr, "modrf overflow on interval %f %f\n", a, b);
   fprintf(stderr, "\t b-a = %12.5e\n", b-a);
   fprintf(stderr, "\t fa  = %12.5e\n", fa);
   fprintf(stderr, "\t fb  = %12.5e\n", fb);
   fprintf(stderr, "\t fx  = %12.5e\n", fx);

   // Evaluate the true values at a and b
   if (invert)
      {
      fb = fa = *scoef;
      for (fp = scoef + 1; fp <= ecoef; fp++)
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }
   else
      {
      fb = fa = *ecoef;
      for (fp = ecoef - 1; fp >= scoef; fp--) 
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }

   fprintf(stderr, "\t true fa = %12.5e\n", fa);
   fprintf(stderr, "\t true fb = %12.5e\n", fb);
   fprintf(stderr, "\t gradient= %12.5e\n", (fb-fa)/(b-a));

   // Print out the polynomial
   fprintf(stderr, "Polynomial coefficients\n");
   for (fp = ecoef; fp >= scoef; fp--) 
      fprintf (stderr, "\t%12.5e\n", *fp);

   return(0);
   }

/*---------------------------------------------------------------------------
 * modrf
 *
 * uses the modified regula-falsi method to evaluate the root
 * in interval [a,b] of the polynomial described in coef. The
 * root is returned is returned in *val. The routine returns zero
 * if it can't converge.
 *--------------------------------------------------------------------------*/

int modrf (int ord, double *coef, double a, double b, double *val)
   {
   // This is an interfact to modrf that takes account of different cases
   // The idea is that the basic routine works badly for polynomials on
   // intervals that extend well beyond [-1, 1], because numbers get too large

   double *fp;
   double *scoef = coef;
   double *ecoef = &coef[ord];
   const int invert = 1;

   double fp1= 0.0, fm1 = 0.0; // Values of function at 1 and -1
   double fa = 0.0, fb  = 0.0; // Values at end points
 
   // We assume that a < b
   if (a > b)
      {
      double temp = a;
      a = b;
      b = temp;
      }

   // The normal case, interval is inside [-1, 1]
   if (b <= 1.0 && a >= -1.0) return modrf_pos (ord, coef, a, b, val, !invert);

   // The case where the interval is outside [-1, 1]
   if (a >= 1.0 || b <= -1.0)
      return modrf_pos (ord, coef, a, b, val, invert);

   // If we have got here, then the interval includes the points 1 or -1.
   // In this case, we need to evaluate at these points

   // Evaluate the polynomial at the end points
   for (fp = ecoef - 1; fp >= scoef; fp--) 
      {
      fp1 = *fp + fp1;
      fm1 = *fp - fm1;
      fa = a * fa + *fp;
      fb = b * fb + *fp;
      }

   // Then there is the case where the interval contains -1 or 1
   if (a < -1.0 && b > 1.0)
      {
      // Interval crosses over 1.0, so cut
      if (fa * fm1 < 0.0)      // The solution is between a and -1
         return modrf_pos (ord, coef, a, -1.0, val, invert);
      else if (fb * fp1 < 0.0) // The solution is between 1 and b
         return modrf_pos (ord, coef, 1.0, b, val, invert);
      else                     // The solution is between -1 and 1
         return modrf_pos(ord, coef, -1.0, 1.0, val, !invert);
      }
   else if (a < -1.0)
      {
      // Interval crosses over 1.0, so cut
      if (fa * fm1 < 0.0)      // The solution is between a and -1
         return modrf_pos (ord, coef, a, -1.0, val, invert);
      else                     // The solution is between -1 and b
         return modrf_pos(ord, coef, -1.0, b, val, !invert); 
      }
   else  // b > 1.0
      {
      if (fb * fp1 < 0.0) // The solution is between 1 and b
         return modrf_pos (ord, coef, 1.0, b, val, invert);
      else                     // The solution is between a and 1
         return modrf_pos(ord, coef, a, 1.0, val, !invert);
      }
   }

/*---------------------------------------------------------------------------
 * modp
 *
 *  calculates the modulus of u(x) / v(x) leaving it in r, it
 *  returns 0 if r(x) is a constant.
 *  note: this function assumes the leading coefficient of v is 1 or -1
 *--------------------------------------------------------------------------*/

static int modp(poly *u, poly *v, poly *r)
   {
   int j, k;  /* Loop indices */

   double *nr = r->coef;
   double *end = &u->coef[u->ord];

   double *uc = u->coef;
   while (uc <= end)
      *nr++ = *uc++;

   if (v->coef[v->ord] < 0.0) {

      for (k = u->ord - v->ord - 1; k >= 0; k -= 2)
         r->coef[k] = -r->coef[k];

      for (k = u->ord - v->ord; k >= 0; k--)
         for (j = v->ord + k - 1; j >= k; j--)
            r->coef[j] = -r->coef[j] - r->coef[v->ord + k]
         * v->coef[j - k];
      } else {
         for (k = u->ord - v->ord; k >= 0; k--)
            for (j = v->ord + k - 1; j >= k; j--)
               r->coef[j] -= r->coef[v->ord + k] * v->coef[j - k];
      }

   k = v->ord - 1;
   while (k >= 0 && fabs(r->coef[k]) < SMALL_ENOUGH) {
      r->coef[k] = 0.0;
      k--;
      }

   r->ord = (k < 0) ? 0 : k;

   return(r->ord);
   }

/*---------------------------------------------------------------------------
 * buildsturm
 *
 * build up a sturm sequence for a polynomial in smat, returning
 * the number of polynomials in the sequence
 *--------------------------------------------------------------------------*/

int buildsturm(int ord, poly *sseq)
   {
   sseq[0].ord = ord;
   sseq[1].ord = ord - 1;

   /* calculate the derivative and normalise the leading coefficient */
      {
      int i;    // Loop index
      poly *sp;
      double f = fabs(sseq[0].coef[ord] * ord);
      double *fp = sseq[1].coef;
      double *fc = sseq[0].coef + 1;

      for (i=1; i<=ord; i++)
         *fp++ = *fc++ * i / f;

      /* construct the rest of the Sturm sequence */
      for (sp = sseq + 2; modp(sp - 2, sp - 1, sp); sp++) {

         /* reverse the sign and normalise */
         f = -fabs(sp->coef[sp->ord]);
         for (fp = &sp->coef[sp->ord]; fp >= sp->coef; fp--)
            *fp /= f;
         }

      sp->coef[0] = -sp->coef[0]; /* reverse the sign */

      return(sp - sseq);
      }
   }

/*---------------------------------------------------------------------------
 * numchanges
 *
 * return the number of sign changes in the Sturm sequence in
 * sseq at the value a.
 *--------------------------------------------------------------------------*/

int numchanges(int np, poly *sseq, double a)
   {
   int changes = 0;

   double lf = evalpoly(sseq[0].ord, sseq[0].coef, a);

   poly *s;
   for (s = sseq + 1; s <= sseq + np; s++) {
      double f = evalpoly(s->ord, s->coef, a);
      if (lf == 0.0 || lf * f < 0)
         changes++;
      lf = f;
      }

   return(changes);
   }

/*---------------------------------------------------------------------------
 * numroots
 *
 * return the number of distinct real roots of the polynomial described in sseq.
 *--------------------------------------------------------------------------*/

int numroots(int np, poly *sseq, int *atneg, int *atpos, bool non_neg)
   {
   int atposinf = 0;
   int atneginf = 0;

   /* changes at positive infinity */
   double f;
   double lf = sseq[0].coef[sseq[0].ord];

   poly *s;
   for (s = sseq + 1; s <= sseq + np; s++) {
      f = s->coef[s->ord];
      if (lf == 0.0 || lf * f < 0)
         atposinf++;
      lf = f;
      }

   // changes at negative infinity or zero
   if (non_neg)
      atneginf = numchanges(np, sseq, 0.0);

   else
      {
      if (sseq[0].ord & 1)
         lf = -sseq[0].coef[sseq[0].ord];
      else
         lf = sseq[0].coef[sseq[0].ord];

      for (s = sseq + 1; s <= sseq + np; s++) {
         if (s->ord & 1)
            f = -s->coef[s->ord];
         else
            f = s->coef[s->ord];
         if (lf == 0.0 || lf * f < 0)
            atneginf++;
         lf = f;
         }
      }

   *atneg = atneginf;
   *atpos = atposinf;

   return(atneginf - atposinf);
   }


/*---------------------------------------------------------------------------
 * sbisect
 *
 * uses a bisection based on the sturm sequence for the polynomial
 * described in sseq to isolate intervals in which roots occur,
 * the roots are returned in the roots array in order of magnitude.
 *--------------------------------------------------------------------------*/

int sbisect(int np, poly *sseq, 
            double min, double max, 
            int atmin, int atmax, 
            double *roots)
   {
   double mid;
   int atmid;
   int its;
   int  n1 = 0, n2 = 0;
   int nroot = atmin - atmax;

   if (nroot == 1) {

      /* first try a less expensive technique.  */
      if (modrf(sseq->ord, sseq->coef, min, max, &roots[0]))
         return 1;

      /*
       * if we get here we have to evaluate the root the hard
       * way by using the Sturm sequence.
       */
      for (its = 0; its < MAXIT; its++) {
         mid = (min + max) / 2.0;
         atmid = numchanges(np, sseq, mid);

         if (fabs(mid) > RELERROR) {
            if (fabs((max - min) / mid) < RELERROR) {
               roots[0] = mid;
               return 1;
               }
            } else if (fabs(max - min) < RELERROR) {
               roots[0] = mid;
               return 1;
            }

         if ((atmin - atmid) == 0)
            min = mid;
         else
            max = mid;
         }

      if (its == MAXIT) {
         fprintf(stderr, "sbisect: overflow min %f max %f\
                         diff %e nroot %d n1 %d n2 %d\n",
                         min, max, max - min, nroot, n1, n2);
         roots[0] = mid;
         }

      return 1;
      }

   /* more than one root in the interval, we have to bisect */
   for (its = 0; its < MAXIT; its++) {

      mid = (double) ((min + max) / 2);
      atmid = numchanges(np, sseq, mid);

      n1 = atmin - atmid;
      n2 = atmid - atmax;

      if (n1 != 0 && n2 != 0) {
         sbisect(np, sseq, min, mid, atmin, atmid, roots);
         sbisect(np, sseq, mid, max, atmid, atmax, &roots[n1]);
         break;
         }

      if (n1 == 0)
         min = mid;
      else
         max = mid;
      }

   if (its == MAXIT) {
      fprintf(stderr, "sbisect: roots too close together\n");
      fprintf(stderr, "sbisect: overflow min %f max %f diff %e\
                      nroot %d n1 %d n2 %d\n",
                      min, max, max - min, nroot, n1, n2);
      for (n1 = atmax; n1 < atmin; n1++)
         roots[n1 - atmax] = mid;
      }

   return 1; 
   }

int find_real_roots_sturm( 
	double *p, int order, double *roots, int *nroots, bool non_neg)
   {
   /*
    * finds the roots of the input polynomial.  They are returned in roots.
    * It is assumed that roots is already allocated with space for the roots.
    */

   poly sseq[Maxdegree+1];
   double  min, max;
   int  i, nchanges, np, atmin, atmax;

   // Copy the coefficients from the input p.  Normalize as we go
   double norm = 1.0 / p[order];
   for (i=0; i<=order; i++)
      sseq[0].coef[i] =  p[i] * norm;

   // Now, also normalize the other terms
   double val0 = fabs(sseq[0].coef[0]);
   double fac = 1.0; // This will be a factor for the roots
   if (val0 > 10.0)  // Do this in case there are zero roots
      {
      fac = pow(val0, -1.0/order);
      double mult = fac;
      for (int i=order-1; i>=0; i--)
         {
         sseq[0].coef[i] *= mult;
         mult = mult * fac; 
         }
      }

   /* build the Sturm sequence */
   np = buildsturm(order, sseq);

#ifdef RH_DEBUG
   {
   int i, j;

   printf("Sturm sequence for:\n");
   for (i=order; i>=0; i--)
      printf("%lf ", sseq[0].coef[i]);
   printf("\n\n");

   for (i = 0; i <= np; i++) {
      for (j = sseq[i].ord; j >= 0; j--)
         printf("%10f ", sseq[i].coef[j]);
      printf("\n");
      }

   printf("\n");
   }
#endif

   // get the number of real roots
   *nroots = numroots(np, sseq, &atmin, &atmax, non_neg);

   if (*nroots == 0) {
      // fprintf(stderr, "solve: no real roots\n");
      return 0 ;
      }

   /* calculate the bracket that the roots live in */
   if (non_neg) min = 0.0;
   else
      {
      min = -1.0;
      nchanges = numchanges(np, sseq, min);
      for (i = 0; nchanges != atmin && i != MAXPOW; i++) { 
         min *= 10.0;
         nchanges = numchanges(np, sseq, min);
         }

      if (nchanges != atmin) {
         printf("solve: unable to bracket all negative roots\n");
         atmin = nchanges;
         }
      }

   max = 1.0;
   nchanges = numchanges(np, sseq, max);
   for (i = 0; nchanges != atmax && i != MAXPOW; i++) { 
      max *= 10.0;
      nchanges = numchanges(np, sseq, max);
      }

   if (nchanges != atmax) {
      printf("solve: unable to bracket all positive roots\n");
      atmax = nchanges;
      }

   *nroots = atmin - atmax;

   /* perform the bisection */
   sbisect(np, sseq, min, max, atmin, atmax, roots);

   /* Finally, reorder the roots */
   for (i=0; i<*nroots; i++)
      roots[i] /= fac;

#ifdef RH_DEBUG

   /* write out the roots */
   printf("Number of roots = %d\n", *nroots);
   for (i=0; i<*nroots; i++)
      printf("%12.5e\n", roots[i]);

#endif

   return 1; 
   }

} // namespace hartley
