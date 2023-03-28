/*
 * Open Source Physics software is free software as described near the bottom of this code file.
 *
 * For additional information and documentation on Open Source Physics please see:
 * <http://www.opensourcephysics.org/>
 */

/**
    Adapted to OSP by Javier E. Hasbun, 2009.
  <P>
    @Copyright (c) 2009
    This software is to support the Open Source Physics library
    http://www.opensourcephysics.org under the terms of the GNU General Public
    License (GPL) as published by the Free Software Foundation.
 **/

/**
   ComplexMatrix.java
   solve, invert, etc. last argument is output
   void solve(Complex[][] A, Complex[] Y, Complex[] X);         X=A^-1*Y
   void invert(Complex[][] A);                                  A=A^-1
   Complex determinant(Complex[][] A);                          d=det A
   void eigenvalues(Complex[][] A, Complex[][] V, Complex[] Y); V,Y=eigen A
   void eigenCheck(Complex[][] A, Complex[][] V, Complex[] Y);  printout
   void multiply(Complex[][] A, Complex[][] B, Complex[][] C);  C=A*B
   void add(Complex[][] A, Complex[][] B, Complex[][] C);       C=A+B
   void subtract(Complex[][] A, Complex[][] B, Complex[][] C);  C=A-B
   double norm1(Complex[][] A);                                 d=norm1 A
   double norm2(Complex[][] A);  sqrt largest eigenvalue A^T*A  d=norm2 A
   double normFro(Complex[][] A); Frobenius                     d=normFro A
   double normInf(Complex[][] A);                               d=normInf A
   void copy(Complex[][] A, Complex[][] B);                     B=A
   void boolean equals(Complex[][] A, Complex[][] B);           A==B
   void fromDouble(double[] A, double[][] b, Complex[][] C);    C=(A,b)
   void fromDouble(double[] A, Complex[][] B);                  B=A
   void identity(Complex[][] A);                                A=I
   void zero(Complex[][] A);                                    A=0
   void print(Complex[][] A);                                   A
   void multiply(Complex[][] A, Complex[] X, Complex[] Y);      Y=A*X
   void add(Complex[] X, Complex[] Y, Complex[] Z);             Z=X+Y
   void subtract(Complex[] X, Complex[] Y, Complex[] Z);        Z=X-Y
   double norm1(Complex[] X);                                   d=norm1 X
   double norm2(Complex[] X);                                   d=norm2 X
   double normInf(Complex[] X);                                 d=normInf X
   void copy(Complex[] X, Complex[] Y);                         Y=X
   void boolean equals(Complex[] X, Complex[] Y);               X==Y
   void fromDouble(double[] X, double[] Y, Complex[] Z);        Z=(X,Y)
   void fromDouble(double[] X, Complex[] Y);                    Y=(X,0)
   void unitVector(X[],I);                                      X[I]=1, else 0
   void zero(Complex[] X);                                      X=0
   void print(Complex[] X);                                     X
   void readSize(String, int[] rowCol)
   void read(String, Complex[][])
   void closeInput()
   void write(String, Complex[][])
   void closeOutput()

   void tensormultiply(
                 Complex A[][], Complex B[][], Complex C[][])  C=A\otimes B
   void conjugatetranspose(Complex A[][], Complex B[][])       B=A^{\dagger}
   Complex[][] myAdd(Complex[][] m1, Complex[][] m2)
   Complex[][] mySubtract(Complex[][] m1, Complex[][] m2)
   Complex[] mySubtract(Complex[] m1, Complex[] m2)
   Complex[][] myMul(Complex[][] m1, Complex[][] m2)
   Complex[][] myConjTrans(Complex[][] m)
   Complex[][] newSquareMatrix(int n)
   Complex innerproduct(Complex[] u, Complex[] v)
   Complex[] div(Complex[] v, double d)
   Complex[][] myTranspose(Complex[][] m)
   Complex[] proj(Complex[] u, Complex[] v)
   Complex[][] GramSchmidt(Complex[][] v)
   Complex[][] supp(Complex[][] m)
   String toString(Complex[][][][] M)
   String toString(Complex[][] M)
   String toString(Complex[] vector)
 **/

// // package org.opensourcephysics.numerics;
package matrix;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.List;

/**
 * Class description
 *
*/
public strictfp class ComplexMatrix {
  public static void solve(final Complex[][] A, final Complex[] Y, Complex[] X) {
    // solve complex linear equations for X where Y = A * X
    // method: Gauss-Jordan elimination using maximum pivot
    // usage: ComplexMatrix.solve(A,Y,X);
    // Translated to java by : Jon Squire , 4 April 2003
    // First written by Jon Squire December 1959 for IBM 650, translated to
    // other languages e.g. Fortran converted to Ada converted to C
    // converted to java
    int n = A.length;
    int m = n+1;
    Complex[][] B = new Complex[n][m]; // working matrix
    int[] row = new int[n];            // row interchange indicies
    int hold, I_pivot;                 // pivot indicies
    Complex pivot;                     // pivot element value
    double abs_pivot;
    if((A[0].length!=n)||(Y.length!=n)||(X.length!=n)) {
      System.out.println("Error in ComplexMatrix.solve inconsistent array sizes."); //$NON-NLS-1$
    }
    // build working data structure
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        B[i][j] = A[i][j];
      }
      B[i][n] = Y[i];
    }
    // set up row interchange vectors
    for(int k = 0; k<n; k++) {
      row[k] = k;
    }
    // begin main reduction loop
    for(int k = 0; k<n; k++) {
      // find largest element for pivot
      pivot = B[row[k]][k];
      abs_pivot = pivot.abs();
      I_pivot = k;
      for(int i = k+1; i<n; i++) {
        if(B[row[i]][k].abs()>abs_pivot) {
          I_pivot = i;
          pivot = B[row[i]][k];
          abs_pivot = pivot.abs();
        }
      }
      // have pivot, interchange row indicies
      hold = row[k];
      row[k] = row[I_pivot];
      row[I_pivot] = hold;
      // check for near singular
      if(abs_pivot<1.0E-10) {
        for(int j = k+1; j<n+1; j++) {
          B[row[k]][j] = new Complex(0.0, 0.0);
        }
        System.out.println("redundant row (singular) "+row[k]); //$NON-NLS-1$
      }                                                         // singular, delete row
        else {
        // reduce about pivot
        for(int j = k+1; j<n+1; j++) {
          B[row[k]][j] = B[row[k]][j].div(B[row[k]][k]);
        }
        // inner reduction loop
        for(int i = 0; i<n; i++) {
          if(i!=k) {
            for(int j = k+1; j<n+1; j++) {
              B[row[i]][j] = B[row[i]][j].subtract(B[row[i]][k].mul(B[row[k]][j]));
            }
          }
        }
      }
      // finished inner reduction
    }
    // end main reduction loop
    // build X for return, unscrambling rows
    for(int i = 0; i<n; i++) {
      X[i] = B[row[i]][n];
    }
  } // end solve

  public static final void invert(Complex[][] A) {
    int n = A.length;
    int[] row = new int[n];
    int[] col = new int[n];
    Complex[] temp = new Complex[n];
    int hold, I_pivot, J_pivot;
    Complex pivot;
    double abs_pivot;
    if(A[0].length!=n) {
      System.out.println("Error in Complex.Matrix.invert,"+ //$NON-NLS-1$
        " matrix not square.");                             //$NON-NLS-1$
    }
    // set up row and column interchange vectors
    for(int k = 0; k<n; k++) {
      row[k] = k;
      col[k] = k;
    }
    // begin main reduction loop
    for(int k = 0; k<n; k++) {
      // find largest element for pivot
      pivot = A[row[k]][col[k]];
      I_pivot = k;
      J_pivot = k;
      for(int i = k; i<n; i++) {
        for(int j = k; j<n; j++) {
          abs_pivot = pivot.abs();
          if(A[row[i]][col[j]].abs()>abs_pivot) {
            I_pivot = i;
            J_pivot = j;
            pivot = A[row[i]][col[j]];
          }
        }
      }
      if(pivot.abs()<1.0E-10) {
        System.out.println("ComplexMatrix is singular !"); //$NON-NLS-1$
        return;
      }
      hold = row[k];
      row[k] = row[I_pivot];
      row[I_pivot] = hold;
      hold = col[k];
      col[k] = col[J_pivot];
      col[J_pivot] = hold;
      // reduce about pivot
      A[row[k]][col[k]] = (new Complex(1.0, 0.0)).div(pivot);
      for(int j = 0; j<n; j++) {
        if(j!=k) {
          A[row[k]][col[j]] = A[row[k]][col[j]].mul(A[row[k]][col[k]]);
        }
      }
      // inner reduction loop
      for(int i = 0; i<n; i++) {
        if(k!=i) {
          for(int j = 0; j<n; j++) {
            if(k!=j) {
              A[row[i]][col[j]] = A[row[i]][col[j]].subtract(A[row[i]][col[k]].mul(A[row[k]][col[j]]));
            }
          }
          A[row[i]][col[k]] = A[row[i]][col[k]].mul(A[row[k]][col[k]]).neg();
        }
      }
    }
    // end main reduction loop
    // unscramble rows
    for(int j = 0; j<n; j++) {
      for(int i = 0; i<n; i++) {
        temp[col[i]] = A[row[i]][j];
      }
      for(int i = 0; i<n; i++) {
        A[i][j] = temp[i];
      }
    }
    // unscramble columns
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        temp[row[j]] = A[i][col[j]];
      }
      for(int j = 0; j<n; j++) {
        A[i][j] = temp[j];
      }
    }
  } // end invert

  public static final Complex determinant(final Complex[][] A) {
    int n = A.length;
    Complex D = new Complex(1.0, 0.0); // determinant
    Complex[][] B = new Complex[n][n]; // working matrix
    int[] row = new int[n];            // row interchange indicies
    int hold, I_pivot;                 // pivot indicies
    Complex pivot;                     // pivot element value
    double abs_pivot;
    if(A[0].length!=n) {
      System.out.println("Error in ComplexMatrix.determinant,"+ //$NON-NLS-1$
        " inconsistent array sizes.");                          //$NON-NLS-1$
    }
    // build working matrix
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        B[i][j] = A[i][j];
      }
    }
    // set up row interchange vectors
    for(int k = 0; k<n; k++) {
      row[k] = k;
    }
    // begin main reduction loop
    for(int k = 0; k<n-1; k++) {
      // find largest element for pivot
      pivot = B[row[k]][k];
      abs_pivot = pivot.abs();
      I_pivot = k;
      for(int i = k; i<n; i++) {
        if(B[row[i]][k].abs()>abs_pivot) {
          I_pivot = i;
          pivot = B[row[i]][k];
          abs_pivot = pivot.abs();
        }
      }
      // have pivot, interchange row indicies
      if(I_pivot!=k) {
        hold = row[k];
        row[k] = row[I_pivot];
        row[I_pivot] = hold;
        D = D.neg();
      }
      // check for near singular
      if(abs_pivot<1.0E-10) {
        return new Complex(0.0, 0.0);
      }
      D = D.mul(pivot);
      // reduce about pivot
      for(int j = k+1; j<n; j++) {
        B[row[k]][j] = B[row[k]][j].div(B[row[k]][k]);
      }
      // inner reduction loop
      for(int i = 0; i<n; i++) {
        if(i!=k) {
          for(int j = k+1; j<n; j++) {
            B[row[i]][j] = B[row[i]][j].subtract(B[row[i]][k].mul(B[row[k]][j]));
          }
        }
      }
      // finished inner reduction
    }
    // end of main reduction loop
    return D.mul(B[row[n-1]][n-1]);
  } // end determinant

  public static final void eigenvalues(final Complex[][] A, Complex[][] V, Complex[] Y) {
    // cyclic Jacobi iterative method of finding eigenvalues
    // advertized for symmetric real
    int n = A.length;
    Complex[][] AA = new Complex[n][n];
    // double norm;  // removed by W. Christian
    Complex c = new Complex(1.0, 0.0);
    Complex s = new Complex(0.0, 0.0);
    if((A[0].length!=n)||(V.length!=n)||(V[0].length!=n)||(Y.length!=n)) {
      System.out.println("Error in ComplexMatrix.eigenvalues,"+ //$NON-NLS-1$
        " inconsistent array sizes.");                          //$NON-NLS-1$
    }
    identity(V); // start V as identity matrix
    copy(A, AA);
    for(int k = 0; k<n; k++) {
      // norm=norm4(AA);  // removed by W. Christian
      for(int i = 0; i<n-1; i++) {
        for(int j = i+1; j<n; j++) {
          schur2(AA, i, j, c, s);
          mat44(i, j, c, s, AA, V);
        }
      } // end one iteration
    }
    // norm = norm4(AA); // final quality check if desired  // removed by W. Christian
    for(int i = 0; i<n; i++) {
      // copy eigenvalues back to caller
      Y[i] = AA[i][i];
    }
  } // end eigenvalues

  public static final void eigenCheck(final Complex[][] A, final Complex[][] V, final Complex[] Y) {
    if((A==null)||(V==null)||(Y==null)) {
      return;
    }
    // check A * X = lambda X lambda=Y[i] X=V[i]
    // check determinant(A- lambda I) = 0
    int n = A.length;
    Complex[][] B = new Complex[n][n];
    Complex[][] C = new Complex[n][n];
    Complex[] X = new Complex[n];
    Complex[] Z = new Complex[n];
    Complex[] T = new Complex[n];
    double norm = 0.0;
    if((A[0].length!=n)||(V.length!=n)||(V[0].length!=n)||(Y.length!=n)) {
      System.out.println("Error in ComplexMatrix.eigenCheck,"+ //$NON-NLS-1$
        " inconsistent array sizes.");                         //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        X[j] = V[j][i];
      }
      mul(A, X, T);
      for(int j = 0; j<n; j++) {
        Z[j] = T[j].subtract(Y[i].mul(X[j]));
      }
      System.out.println("check for near zero norm of Z["+i+"]="+Z[i]); //$NON-NLS-1$ //$NON-NLS-2$
    }
    norm = norm2(Z);
    System.out.println("norm ="+norm+" is eigen vector error indication 1."); //$NON-NLS-1$ //$NON-NLS-2$
    System.out.println("det V = "+ComplexMatrix.determinant(V));              //$NON-NLS-1$
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        Z[j] = V[j][i];
      }
      System.out.println("check for 1.0 = "+norm2(Z)); //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      identity(B);
      mul(B, Y[i], C);
      subtract(A, C, B);
      Z[i] = determinant(B);
    }
    norm = norm2(Z);
    System.out.println("norm ="+norm+" is eigen value error indication."); //$NON-NLS-1$ //$NON-NLS-2$
  }                                                                        // end eigenCheck

  static void schur2(final Complex[][] A, final int p, final int q, Complex c, Complex s) {
    Complex tau;
    Complex tau_tau_1;
    Complex t;
    if(A[0].length!=A.length) {
      System.out.println("Error in schur2 of Complex jacobi,"+ //$NON-NLS-1$
        " inconsistent array sizes.");                         //$NON-NLS-1$
    }
    if(A[p][q].abs()!=0.0) {
      tau = (A[q][q].subtract(A[p][p])).div((A[p][q].mul(2.0)));
      tau_tau_1 = (tau.mul(tau).add(1.0)).sqrt();
      if(tau.abs()>=0.0) {
        t = tau.add(tau_tau_1).invert();
      } else {
        t = (tau_tau_1.subtract(tau)).invert().neg();
      }
      c = (t.mul(t)).add(1.0).sqrt().invert();
      s = t.mul(c);
    } else {
      c = new Complex(1.0, 0.0);
      s = new Complex(0.0, 0.0);
    }
  } // end schur2

  static void mat22(final Complex c, final Complex s, final Complex[][] A, Complex[][] B) {
    if((A.length!=2)||(A[0].length!=2)||(B.length!=2)||(B[0].length!=2)) {
      System.out.println("Error in mat22 of Jacobi, not both 2 by 2"); //$NON-NLS-1$
    }
    Complex[][] T = new Complex[2][2];
    T[0][0] = c.mul(A[0][0]).subtract(s.mul(A[0][1]));
    T[0][1] = s.mul(A[0][0]).add(c.mul(A[0][1]));
    T[1][0] = c.mul(A[1][0]).subtract(s.mul(A[1][1]));
    T[1][1] = s.mul(A[1][0]).add(c.mul(A[1][1]));
    B[0][0] = c.mul(T[0][0]).subtract(s.mul(T[1][0]));
    B[0][1] = c.mul(T[0][1]).subtract(s.mul(T[1][1]));
    B[1][0] = s.mul(T[0][0]).add(c.mul(T[1][0]));
    B[1][1] = s.mul(T[0][1]).add(c.mul(T[1][1]));
  } // end mat2

  static void mat44(final int p, final int q, final Complex c, final Complex s, final Complex[][] A, Complex[][] V) {
    int n = A.length;
    Complex[][] B = new Complex[n][n];
    Complex[][] J = new Complex[n][n];
    if((A[0].length!=n)||(V.length!=n)||(V[0].length!=n)) {
      System.out.println("Error in mat44 of Complex Jacobi,"+ //$NON-NLS-1$
        " A or V not same and square");                       //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        J[i][j] = new Complex(0.0, 0.0);
      }
      J[i][i] = new Complex(1.0, 0.0);
    }
    J[p][p] = c; /* J transpose */
    J[p][q] = s.neg();
    J[q][q] = c;
    J[q][p] = s;
    mul(J, A, B);
    J[p][q] = s;
    J[q][p] = s.neg();
    mul(B, J, A);
    mul(V, J, B);
    copy(B, V);
  }                                        // end mat44

  static double norm4(final Complex[][] A) // for Jacobi
  {
    int n = A.length;
    int nr = A[0].length;
    double nrm = 0.0;
    if(n!=nr) {
      System.out.println("Error in Complex norm4, non square A["+ //$NON-NLS-1$
        n+"]["+nr+"]");                                           //$NON-NLS-1$ //$NON-NLS-2$
    }
    for(int i = 0; i<n-1; i++) {
      for(int j = i+1; j<n; j++) {
        nrm = nrm+A[i][j].abs()+A[j][i].abs();
      }
    }
    return nrm/(n*n-n);
  } // end norm4

  public static final void mul(final Complex[][] A, final Complex[][] B, Complex[][] C) {
    int ni = A.length;
    int nk = A[0].length;
    int nj = B[0].length;
    if((B.length!=nk)||(C.length!=ni)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.mul,"+ //$NON-NLS-1$
        " incompatible sizes");                         //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = new Complex(0.0, 0.0);
        for(int k = 0; k<nk; k++) {
          C[i][j] = C[i][j].add(A[i][k].mul(B[k][j]));
        }
      }
    }
  } // end mul

  public static final void mul(final Complex[][] A, final Complex B, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((C.length!=ni)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.mul,"+ //$NON-NLS-1$
        " incompatible sizes");                         //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = A[i][j].mul(B);
      }
    }
  } // end mul

  public static final void add(final Complex[][] A, final Complex[][] B, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((B.length!=ni)||(C.length!=ni)||(B[0].length!=nj)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.add,"+ //$NON-NLS-1$
        " incompatible sizes");                         //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = A[i][j].add(B[i][j]);
      }
    }
  } // end add

  public static final void add(final Complex[][] A, final Complex B, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((C.length!=ni)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.add,"+ //$NON-NLS-1$
        " incompatible sizes");                         //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = A[i][j].add(B);
      }
    }
  } // end add

  public static final void subtract(final Complex[][] A, final Complex[][] B, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((B.length!=ni)||(C.length!=ni)||(B[0].length!=nj)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.subtract,"+ //$NON-NLS-1$
        " incompatible sizes");                              //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = A[i][j].subtract(B[i][j]);
      }
    }
  } // end subtract

  public static final void subtract(final Complex[][] A, final Complex B, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((C.length!=ni)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.subtract,"+ //$NON-NLS-1$
        " incompatible sizes");                              //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = A[i][j].subtract(B);
      }
    }
  } // end subtract

  public static final double norm1(final Complex[][] A) {
    double norm = 0.0;
    double colSum;
    int ni = A.length;
    int nj = A[0].length;
    for(int j = 0; j<nj; j++) {
      colSum = 0.0;
      for(int i = 0; i<ni; i++) {
        colSum = colSum+A[i][j].abs();
      }
      norm = Math.max(norm, colSum);
    }
    return norm;
  } // end norm1

  public static final double normInf(final Complex[][] A) {
    double norm = 0.0;
    double rowSum;
    int ni = A.length;
    int nj = A[0].length;
    for(int i = 0; i<ni; i++) {
      rowSum = 0.0;
      for(int j = 0; j<nj; j++) {
        rowSum = rowSum+A[i][j].abs();
      }
      norm = Math.max(norm, rowSum);
    }
    return norm;
  } // end normInf

  public static final double normFro(final Complex[][] A) {
    double norm = 0.0;
    int n = A.length;
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        norm = norm+A[i][j].abs()*A[i][j].abs();
      }
    }
    return Math.sqrt(norm);
  } // end normFro

  public static final double norm2(final Complex[][] A) {
    double r = 0.0; // largest eigenvalue
    int n = A.length;
    Complex B[][] = new Complex[n][n];
    Complex V[][] = new Complex[n][n];
    Complex BI[] = new Complex[n];
    if(A[0].length!=n) {
      System.out.println("Error in ComplexMatrix.norm2,"+ //$NON-NLS-1$
        " matrix not square.");                           //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) { // B = A^T * A 
      for(int j = 0; j<n; j++) {
        B[i][j] = new Complex(0.0, 0.0);
        for(int k = 0; k<n; k++) {
          B[i][j] = B[i][j].add(A[k][i].mul(A[k][j]));
        }
      }
    }
    eigenvalues(B, V, BI);
    for(int i = 0; i<n; i++) {
      r = Math.max(r, BI[i].abs());
    }
    return Math.sqrt(r);
  } // end subtract

  public static final void copy(final Complex[][] A, Complex[][] B) {
    int ni = A.length;
    int nj = A[0].length;
    if((B.length!=ni)||(B[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.copy,"+ //$NON-NLS-1$
        " inconsistent sizes.");                         //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        B[i][j] = A[i][j];
      }
    }
  } // end copy

  public static final void deepcopy(final Complex[][] A, Complex[][] B) {
    int ni = A.length;
    int nj = A[0].length;
    if((B.length!=ni)||(B[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.copy,"+ //$NON-NLS-1$
        " inconsistent sizes.");                         //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        B[i][j] = A[i][j].clone();
      }
    }
  } // end copy

  public static final boolean equals(final Complex[][] A, final Complex[][] B) {
    int ni = A.length;
    int nj = A[0].length;
    boolean same = true;
    if((B.length!=ni)||(B[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.equals,"+ //$NON-NLS-1$
        " inconsistent sizes.");                           //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        same = same&&A[i][j].equals(B[i][j]);
      }
    }
    return same;
  } // end equals

  public static final void fromDouble(final double[][] A, final double[][] B, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((C.length!=ni)||(C[0].length!=nj)||(B.length!=ni)||(B[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.fromDouble,"+ //$NON-NLS-1$
        " inconsistent sizes.");                               //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = new Complex(A[i][j], B[i][j]);
      }
    }
  } // end fromDouble

  public static final void fromDouble(final double[][] A, Complex[][] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((C.length!=ni)||(C[0].length!=nj)) {
      System.out.println("Error in ComplexMatrix.fromDouble,"+ //$NON-NLS-1$
        " inconsistent sizes.");                               //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        C[i][j] = new Complex(A[i][j]);
      }
    }
  } // end fromDouble

  public static final void identity(Complex[][] A) {
    int n = A.length;
    if(n!=A[0].length) {
      System.out.println("Error in ComplexMatrix.identity,"+ //$NON-NLS-1$
        " inconsistent sizes.");                             //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      for(int j = 0; j<n; j++) {
        A[i][j] = new Complex(0.0);
      }
      A[i][i] = new Complex(1.0);
    }
  } // end identity

  public static final void zero(Complex[][] A) {
    int ni = A.length;
    int nj = A[0].length;
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        A[i][j] = new Complex(0.0);
      }
    }
  } // end zero

  public static final void print(final Complex[][] A) {
    int ni = A.length;
    int nj = A[0].length;
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        System.out.println("A["+i+"]["+j+"]="+A[i][j]); //$NON-NLS-1$ //$NON-NLS-2$ //$NON-NLS-3$
      }
    }
  }                                                     // end print

  public static final void mul(final Complex[][] A, final Complex[] B, Complex[] C) {
    int ni = A.length;
    int nj = A[0].length;
    if((B.length!=nj)||(C.length!=ni)) {
      System.out.println("Error in ComplexMatrix.mul,"+ //$NON-NLS-1$
        " incompatible sizes.");                        //$NON-NLS-1$
    }
    for(int i = 0; i<ni; i++) {
      C[i] = new Complex(0.0, 0.0);
      for(int j = 0; j<nj; j++) {
        C[i] = C[i].add(A[i][j].mul(B[j]));
      }
    }
  } // end mul

  public static final void add(final Complex[] X, final Complex[] Y, Complex[] Z) {
    int n = X.length;
    if((Y.length!=n)||(Z.length!=n)) {
      System.out.println("Error in ComplexMatrix.add,"+ //$NON-NLS-1$
        " incompatible sizes.");                        //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      Z[i] = X[i].add(Y[i]);
    }
  } // end add

  public static final void subtract(final Complex[] X, final Complex[] Y, Complex[] Z) {
    int n = X.length;
    if((Y.length!=n)||(Z.length!=n)) {
      System.out.println("Error in ComplexMatrix.subtract,"+ //$NON-NLS-1$
        " incompatible sizes.");                             //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      Z[i] = X[i].subtract(Y[i]);
    }
  } // end subtract

  public static final double norm1(final Complex[] X) {
    double norm = 0.0;
    int n = X.length;
    for(int i = 0; i<n; i++) {
      norm = norm+X[i].abs();
    }
    return norm;
  } // end norm1

  public static final double norm2(final Complex[] X) {
    double norm = 0.0;
    int n = X.length;
    for(int i = 0; i<n; i++) {
      norm = norm+X[i].abs()*X[i].abs();
    }
    return StrictMath.sqrt(norm);
  } // end norm2

  public static final double normInf(final Complex[] X) {
    double norm = 0.0;
    int n = X.length;
    for(int i = 0; i<n; i++) {
      norm = Math.max(norm, X[i].abs());
    }
    return norm;
  } // end normInf

  public static final void copy(final Complex[] X, Complex[] Y) {
    int n = X.length;
    if(Y.length!=n) {
      System.out.println("Error in ComplexMatrix.copy,"+ //$NON-NLS-1$
        " incompatible sizes");                          //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      Y[i] = X[i];
    }
  } // end copy

  public static final void deepcopy(final Complex[] X, Complex[] Y) {
    int n = X.length;
    if(Y.length!=n) {
      System.out.println("Error in ComplexMatrix.copy,"+ //$NON-NLS-1$
        " incompatible sizes");                          //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      Y[i] = X[i].clone();
    }
  } // end copy

  public static final boolean equals(final Complex[] X, final Complex[] Y) {
    int n = X.length;
    boolean same = true;
    if(Y.length!=n) {
      System.out.println("Error in ComplexMatrix.equals,"+ //$NON-NLS-1$
        " incompatible sizes");                            //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      same = same&&X[i].equals(Y[i]);
    }
    return same;
  } // end equals

  public static final void fromDouble(final double[] X, Complex[] Z) {
    int n = X.length;
    if(Z.length!=n) {
      System.out.println("Error in ComplexMatrix.fromDouble,"+ //$NON-NLS-1$
        " incompatible sizes");                                //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      Z[i] = new Complex(X[i]);
    }
  } // end fromDouble

  public static final void fromDouble(final double[] X, final double[] Y, Complex[] Z) {
    int n = X.length;
    if((Z.length!=n)||(Y.length!=n)) {
      System.out.println("Error in ComplexMatrix.fromDouble,"+ //$NON-NLS-1$
        " incompatible sizes");                                //$NON-NLS-1$
    }
    for(int i = 0; i<n; i++) {
      Z[i] = new Complex(X[i], Y[i]);
    }
  } // end fromDouble

  public static void fromRoots(Complex[] X, Complex[] Y) {
    int n = X.length;
    if(Y.length!=n+1) {
      System.out.println("Error in ComplexMatrix.fromRoots,"+ //$NON-NLS-1$
        " incompatible sizes");                               //$NON-NLS-1$
    }
    Y[0] = X[0].neg();
    Y[1] = new Complex(1.0);
    if(n==1) {
      return;
    }
    for(int i = 1; i<n; i++) {
      Y[i+1] = new Complex(0.0);
      for(int j = 0; j<=i; j++) {
        Y[i+1-j] = Y[i-j].subtract(Y[i+1-j].mul(X[i]));
      }
      Y[0] = Y[0].mul(X[i]).neg();
    }
  }

  public static final void unitVector(Complex[] X, int j) {
    int n = X.length;
    for(int i = 0; i<n; i++) {
      X[i] = new Complex(0.0);
    }
    X[j] = new Complex(1.0);
  } // end unitVector

  public static final void zero(Complex[] X) {
    int n = X.length;
    for(int i = 0; i<n; i++) {
      X[i] = new Complex(0.0);
    }
  } // end zero

  public static final void print(final Complex[] X) {
    int n = X.length;
    for(int i = 0; i<n; i++) {
      System.out.println("X["+i+"]="+X[i]); //$NON-NLS-1$ //$NON-NLS-2$
    }
  }                                         // end print

  private static String input_file_name;  // for read
  private static BufferedReader in;       // for read
  private static String output_file_name; // for write
  private static BufferedWriter out;      // for write
  private static PrintWriter file_out;    // for write

  public static final void readSize(String file_name, int[] rowCol) {
    String input_line = new String("@"); // unread //$NON-NLS-1$
    int len;                             // input_line length
    int index;                           // start of field
    int last;                            // end of field
    String intStr;                       // for parseInt (does not ignore leading and
    // trailing white space)
    int ni;                              // number of rows
    int nj;                              // number of columns
    if((input_file_name==null)||!file_name.equals(input_file_name)) {
      input_file_name = file_name;
      try {
        in = new BufferedReader(new FileReader(file_name));
      } catch(Exception e) {
        System.out.println("ComplexMatrix.read unable to open file "+file_name); //$NON-NLS-1$
        return;
      }
    }
    ni = 0;
    nj = 0;
    try {
      input_line = in.readLine();                                                     // first read before 'while'
      while(input_line!=null)                                                         // allow leading blank lines
      {
        input_line = input_line.trim();
        len = input_line.length();
        if(len==0) {
          input_line = in.readLine();
          continue;                                                                   // skip blank lines
        }
        if(input_line.charAt(0)=='(') {
          System.out.println("ComplexMatrix.readSize unable to get size "+file_name); //$NON-NLS-1$
          break;
        }
        index = 0;
        last = input_line.indexOf(' ');                                               // first blank after number
        if(last==-1) {
          last = len;
        }
        intStr = input_line.substring(index, last);
        ni = Integer.parseInt(intStr);
        input_line = input_line.substring(last, len);
        input_line = input_line.trim();
        len = input_line.length();
        if(len==0) {
          nj = ni;
          break;
        }
        index = 0;
        last = input_line.indexOf(' ');                                               // first blank after number
        if(last==-1) {
          last = len;
        }
        intStr = input_line.substring(index, last);
        nj = Integer.parseInt(intStr);
        break;
      }                                                                               // end while
    } catch(Exception e) {
      System.out.println("ComplexMatrix.readSize unable to get size "+file_name);     //$NON-NLS-1$
    }
    rowCol[0] = ni;
    rowCol[1] = nj;
  } // end readSize

  public static final void read(String file_name, Complex[][] A) {
    String input_line = new String("@"); // unread //$NON-NLS-1$
    int len;                             // input_line length
    int index;                           // start of field
    int last;                            // end of field
    String intStr;                       // for parseInt (does not ignore leading and
    // trailing white space)
    int i, ni;                           // 0..ni-1
    int j, nj;                           // 0..nj-1
    boolean have_line = false;
    if((input_file_name==null)||!file_name.equals(input_file_name)) {
      input_file_name = file_name;
      try {
        in = new BufferedReader(new FileReader(file_name));
      } catch(Exception e) {
        System.out.println("ComplexMatrix.read unable to open file "+file_name); //$NON-NLS-1$
        return;
      }
    }
    ni = 0;
    nj = 0;
    try {
      input_line = in.readLine();                                             // first read before 'while'
      while(input_line!=null)                                                 // allow leading blank lines
      {
        input_line = input_line.trim();
        len = input_line.length();
        if(len==0) {
          input_line = in.readLine();
          continue;                                                           // skip blank lines
        }
        if(input_line.charAt(0)=='(') {
          ni = A.length;                                                      // no size, just complex data
          nj = A[0].length;
          have_line = true;
          break;
        }
        index = 0;
        last = input_line.indexOf(' ');                                       // first blank after number
        if(last==-1) {
          last = len;
        }
        intStr = input_line.substring(index, last);
        ni = Integer.parseInt(intStr);
        input_line = input_line.substring(last, len);
        input_line = input_line.trim();
        len = input_line.length();
        if(len==0) {
          nj = ni;
          break;
        }
        index = 0;
        last = input_line.indexOf(' ');                                       // first blank after number
        if(last==-1) {
          last = len;
        }
        intStr = input_line.substring(index, last);
        nj = Integer.parseInt(intStr);
        break;
      }                                                                       // end while
    } catch(Exception e) {
      System.out.println("ComplexMatrix.read unable to get size "+file_name); //$NON-NLS-1$
    }
    // now read complex numbers
    i = 0;
    j = 0;
    if((A.length!=ni)||(A[0].length!=nj)) {
      System.out.println("incompatible size in ComplexMatrix.read"); //$NON-NLS-1$
      return;
    }
    try {
      if(!have_line) {
        input_line = in.readLine();                                            // first read before 'while'
      }
      have_line = false;
      while(input_line!=null) {
        input_line = input_line.trim();
        len = input_line.length();
        if(len==0) {
          input_line = in.readLine();
          continue;                                                            // skip blank lines
        }
        index = 0;
        last = input_line.indexOf(')');                                        // closing )
        if(last==-1) {
          input_line = in.readLine();
          continue;
        }
        intStr = input_line.substring(index, last+1);
        A[i][j] = Complex.parseComplex(intStr);
        j++;
        if(j==nj) {
          j = 0;
          i++;
        }
        if(i==ni) {
          break;
        }
        input_line = input_line.substring(last+1);
      }                                                                        // end while
    } catch(Exception e) {
      System.out.println("ComplexMatrix.read unable to read data "+file_name); //$NON-NLS-1$
    }
  }                                                                            // end read

  public static final void closeInput() {
    try {
      in.close();
    } catch(Exception e) {
      System.out.println("ComplexMatrix.closeInput not closed"); //$NON-NLS-1$
    }
    input_file_name = null;
  } // end closeInput

  public static final void write(String file_name, Complex[][] A) {
    int ni = A.length;
    int nj = A[0].length;
    if((output_file_name==null)||!file_name.equals(output_file_name)) {
      output_file_name = file_name;
      try {
        out = new BufferedWriter(new FileWriter(file_name));
        file_out = new PrintWriter(out);
      } catch(Exception e) {
        System.out.println("ComplexMatrix.write unable to open file "+file_name); //$NON-NLS-1$
        return;
      }
    }
    // write size
    if(ni==nj) {
      file_out.println(ni);
    } else {
      file_out.println(ni+" "+nj); //$NON-NLS-1$
    }
    // now write complex numbers
    try {
      for(int i = 0; i<ni; i++) {
        for(int j = 0; j<nj; j++) {
          file_out.println(A[i][j].toString());
        }
      }
      file_out.println();
    } catch(Exception e) {
      System.out.println("ComplexMatrix.write unable to write data "+file_name); //$NON-NLS-1$
    }
  }                                                                              // end write

  public static final void closeOutput() {
    file_out.close();
    output_file_name = null;
  } // end closeOutput
   
  public static final void tensormultiply(
           final Complex[][] A, final Complex[][] B, Complex[][] C) {
    // C = A \otimes B
    int ni = A.length;
    int nj = A[0].length;
    int nk = B.length;
    int nm = B[0].length;
    for(int i = 0; i<ni; i++) {
      for(int j = 0; j<nj; j++) {
        for(int k = 0; k<nk; k++) {
          for(int m = 0; m<nm; m++) {
            C[i*nk+k][j*nm+m] = A[i][j].mul(B[k][m]);
          }
        }
      }
    }
  }

  public static final void conjugatetranspose(final Complex[][] A, 
                                              final Complex[][] B) {
    // B = A^{\dagger} 
    int n = A.length;
    for (int i=0; i<A.length; i++) {
      for (int j=0; j<A[0].length; j++) {
        B[i][j] = (A[j][i]).conjugate();
      }
    }
  }

  public static Complex[][] myAdd(Complex[][] m1, Complex[][] m2) {
    Complex[][] sum = newSquareMatrix(m1.length);
    add(m1,m2,sum);
    return sum;
  }

  public static Complex[][] mySubtract(Complex[][] m1, Complex[][] m2) {
    Complex[][] diff = newSquareMatrix(m1.length);
    subtract(m1,m2,diff);
    return diff;
  }

  public static Complex[] mySubtract(Complex[] m1, Complex[] m2) {
    Complex[] diff = new Complex[m1.length];
    subtract(m1,m2,diff);
    return diff;
  }

  public static Complex[][] myMul(Complex[][] m1, Complex[][] m2) {
    Complex[][] product = newSquareMatrix(m1.length);
    mul(m1,m2,product);
    return product;
  }

  public static Complex[][] myMul(Complex[][] m1, 
                                  Complex[][] m2, 
                                  Complex[][] m3) {
    Complex[][] product12  = newSquareMatrix(m1.length);
    Complex[][] product123 = newSquareMatrix(m1.length);
    mul(m1,m2,product12);
    mul(product12,m3,product123);
    return product123;
  }

  public static Complex[][] myConjTrans(Complex[][] m) {
    Complex[][] mDagger = newSquareMatrix(m.length);
    conjugatetranspose(m,mDagger);
    return mDagger;
  }

  private static Complex[][] newTwoByTwo() {
    Complex[][] m =
        {{new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0)}};
    return m;
  }

  private static Complex[][] newFourByFour() {
    Complex[][] m =
        {{new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)}};
    return m;
  }

  private static Complex[][] newEightByEight() {
    Complex[][] m =
        {{new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0),
          new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)}};
    return m;
  }

  public static Complex[][] newSquareMatrix(int n) {
    if (n<2) {
      throw new NullPointerException("Unsupported matrix dimension");
    }
    if (n==2) {
      return newTwoByTwo();
    }
    else 
    if (n==4) {
      return newFourByFour();
    }
    else
    if (n==8) {
      return newEightByEight();
    }
    else {
      Complex[][] result = new Complex[n][];
      for (int i=0; i<n; i++) { 
          result[i] = new Complex[n]; 
          for (int j=0; j<n; j++) {
              result[i][j] = new Complex(0);
          }
      }
      return result;
    }
  }

  public static Complex innerproduct(Complex[] u, Complex[] v) {
    Complex result = new Complex(0);
    for (int i=0; i<u.length; i++) {
      result = result.add( (u[i].conjugate()).mul(v[i]) );
    }
    return result;
  }

  public static double selfinnerproduct(Complex[] u) {
    double result = 0;
    for (int i=0; i<u.length; i++) {
      result = result + u[i].abs2();
    }
    return result;
  }

  public static Complex[] div(Complex[] v, double d) {
    int len = v.length;
    Complex[] result = new Complex[len];
    for (int i=0; i<len; i++) {
      result[i] = v[i].div(d);
    }
    return result;
  }

  public static Complex[][] myTranspose(Complex[][] m) {
    Complex[][] result = newSquareMatrix(m.length);
    for (int i=0; i<m.length; i++) {
      for (int j=0; j<m.length; j++) {
        result[i][j] = m[j][i];
      }
    }
    return result;
  }

  public static Complex[][] rows2columns(Complex[][] m) {
    return myTranspose(m);
  }

  public static Complex[][] columns2rows(Complex[][] m) {
    return myTranspose(m);
  }

  public static double doublenorm(Complex[] v) {
    return Math.sqrt(selfinnerproduct(v));
  }

  public static Complex[] proj(Complex[] u, Complex[] v) {
    if (doublenorm(u) < staticanalysis.StaticAnalysis.epsilonZeroVector) {   
       return myZeroVec(u.length);
    }
    else {
       Complex n = innerproduct(u,v);
       double d = selfinnerproduct(u);
       Complex f = n.div(d);
       Complex[] result = new Complex[u.length];
       for (int i=0; i<u.length; i++) {
         result[i] = u[i].mul(f);
       }
       return result;
    }
  }

  public static Complex[][] GramSchmidt(Complex[][] v) {
    // Maps a nonempty list v of vectors to a list e of orthonormal vectors
    // such that span(v) = span(e).  Notice: e.filled <= v.length.
    // The dimension of each vector is v[0].length.

    // Step 1, from v to u:
    //   u[1] = v[1]
    //   u[2] = v[2] - proj(u[1],v[2])
    //   u[3] = v[3] - proj(u[1],v[3]) - proj(u[2],v[3])
    //   ...
    //   u[k] = v[k] - Sum_{j=1}^{k-1} proj(u[j],v[k])
    Complex[][] u = newSquareMatrix(v[0].length);
    deepcopy(v,u);
    for (int i=0; i<v.length; i++) {
         for (int j=0; j<i; j++) {
            u[i] = mySubtract(u[i], proj(u[j],v[i]));
         }
    }

    // Step 2, from u to e:
    //   for all i:  e[i] = u[i]/norm(u[i])
    Complex[][] e = newSquareMatrix(v[0].length);
    int j=0; // the next index in e to be filled
    for (int i=0; i<u.length; i++) {
         double n = doublenorm(u[i]); 
         if (n > staticanalysis.StaticAnalysis.epsilonGramSchmidt) { 
             // is u[i] close to be (0,0,...,0)?
            e[j] = div(u[i],n);
            j = j+1;
         }
    }
   
    return e;
  }

  public static Complex[][] supp(Complex[][] m) {
    Complex[][] v = rows2columns(m);
    Complex[][] e = GramSchmidt(v); // column matrix -> column matrix
    Complex[][] eRows = columns2rows(e);

    return myMul( eRows, myConjTrans(eRows) );
  }

  public static int GramSchmidtThenRank(Complex[][] v) {
    // Maps a nonempty list v of vectors to a list e of orthonormal vectors
    // such that span(v) = span(e).  Notice: e.filled <= v.length.
    // The dimension of each vector is v[0].length.

    // Step 1, from v to u:
    //   u[1] = v[1]
    //   u[2] = v[2] - proj(u[1],v[2])
    //   u[3] = v[3] - proj(u[1],v[3]) - proj(u[2],v[3])
    //   ...
    //   u[k] = v[k] - Sum_{j=1}^{k-1} proj(u[j],v[k])
    Complex[][] u = newSquareMatrix(v[0].length);
    deepcopy(v,u);
    for (int i=0; i<v.length; i++) {
         for (int j=0; j<i; j++) {
            u[i] = mySubtract(u[i], proj(u[j],v[i]));
         }
    }

    // Step 2, from u to rank:
    //   for all i:  e[i] = u[i]/norm(u[i]), is u[i] close to be (0,0,...,0)?
    int j=0; // was: the next index in e to be filled
    for (int i=0; i<u.length; i++) {
         double n = doublenorm(u[i]);
         if (n > staticanalysis.StaticAnalysis.epsilonGramSchmidt) {
             // is u[i] close to be (0,0,...,0)?
            j = j+1;
         }
    }

    return j;
  }


  public static int rank(Complex[][] m) {
    Complex[][] v = rows2columns(m);
    return GramSchmidtThenRank(v); // column matrix -> rank
  }

  public static String toString(Complex[][][][] M) {
    // works when M is a matrix of 4x4 matrices
    String result = "";
    for (int i = 0; i < M.length; i++) {
      for (int j = i+1; j < M.length; j++) {
         result = result + "\n" + "Matrix(" + i + "," + j + "): \n" +
                           toString(M[i][j]);
      }
    }
    return result;
  }

  public static String toString(Complex[][] M) {
    String result = "";
    for (int k = 0; k < M.length; k++) {
       result = result + toString(M[k]) + "\n";
    }
    return result;
  }

  public static String toString(Complex[] vector) {
    String result = "";
    for (int m = 0; m < vector.length; m++) {
          String complexNumber = vector[m].toString();
          int l = complexNumber.length();
          String blanks = "                   ";
          String padding = blanks.substring(0,20-l);
          result = result + complexNumber + padding;
    }
    return result;
  }

  public static Complex[][] String2matrix(String g, boolean descending) {
       Complex[][] H =
           {{new Complex( 1.0/Math.sqrt(2)), new Complex( 1.0/Math.sqrt(2))},
            {new Complex( 1.0/Math.sqrt(2)), new Complex(-1.0/Math.sqrt(2))}};
       Complex[][] T =
           {{new Complex( 1), new Complex( 0)},
            {new Complex( 0), new Complex(1.0/Math.sqrt(2), 1.0/Math.sqrt(2))}};
       Complex[][] D =
           {{new Complex( 1), new Complex( 0)},
            {new Complex( 0), new Complex(1.0/Math.sqrt(2),-1.0/Math.sqrt(2))}};
       Complex[][] S =
           {{new Complex( 1), new Complex( 0)},
            {new Complex( 0), new Complex( 0,1)}};
       Complex[][] X =
           {{new Complex( 0), new Complex( 1)},
            {new Complex( 1), new Complex( 0)}};
       Complex[][] Z =
           {{new Complex( 1), new Complex( 0)},
            {new Complex( 0), new Complex(-1)}};

       Complex[][] CNOT = // if 1, then flip
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)}};
       Complex[][] invCNOT = // if 1, then flip, but in reverse order
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)}};
       Complex[][] CZ = 
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex(-1)}};

       Complex[][] CCNOT = // if 1,1 then flip
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)}};
       Complex[][] invCCNOT = // if 1,1 then flip, but in reverse order; invCCNOT = (H \otimes I \otimes H) CCNOT (H \otimes I \otimes H)
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)}};

       Complex[][] NCCNOT = // if 0,1 then flip
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)}};
       Complex[][] invNCCNOT = // if 0,1 then flip, but in reverse order; invNCCNOT = (HX \otimes I \otimes XH) NCCNOT (XH \otimes I \otimes HX)
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)}};

       Complex[][] CNCNOT = // if 1,0 then flip
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)}};
       Complex[][] invCNCNOT = // if 1,0 then flip; but in reverse order; invCNCNOT = (H \otimes I \otimes H) CNCNOT (H \otimes I \otimes H)
        {{new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)}};

       Complex[][] NCNCNOT = // if 0,0 then flip
        {{new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)}};
       Complex[][] invNCNCNOT = // if 0,0 then flip, but in reverse order; invNCNCNOT = (HX \otimes I \otimes XH) NCNCNOT (XH \otimes I \otimes HX)
        {{new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 1), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 0), new Complex( 1)}};
    

       if (g.equals("H")) {
          return H;
       }
       else
       if (g.equals("T")) {
          return T;
       }   
       else 
       if (g.equals("D")) {
          return D;
       }    
       else
       if (g.equals("S")) {
          return S;
       }
       else
       if (g.equals("X")) {
          return X;
       }   
       else 
       if (g.equals("Z")) {
          return Z;
       }
       else
       if (g.equals("H")) {
          return H;
       }
       else
       if (g.equals("T")) {
          return T;
       }   
       else 
       if (g.equals("D")) {
          return D;
       }    
       else
       if (g.equals("S")) {
          return S;
       }
       else
       if (g.equals("X")) {
          return X;
       }   
       else 
       if (g.equals("Z")) {
          return Z;
       }
       else
       if (g.equals("CNOT")) { 
          if (descending) {
             return CNOT; 
          }
          else {
             return invCNOT;
          }
       }
       else
       if (g.equals("CZ")) {
          return CZ;
       }
       if (g.equals("CCNOT")) {
          if (descending) {
             return CCNOT;
          }
          else {
             return invCCNOT;
          }
       }
       if (g.equals("NCCNOT")) {
          if (descending) {
             return NCCNOT;
          }
          else {
             return invNCCNOT;
          }
       }
       if (g.equals("CNCNOT")) {
          if (descending) {
             return CNCNOT;
          }        
          else {
             return invCNCNOT;
          }
       }
       if (g.equals("NCNCNOT")) {
          if (descending) {
             return NCNCNOT;
          }    
          else {
             return invNCNCNOT;
          }
       }

       System.out.println("Warning: this should never happen");
       return H;
  }

  public static Complex[][] Id2() {
    Complex[][] m =
        {{new Complex( 1), new Complex( 0)},
         {new Complex( 0), new Complex( 1)}};
    return m;
  }

  public static Complex[] myZeroVec(int n) {
    Complex[] result = new Complex[n];
    zero(result);
    return result;
  }

  private static void Exchange2inPlace(Complex[][] m132, int x, int y) {
    // exchange rows x,y
    Complex[] row = m132[x];
          m132[x] = m132[y];
          m132[y] = row;
  
    // exchange columns x,y
    for (int i=0; i<m132.length; i++) {
       Complex c = m132[i][x];
      m132[i][x] = m132[i][y];
      m132[i][y] = c;
    }
  }

  private static void swap0andk(Complex[][] m, int k) {
    // assume: m.length is a power of 2
    // assume: k is counting from 0 to m.length-1
    // maps an nxn matrix to an nxn matrix, where n = m.length
    // It is like maps, ijk->jik, for all i,j,k being 0 or 1.
    // In particular, swap0andk(m,1) exchanges
    // 0:  000<--->000
    // 1:  001<--->001
    // 2:  010<--->100 !!
    // 3:  011<--->101 &&
    // 4:  100<--->010 !!
    // 5:  101<--->011 &&
    // 6:  110<--->110
    // 7:  111<--->111
    // In other words, it exchanges
    // the 3rd and 5th rows, and columns, and
    // the 4th and 6th rows, and columns.

    if ((0 < k) && (Math.pow(2,k) < m.length)) {
       int numberOfQubits = (int) (Math.log(m.length) / Math.log(2));
       int half = m.length / 2;
       int stride = (int) ( half / Math.pow(2,k) );
       int steps = (int) ( half / stride );
       for (int i=0; i<steps; i++) {
           for (int j=0; j<stride; j++) {
               if (i % 2 == 1) {
                  int pos = i*stride + j;
                  int otherPos =
                        pos + half - (int) (Math.pow(2,numberOfQubits-k-1));
                  Exchange2inPlace(m, pos, otherPos);
               }
           }
       }
    }
  }

  private static void swapkandkplus1(Complex[][] m, int k) {
    if (k==0) {
      swap0andk(m,1);
    }
    else {
      swap0andk(m,k);
      swap0andk(m,k+1);
      swap0andk(m,k);
    }
  }

  private static void move0tok(Complex[][] m, int k) {
    for (int i=0; i<k; i++) {
        swapkandkplus1(m,i);
    }
  }

  private static void movekto0(Complex[][] m, int k) {
    if (0<k) {
      for (int i=k-1; 0<=i; i--) {
          swapkandkplus1(m,i);
      }
    }
  }

  public static Complex[][] traceout(Complex[][] U, 
                                     List<Integer> indices,
                                     int baseSize) {
    Complex[][] result = U;
    for (int i=0; i<indices.size(); i++) {
      result = traceoutAt(result,(baseSize-1)-indices.get(i));
    }
    return result;
  }

  private static Complex[][] traceoutAt(Complex[][] mabc, int t) {
    // maps a ( 2^(n+1) x 2^(n+1) ) matrix to a (2^n x 2^n) matrix
    // assume 0<=t

    if (t==0) {
      // We first write M_{1,2,...,n} into block matrix
      // M_{11}  M_{12}
      // M_{21}  M_{22}
      // where each M_{i,j} is a 2^(n-1) x 2^(n-1) matrix.
      // Now: traceoutAt(M_{1,2,...,n},1) = M_{11} + M_{22}.

      int half = (int) (mabc.length / 2);
      Complex[][] result = newSquareMatrix(half);
      for (int i=0; i<half; i++) {
        for (int j=0; j<half; j++) {
          result[i][j] = mabc[i][j].add( mabc[i+half][j+half] );
        }
      }
      return result;
    }
    else
    if (t>0) {
      // Step 1. move t to position 0
      // Step 2. traceout the current 0.

      Complex[][] result = newSquareMatrix(mabc.length);
      deepcopy(mabc,result);
      movekto0(result,t);
      return traceoutAt(result,0);
    }
    else {
      throw new RuntimeException("Problem in traceout");
    }
  }

  private static Complex[][] expandAt(Complex[][] U, int k) {
    Complex[][] result = newSquareMatrix(2*U.length);
    int exp2k = staticanalysis.StaticAnalysis.exp2(k);
    if (exp2k == 0) {
       tensormultiply(myId(2),U,result);   
    }
    else
    if (exp2k < U.length) {
       tensormultiply(myId(2),U,result);
       move0tok(result,k);
    } 
    else {
       tensormultiply(U,myId(2),result);
    }
    return result;
  }

  public static Complex[][] expand(Complex[][] U, List<Integer> indices) {
    // assert U.length = indices.size() - 1
 
    // System.out.println("In expand: indices = " + indices);
    // System.out.println("Should be equal: U.length = " + U.length + " and " +
    //                         "indices.size()-1 = " + (indices.size()-1));

    Complex[][] result = U;
    for (int i=0; i<indices.size(); i++) {
      for (int x=0; x<indices.get(i); x++) {
        result = expandAt(result,(indices.size()-1)-i);
      }
    }
    return result;
  }

  public static final Complex[][] myId(int n) {
    Complex[][] result = newSquareMatrix(n);   
    identity(result);
    return result;
  }

  public static Complex[][] intersectProjections(List<Complex[][]> l) {
    // the intersection of the projections l.0, ..., l.(k-1)
    //    (I \otimes ... \otimes I) -
    //    supp( k(I \otimes I \times I) - (l.0 + ... + l.(k-1)) )
    // assume l.size() > 0

    // System.out.println("At intersectProjections: ");
    // for (Complex[][] m: l) {
    //   System.out.println(ComplexMatrix.toString(m));
    // }

    int n = l.get(0).length;
    int k = l.size();

    Complex[][] kI = myId(n);
    for (int i=0; i<n; i++) {
        kI[i][i] = kI[i][i].mul(k);
    }

    Complex[][] sum = newSquareMatrix(n);
    for (Complex[][] m: l) {
      sum = myAdd(sum,m);
    }

    return mySubtract(
             myId(n),
             supp( mySubtract( kI, sum ) ));
  }

  public static boolean subsetCheckForProjections(Complex[][] p, 
                                                  Complex[][] q) {
    Complex[][] diff =
           ComplexMatrix.mySubtract( ComplexMatrix.myMul(q,p), p );
    double n1 = ComplexMatrix.norm1(diff);
    if (staticanalysis.StaticAnalysis.whyWrong &&
        (n1 > staticanalysis.StaticAnalysis.epsilonAbstractState)) {
       System.out.println( ComplexMatrix.toString(p) + "is different from\n" +
                           ComplexMatrix.toString(q) );
    }
    return (n1 < staticanalysis.StaticAnalysis.epsilonAbstractState);
  }

} // end class ComplexMatrix

/*
 * Open Source Physics software is free software; you can redistribute
 * it and/or modify it under the terms of the GNU General Public License (GPL) as
 * published by the Free Software Foundation; either version 2 of the License,
 * or(at your option) any later version.
 * Code that uses any portion of the code in the org.opensourcephysics package
 * or any subpackage (subdirectory) of this package must must also be be released
 * under the GNU GPL license.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
 * or view the license online at http://www.gnu.org/copyleft/gpl.html
 *
 * Copyright (c) 2007  The Open Source Physics project
 *                     http://www.opensourcephysics.org
 */
