package staticanalysis;

import representation.Qubit;
import matrix.*;
import java.util.List;
import java.util.ArrayList;

public class AbstractState {

  public Domain domain;
  List<Complex[][]> range = new ArrayList<Complex[][]>();

  public AbstractState(Domain domain, InitialValueGenerator initValGen) {
    this.domain = domain;
    for (int i=0; i<domain.size(); i++) {
      this.range.add(initValGen.operation(i));
    }
  }

/*
  public void put(List<Integer> key, Complex[][] value) {
    // assume key in this.domain
    int i = domain.indexOf(key);
    this.range.set(i,value);
  }

  public Complex[][] get(List<Integer> key) {
    // assume key in this.domain
    int i = domain.indexOf(key);
    return this.range.get(i);
  }
*/

  public Complex[][] get(int i) {
    return this.range.get(i);
  }
 
  public boolean subsetOf(AbstractState other) {
    // assume this,other are defined over the same n,k
    for (int i=0; i<this.domain.size(); i++) {
        if (!(ComplexMatrix.subsetCheckForProjections(
                 this.range.get(i), other.range.get(i)))) {
           return false;
        }
    }
    return true;
  }


  public boolean valid() {
    boolean result = true;

    // check that no matrix is close to be 0
    double minNorm1 = 1;
    for (int i=0; i<this.domain.size(); i++) {
        double current = ComplexMatrix.norm1( this.range.get(i) );
        if (current < minNorm1) {
           minNorm1 = current;
           if (current < staticanalysis.StaticAnalysis.epsilonZeroMatrix) {
              result = false;
              System.out.println("The matrix for " + this.domain.get(i) + 
                                 " is close to 0:");
              if (this.range.get(i).length <= 8) {
                 System.out.println("The matrix is:\n" +
                     ComplexMatrix.toString( this.range.get(i) ));
              }
              else {
                 System.out.println("The matrix has dimension " +
                       this.range.get(i).length + " x " +
                       this.range.get(i).length);
              }
           }
        }
    }

    // check that every matrix P satisfies P x P = P
    for (int i=0; i<this.domain.size(); i++) {
        Complex[][] current = this.range.get(i);
        if (ComplexMatrix.norm1(
             ComplexMatrix.mySubtract(ComplexMatrix.myMul(
                                         this.range.get(i), 
                                         this.range.get(i)),
                               this.range.get(i))) > 
                               staticanalysis.StaticAnalysis.epsilonProjection) {
           result = false;
           System.out.println("The matrix for " + this.domain.get(i) + 
                              " is not a projection:");
           if (this.range.get(i).length <= 8) {
              System.out.println("The matrix is:\n" +
                     ComplexMatrix.toString( this.range.get(i) ) +
                              "\n" +
                              "indeed, its product with itself is:\n" +
                              ComplexMatrix.toString( 
                                      ComplexMatrix.myMul(
                                         this.range.get(i),
                                         this.range.get(i))));
           }
           else {
                 System.out.println("The matrix has dimension " +
                       this.range.get(i).length + " x " +
                       this.range.get(i).length);
           }
        }
    }

    for (int i=0; i<this.domain.size(); i++) {
        // check that P dagger = P
        Complex[][] current = this.range.get(i);
        if (ComplexMatrix.norm1(
             ComplexMatrix.mySubtract(ComplexMatrix.myConjTrans(
                                         this.range.get(i)),
                                this.range.get(i))) > 
                                staticanalysis.StaticAnalysis.epsilonHermetian) {
           System.out.println("The matrix for " + this.domain.get(i) + 
                              " is not Hermetian:\n" + 
                                  ComplexMatrix.toString( this.range.get(i) ));
           result = false;
        }
    }

    return result;
  }

  public List<Integer> ranks() {
    List<Integer> result = new ArrayList<Integer>();
    for (int i=0; i<this.domain.size(); i++) {
        result.add( ComplexMatrix.rank(this.range.get(i)) );
    }
    return result;
  }

  public String toString() {
    String result = "";
    for (int i=0; i<this.domain.size(); i++) {
        result = result + "\n" + "Matrix" + this.domain.get(i) + "\n" +
                          ComplexMatrix.toString(this.range.get(i));
    }
    return result;
  }
}

