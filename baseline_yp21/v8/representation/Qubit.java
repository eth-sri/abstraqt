package representation;

import matrix.Complex;

public class Qubit {
  public Complex amplitude0;
  public Complex amplitude1;

  public Qubit(Complex amplitude0, Complex amplitude1) {
    this.amplitude0 = amplitude0;
    this.amplitude1 = amplitude1;
  }

  public Complex[] asVector() {
    Complex[] result = new Complex[2];
    result[0] = amplitude0;
    result[1] = amplitude1;
    return result;
  }

  public String toString() {
    double epsilon = 0.00001;
    if (amplitude1.abs() < epsilon) {
       return "|0>";
    } 
    else
    if (amplitude0.abs() < epsilon) {
       return "|1>";
    }
    else {
       return "(" + amplitude0.toString() + "|0> + " + 
                    amplitude1.toString() + "|1>)";
    }
  }
}
