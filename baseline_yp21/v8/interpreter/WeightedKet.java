package interpreter;
import matrix.*;
import java.util.BitSet;
import java.util.Comparator;

public class WeightedKet {
  public int size;
  public Complex amplitude;
  BitSet ket;

  public WeightedKet(int size) {
         this.size = size;
    this.amplitude = new Complex(1.0,0.0);
          this.ket = new BitSet(size);
  }

  public WeightedKet(int size, Complex amplitude, BitSet ket) {
         this.size = size;
    this.amplitude = amplitude;
          this.ket = ket;
  }

  public void merge(WeightedKet wk) {
    this.amplitude = this.amplitude.add(wk.amplitude);
  }

  public String toString() {
    double epsilon = 0.00001;
    String bitstring = "";
    for (int i=this.size-1; i>=0; i--) {
      bitstring = bitstring + (this.ket.get(i) ? "1" : "0");
    }
    Complex ampl = this.amplitude;
    if ((0.5 - epsilon < ampl.re()) && (ampl.re() < 0.5 + epsilon) &&
        (    - epsilon < ampl.im()) && (ampl.im() < epsilon)) {
       ampl = new Complex(0.5,0.0);
    }
    else 
    if ((-0.5 - epsilon < ampl.re()) && (ampl.re() < -0.5 + epsilon) &&
        (     - epsilon < ampl.im()) && (ampl.im() < epsilon)) {
       ampl = new Complex(-0.5,0.0);
    }
    else 
    if ((1.0 - epsilon < ampl.re()) && (ampl.re() < 1.0 + epsilon) &&
        (    - epsilon < ampl.im()) && (ampl.im() < epsilon)) {
       ampl = new Complex(1.0,0.0);
    }
    return ampl.toString() + " * " + "|" + bitstring + ">";
  }

  public String bitsToString(int lo, int hi) {
    String s = "";
    for (int i=hi; i>=lo; i--) {
      s = s + (this.ket.get(i) ? "1" : "0");
    }
    return s;
  }
}

