package interpreter;

import representation.*;
import matrix.*;
import java.util.ArrayList;
import java.util.BitSet;

public class Interpreter {
  Program p;

  public Interpreter(Program p) {
    this.p = p;
  }

  ArrayList<WeightedKet> state = new ArrayList<WeightedKet>();

  public void run() {
    this.state.add(new WeightedKet(this.p.getSize()));
    printState(); System.out.println(); System.out.println();

    for (Operation op : p.getOps()) {
      System.out.println("Apply " + op.toString() + ":");
      step(op);
      // System.out.print("After applying:     "); printState(); System.out.println();
      state.sort(new MyComparator());
      // System.out.print("After sorting:      "); printState(); System.out.println();
      consolidate();
      // System.out.print("After consolidating "); printState(); System.out.println();
      printState(); System.out.println(); System.out.println();
    }
      
    // printState(); System.out.println(); System.out.println();

    measure();
  }

  void step(Operation op) {
    ArrayList<WeightedKet> extraState = new ArrayList<WeightedKet>();
    for (WeightedKet wk : state) {
       if (op instanceof OneQubitOperation) {
          OneQubitOperation op1 = (OneQubitOperation) op;
          if (op1.gate.equals("H")) {
             Complex wka = wk.amplitude;
             Complex oneDivSqrt2 = new Complex(1.0/Math.sqrt(2),0.0);
             if (wk.ket.get(op1.arg)) {
                wk.amplitude = new Complex(0.0,0.0).subtract(
                                     wka.mul(oneDivSqrt2));
             }
             else {
                wk.amplitude = wka.mul(oneDivSqrt2);
             }
             BitSet bs = (BitSet) wk.ket.clone();
             bs.flip(op1.arg);
             extraState.add(new WeightedKet(
                                  wk.size,  
                                  wka.mul(oneDivSqrt2),
                                  bs));
          }
          else
          if (op1.gate.equals("T")) {
             // https://en.wikipedia.org/wiki/Quantum_logic_gate
             // e^{i pi/4} = cos(pi/4) + i * sin(pi/4) = (1+i)/sqrt(2)
             Complex eipi4 = new Complex(1.0/Math.sqrt(2),1.0/Math.sqrt(2));
             if (wk.ket.get(op1.arg)) {
                wk.amplitude = wk.amplitude.mul(eipi4);
             }
          }
          else
          if (op1.gate.equals("D")) {
             // D = T dagger
             Complex eipi4dagger = 
                   new Complex(1.0/Math.sqrt(2),-1.0/Math.sqrt(2));
             if (wk.ket.get(op1.arg)) {
                wk.amplitude = wk.amplitude.mul(eipi4dagger);
             }
          }
          else
          if (op1.gate.equals("S")) {
             // S = T^2 = (1, 0; 0, i)
             Complex i = new Complex(0,1.0);
             if (wk.ket.get(op1.arg)) {
                wk.amplitude = wk.amplitude.mul(i);
             }
          }
          else
          if (op1.gate.equals("X")) {
             wk.ket.flip(op1.arg);
          }
          else 
          if (op1.gate.equals("Z")) {
             if (wk.ket.get(op1.arg)) {
                wk.amplitude = new Complex(0.0,0.0).subtract(wk.amplitude);
             }
          }
       } 
       else 
       if (op instanceof TwoQubitOperation) {
          TwoQubitOperation op2 = (TwoQubitOperation) op;
          if (op2.gate.equals("CNOT")) {
             if (wk.ket.get(op2.arg1)) {
                wk.ket.flip(op2.arg2);
             }
          }
          else
          if (op2.gate.equals("CZ")) {
             if (wk.ket.get(op2.arg1)) {
                if (wk.ket.get(op2.arg2)) {
                   wk.amplitude = new Complex(0.0,0.0).subtract(wk.amplitude);
                }
             }
          }
       }
       if (op instanceof ThreeQubitOperation) {
          ThreeQubitOperation op3 = (ThreeQubitOperation) op;
          if (op3.gate.equals("CCNOT")) {
             if ((wk.ket.get(op3.arg1)) && (wk.ket.get(op3.arg2))) {
                wk.ket.flip(op3.arg3);
             }
          }
          else
          if (op3.gate.equals("NCCNOT")) {
             if ((!wk.ket.get(op3.arg1)) && (wk.ket.get(op3.arg2))) {
                wk.ket.flip(op3.arg3);
             }
          }
          else
          if (op3.gate.equals("CNCNOT")) {
             if ((wk.ket.get(op3.arg1)) && (!wk.ket.get(op3.arg2))) {
                wk.ket.flip(op3.arg3);
             }
          }
          else
          if (op3.gate.equals("NCNCNOT")) {
             if ((!wk.ket.get(op3.arg1)) && (!wk.ket.get(op3.arg2))) {
                wk.ket.flip(op3.arg3);
             }
          }
       }
    }
    for (WeightedKet wk : extraState) {
       state.add(wk);
    }
  }

  void consolidate() {
    int i=0; 
    MyComparator cpt = new MyComparator();
    while (i<state.size()-1) {
      if (cpt.compare(state.get(i),state.get(i+1)) == 0) {
         WeightedKet keti = state.get(i);
         WeightedKet ketiPlus1 = state.get(i+1);
         keti.merge(ketiPlus1);
         state.remove(i+1);  
      }
      else {
         i=i+1;
      }
    }
  }

  void measure() {
    java.util.Random rand = new java.util.Random();
    double prob = rand.nextDouble();
    double sum = 0.0;
    WeightedKet result = new WeightedKet(this.p.getSize());
    boolean found = false;
    for (WeightedKet wk: state) {
      sum = sum + (wk.amplitude.abs())*(wk.amplitude.abs());
      if (!found && (sum > prob)) {
         found = true;
         result = wk;
      }
    }
    System.out.println(
       "Measure: " + result.bitsToString(p.getMeasureLo(), p.getMeasureHi()));
  }

  public void printState() {
    double epsilon = 0.000001;
    String result = "";
    for (WeightedKet wk: state) {
      if (wk.amplitude.abs() > epsilon) {
         result = result + wk.toString() + " + ";
      }
    }
    result = result.substring(0,result.length()-2);
    System.out.print(result);
  }
}
