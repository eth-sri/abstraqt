package interpreter;

import java.util.BitSet;
import java.util.Comparator;

public class MyComparator implements Comparator<WeightedKet> {
  public int compare(WeightedKet o1, WeightedKet o2) {
    if (o1.size == o2.size) { 
       return ket2Int(o1.ket,o1.size) - ket2Int(o2.ket,o2.size);
    }
    else {
       throw new NullPointerException("Kets are of difference size"); 
    }
  }

  public int ket2Int(BitSet ket, int size) {
    // System.out.println(ket.toString());
    int result = 0; 
    int factor = 1;
    for (int i=0; i<size; i++) {
      result = result + factor * (ket.get(i) ? 1 : 0);
      factor = 2 * factor;
    }
    // System.out.println("ket has integer value: " + result);
    return result;
  }
}
