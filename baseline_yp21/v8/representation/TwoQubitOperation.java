package representation;

import java.util.List;
import java.util.ArrayList;

public class TwoQubitOperation extends Operation {
  public int arg1;
  public int arg2;
  public TwoQubitOperation(String gate, int arg1, int arg2) {
       this.gate = gate;
       this.arg1 = arg1;
       this.arg2 = arg2;
  }

  public List<Integer> args() {
    List<Integer> result = new ArrayList<Integer>();
    result.add(this.arg1);
    result.add(this.arg2);
    return result;
  }

  public String toString() {
    return this.gate + "(" + this.arg1 + "," + this.arg2 + ")";
  }
}

