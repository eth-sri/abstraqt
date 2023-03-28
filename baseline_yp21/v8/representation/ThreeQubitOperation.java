package representation;

import java.util.List;
import java.util.ArrayList;

public class ThreeQubitOperation extends Operation {
  public int arg1;
  public int arg2;
  public int arg3;
  public ThreeQubitOperation(String gate, int arg1, int arg2, int arg3) {
       this.gate = gate;
       this.arg1 = arg1;
       this.arg2 = arg2;
       this.arg3 = arg3;
  }

  public List<Integer> args() {
    List<Integer> result = new ArrayList<Integer>();
    result.add(this.arg1);
    result.add(this.arg2);
    result.add(this.arg3);
    return result;
  }

  public String toString() {
    return this.gate + "(" + this.arg1 + "," + 
                             this.arg2 + "," + 
                             this.arg3 + ")";
  }
}

