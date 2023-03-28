package representation;

import java.util.List;
import java.util.ArrayList;

public class OneQubitOperation extends Operation {
  public int arg;
  public OneQubitOperation(String gate, int arg) {
      this.gate = gate;
       this.arg = arg;
  }

  public List<Integer> args() {
    List<Integer> result = new ArrayList<Integer>();
    result.add(this.arg);
    return result;
  }

  public String toString() {
    return this.gate + "(" + this.arg + ")";
  }
}

