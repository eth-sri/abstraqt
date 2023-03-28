import java.util.Arrays;
import java.util.List;

class Gen {
  public static void main(String [] args) {
    List<String> argList = Arrays.asList(args);

    if (argList.contains("grover")) {
      int generateParam = Integer.parseInt(
            argList.get(argList.indexOf("grover") + 1));
      String initParam = argList.get(argList.indexOf("grover") + 2);
      String shape = "tree";
      if (4 <= argList.size()) {
         shape = argList.get(argList.indexOf("grover") + 3);
      }
      System.out.println( new GroverGen(generateParam,initParam,shape).run() );
    }
  }
}
