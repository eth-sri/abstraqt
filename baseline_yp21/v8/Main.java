import syntaxtree.*;
import visitor.*;
import representation.*;
import interpreter.*;
import staticanalysis.*;
import java.util.Arrays;
import java.util.List;

public class Main {
   public static void main(String [] args) {
      long time1 = System.currentTimeMillis();
      try {
         // new QuantumParser(System.in).Goal();
         Node root = new QuantumParser(System.in).Goal();
         // System.out.println("Program parsed successfully");
         ProgramConstructor pc = new ProgramConstructor();
         root.accept(pc);
         Program p = pc.getProgram();

         List<String> argList = Arrays.asList(args);

         if (argList.contains("input")) {
           System.out.println(p.toString());
         } 
         else 
         if (argList.contains("static")) {
            int k = Integer.parseInt( 
               argList.get(argList.indexOf("static") + 1));
            boolean debug = false;
            boolean simplealpha = false;
            boolean timed = false;
            boolean cacheSubsets = false;
            boolean cacheSupersets = false;
            boolean validityCheck = false;
            boolean showGate = false;
            boolean whyWrong = false;
            int extensionBase = 0;
            boolean rank = false;

            if (argList.contains("debug")) {
               debug = true;
            }
            if (argList.contains("simplealpha")) {
               simplealpha = true;
            }
            if (argList.contains("timed")) {
               timed = true;
            }
            if (argList.contains("cacheSubsets")) {
               cacheSubsets = true;
            }
            if (argList.contains("cacheSupersets")) {
               cacheSubsets = true;
            }
            if (argList.contains("valid")) {
               validityCheck = true;
            }
            if (argList.contains("gate")) {
               showGate = true;
            }
            if (argList.contains("whywrong")) {
               whyWrong = true;
            }
            if (argList.contains("ex")) {
               extensionBase = Integer.parseInt(
                   argList.get(argList.indexOf("ex") + 1));
            }
            if (argList.contains("rank")) {
               rank = true;
            }

            if ((1<k) && (k<=p.getSize())) {
               new StaticAnalysis(p,k,debug,simplealpha,timed,
                                  cacheSubsets,cacheSupersets,
                                  validityCheck,showGate,whyWrong,
                                  extensionBase,rank)
                    .run();        
               if (timed) {
                 long time2 = System.currentTimeMillis();
                 long timeSpent = time2-time1;
                 System.out.println("Total:                   " +
                             timeSpent/1000 + " seconds");
               }
            }
            else {
               throw new RuntimeException("\nFound k = " + k + " and " +
                                          "n = " + p.getSize() + ". " +
                                          "Supports only 1<k and k<=n");
            }
         }
         else {
            new Interpreter(p).run();
         }
      }
      catch (ParseException e) {
         System.out.println(e.toString());
      }
   }

}

