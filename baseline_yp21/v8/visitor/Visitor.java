//
// Generated by JTB 1.3.2
//

package visitor;
import syntaxtree.*;
import java.util.*;

/**
 * All void visitors must implement this interface.
 */

public interface Visitor {

   //
   // void Auto class visitors
   //

   public void visit(NodeList n);
   public void visit(NodeListOptional n);
   public void visit(NodeOptional n);
   public void visit(NodeSequence n);
   public void visit(NodeToken n);

   //
   // User-generated visitor methods below
   //

   /**
    * f0 -> Init()
    * f1 -> ( Instruction() )*
    * f2 -> [ Assertion() ]
    * f3 -> Measure()
    * f4 -> <EOF>
    */
   public void visit(Goal n);

   /**
    * f0 -> "circuit"
    * f1 -> ":"
    * f2 -> IntegerLiteral()
    * f3 -> "qubits"
    */
   public void visit(Init n);

   /**
    * f0 -> OneQubitInstruction()
    *       | TwoQubitInstruction()
    *       | ThreeQubitInstruction()
    */
   public void visit(Instruction n);

   /**
    * f0 -> OneQubitGate()
    * f1 -> "("
    * f2 -> IntegerLiteral()
    * f3 -> ")"
    */
   public void visit(OneQubitInstruction n);

   /**
    * f0 -> TwoQubitGate()
    * f1 -> "("
    * f2 -> IntegerLiteral()
    * f3 -> ","
    * f4 -> IntegerLiteral()
    * f5 -> ")"
    */
   public void visit(TwoQubitInstruction n);

   /**
    * f0 -> ThreeQubitGate()
    * f1 -> "("
    * f2 -> IntegerLiteral()
    * f3 -> ","
    * f4 -> IntegerLiteral()
    * f5 -> ","
    * f6 -> IntegerLiteral()
    * f7 -> ")"
    */
   public void visit(ThreeQubitInstruction n);

   /**
    * f0 -> "H"
    *       | "T"
    *       | "X"
    *       | "Z"
    *       | "D"
    *       | "S"
    */
   public void visit(OneQubitGate n);

   /**
    * f0 -> "CNOT"
    *       | "CZ"
    */
   public void visit(TwoQubitGate n);

   /**
    * f0 -> "CCNOT"
    *       | "NCCNOT"
    *       | "CNCNOT"
    *       | "NCNCNOT"
    */
   public void visit(ThreeQubitGate n);

   /**
    * f0 -> "assert"
    * f1 -> "state"
    * f2 -> "in"
    * f3 -> "span"
    * f4 -> "{"
    * f5 -> AVector()
    * f6 -> ","
    * f7 -> AVector()
    * f8 -> "}"
    */
   public void visit(Assertion n);

   /**
    * f0 -> Component()
    * f1 -> ( Component() )*
    */
   public void visit(AVector n);

   /**
    * f0 -> BracketedComponent()
    *       | Ket()
    */
   public void visit(Component n);

   /**
    * f0 -> "("
    * f1 -> [ Amplitude() ]
    * f2 -> Ket()
    * f3 -> "+"
    * f4 -> [ Amplitude() ]
    * f5 -> Ket()
    * f6 -> ")"
    */
   public void visit(BracketedComponent n);

   /**
    * f0 -> Real()
    *       | Complex()
    */
   public void visit(Amplitude n);

   /**
    * f0 -> "|"
    * f1 -> ( KetContent() )+
    * f2 -> ">"
    */
   public void visit(Ket n);

   /**
    * f0 -> IntegerLiteral()
    *       | "+"
    *       | "-"
    */
   public void visit(KetContent n);

   /**
    * f0 -> "("
    * f1 -> Real()
    * f2 -> "+"
    * f3 -> Real()
    * f4 -> "i"
    * f5 -> ")"
    */
   public void visit(Complex n);

   /**
    * f0 -> [ Sign() ]
    * f1 -> IntegerLiteral()
    * f2 -> "."
    * f3 -> IntegerLiteral()
    */
   public void visit(Real n);

   /**
    * f0 -> "-"
    */
   public void visit(Sign n);

   /**
    * f0 -> "measure"
    * f1 -> IntegerLiteral()
    * f2 -> ".."
    * f3 -> IntegerLiteral()
    */
   public void visit(Measure n);

   /**
    * f0 -> <INTEGER_LITERAL>
    */
   public void visit(IntegerLiteral n);

}

