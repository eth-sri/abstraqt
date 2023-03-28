//
// Generated by JTB 1.3.2
//

package visitor;
import syntaxtree.*;
import java.util.*;

/**
 * All GJ visitors with no argument must implement this interface.
 */

public interface GJNoArguVisitor<R> {

   //
   // GJ Auto class visitors with no argument
   //

   public R visit(NodeList n);
   public R visit(NodeListOptional n);
   public R visit(NodeOptional n);
   public R visit(NodeSequence n);
   public R visit(NodeToken n);

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
   public R visit(Goal n);

   /**
    * f0 -> "circuit"
    * f1 -> ":"
    * f2 -> IntegerLiteral()
    * f3 -> "qubits"
    */
   public R visit(Init n);

   /**
    * f0 -> OneQubitInstruction()
    *       | TwoQubitInstruction()
    *       | ThreeQubitInstruction()
    */
   public R visit(Instruction n);

   /**
    * f0 -> OneQubitGate()
    * f1 -> "("
    * f2 -> IntegerLiteral()
    * f3 -> ")"
    */
   public R visit(OneQubitInstruction n);

   /**
    * f0 -> TwoQubitGate()
    * f1 -> "("
    * f2 -> IntegerLiteral()
    * f3 -> ","
    * f4 -> IntegerLiteral()
    * f5 -> ")"
    */
   public R visit(TwoQubitInstruction n);

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
   public R visit(ThreeQubitInstruction n);

   /**
    * f0 -> "H"
    *       | "T"
    *       | "X"
    *       | "Z"
    *       | "D"
    *       | "S"
    */
   public R visit(OneQubitGate n);

   /**
    * f0 -> "CNOT"
    *       | "CZ"
    */
   public R visit(TwoQubitGate n);

   /**
    * f0 -> "CCNOT"
    *       | "NCCNOT"
    *       | "CNCNOT"
    *       | "NCNCNOT"
    */
   public R visit(ThreeQubitGate n);

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
   public R visit(Assertion n);

   /**
    * f0 -> Component()
    * f1 -> ( Component() )*
    */
   public R visit(AVector n);

   /**
    * f0 -> BracketedComponent()
    *       | Ket()
    */
   public R visit(Component n);

   /**
    * f0 -> "("
    * f1 -> [ Amplitude() ]
    * f2 -> Ket()
    * f3 -> "+"
    * f4 -> [ Amplitude() ]
    * f5 -> Ket()
    * f6 -> ")"
    */
   public R visit(BracketedComponent n);

   /**
    * f0 -> Real()
    *       | Complex()
    */
   public R visit(Amplitude n);

   /**
    * f0 -> "|"
    * f1 -> ( KetContent() )+
    * f2 -> ">"
    */
   public R visit(Ket n);

   /**
    * f0 -> IntegerLiteral()
    *       | "+"
    *       | "-"
    */
   public R visit(KetContent n);

   /**
    * f0 -> "("
    * f1 -> Real()
    * f2 -> "+"
    * f3 -> Real()
    * f4 -> "i"
    * f5 -> ")"
    */
   public R visit(Complex n);

   /**
    * f0 -> [ Sign() ]
    * f1 -> IntegerLiteral()
    * f2 -> "."
    * f3 -> IntegerLiteral()
    */
   public R visit(Real n);

   /**
    * f0 -> "-"
    */
   public R visit(Sign n);

   /**
    * f0 -> "measure"
    * f1 -> IntegerLiteral()
    * f2 -> ".."
    * f3 -> IntegerLiteral()
    */
   public R visit(Measure n);

   /**
    * f0 -> <INTEGER_LITERAL>
    */
   public R visit(IntegerLiteral n);

}

