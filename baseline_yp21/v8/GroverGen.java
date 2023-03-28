class GroverGen {
  int generateParam; 
  // Grover search in a space addressed with 2^generateParam bits
  String initParam;
  String shape;

  public GroverGen(int generateParam, String initParam, String shape) {
    this.generateParam = generateParam;
        this.initParam = initParam;
            this.shape = shape;
  }

  public String run() {
    int bits;
    int helperBits;
    if (this.shape .equals ("tree")) {
      bits = staticanalysis.StaticAnalysis.exp2(generateParam);
      helperBits = bits-1;
    } 
    else { // the shape is linear
      bits = generateParam;
      helperBits = bits;
    }

    String result = "";
    result = result + preamble(bits,helperBits);
    result = result + "// Init\n";
    if (initParam.equals("plus")) {
       result = result + hadamardSeq(bits);
    }
    else {
       result = result + "// Nothing!\n\n";
    }
    result = result + "// Grover iteration 1\n";
    result = result + groverZ0(bits,helperBits)
                    + hadamardSeq(bits)
                    + groverZ0(bits,helperBits)
                    + hadamardSeq(bits) 
                    + postfix(bits,helperBits);
    return result;
  }

  public String preamble(int bits, int helperBits) {
    String result = "";
    result = result + "// Grover search on " + bits + " qubits and " +
                      helperBits + " helper qubits\n\n" 
                    + "circuit: " + (bits + helperBits) + " qubits\n\n";
    return result;
  }

  public String hadamardSeq(int bits) {
    String result = "";
    for (int i=0; i<bits; i++) {
      result = result + "H(" + i + ")\n";
    }
    result = result + "\n";
    return result;
  }

  public String groverZ0(int bits, int helperBits) {
    if (this.shape . equals ("tree")) {
      String result = "";
      result = result + "// Z_0\n";
      for (int i=0; i<bits-1; i++) {
        if (i<bits/2) {
           result = result + "NCNCNOT(" + (2*i) + "," 
                                        + (2*i+1) + ","
                                        + (bits+i) + ")\n";
        } 
        else {
           result = result +   "CCNOT(" + (2*i) + "," 
                                        + (2*i+1) + ","
                                        + (bits+i) + ")\n";
        }
      }
      result = result + "// The key step!\n" 
                      + "Z(" + (bits+helperBits-1) + ")\n";
      for (int i=bits-2; i>=0; i--) {
        if (i<bits/2) {
           result = result + "NCNCNOT(" + (2*i) + "," 
                                        + (2*i+1) + ","
                                        + (bits+i) + ")\n";
        } 
        else {
           result = result +   "CCNOT(" + (2*i) + "," 
                                        + (2*i+1) + ","
                                        + (bits+i) + ")\n";
        }
      }
      result = result + "\n";
      return result;
    }
    else { // the shape is "linear"
      String result = "";
      result = result + "NCNCNOT(" + "0" + "," + "1" + "," + bits + ")\n";
      for (int i=2; i<bits; i++) {
          result = result + "NCCNOT(" + i + "," 
                                      + (bits+i-2) + "," 
                                      + (bits+i-1) + ")\n";
      }
      result = result + "CNOT(" + (bits+helperBits-2) + "," 
                                + (bits+helperBits-1) + ")\n";
      result = result + "// The key step!\n"
                      + "Z(" + (bits+helperBits-1) + ")\n";
      result = result + "CNOT(" + (bits+helperBits-2) + "," 
                                + (bits+helperBits-1) + ")\n";
      for (int i=bits-1; i>=2; i--) {
          result = result + "NCCNOT(" + i + "," 
                                      + (bits+i-2) + "," 
                                      + (bits+i-1) + ")\n";
      }
      result = result + "NCNCNOT(" + "0" + "," + "1" + "," + bits + ")\n";
      result = result + "\n";
      return result;
    }   
  }

  public String postfix(int bits, int helperBits) {
    String result = "";
    result = result + "assert state in span {\n" 
                    + "  " + "|" + zeros(helperBits+bits) + "> ,\n" 
                    + "  " + "|" + zeros(helperBits) + plusses(bits) + ">\n" 
                    + "}\n\n" 
                    + "measure 0.." + (bits + helperBits - 1);
          
    return result;
  }

  public String zeros(int bits) {
    String result = "";
    for (int i=0; i<bits; i++) {
      result = result + "0";
    }
    return result;
  }

  public String plusses(int bits) {
    String result = "";
    for (int i=0; i<bits; i++) {
      result = result + "+";
    }
    return result;
  }
}
