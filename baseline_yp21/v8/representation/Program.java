package representation;

import java.util.ArrayList;

public class Program {
	private int size;
	private ArrayList<Operation> ops;
	private boolean assertionPresent;
	private ArrayList<Qubit> assertionv1;
	private ArrayList<Qubit> assertionv2;
	private int measureLo;
	private int measureHi;

	public Program(int size, ArrayList<Operation> ops, boolean assertionPresent, ArrayList<Qubit> assertionv1,
			ArrayList<Qubit> assertionv2, int measureLo, int measureHi) {
		this.size = size;
		this.ops = ops;
		this.assertionPresent = assertionPresent;
		this.assertionv1 = assertionv1;
		this.assertionv2 = assertionv2;
		this.measureLo = measureLo;
		this.measureHi = measureHi;
	}

	/**
	 * Number of qubits
	 */
	public int getSize() {
		return this.size;
	}

	public ArrayList<Operation> getOps() {
		return this.ops;
	}

	public boolean getAssertionPresent() {
		return this.assertionPresent;
	}

	public ArrayList<Qubit> getAssertionv1() {
		return this.assertionv1;
	}

	public ArrayList<Qubit> getAssertionv2() {
		return this.assertionv2;
	}

	public int getMeasureLo() {
		return this.measureLo;
	}

	public int getMeasureHi() {
		return this.measureHi;
	}

	public String toString() {
		String result = "";
		result = "circuit: " + this.getSize() + " qubits\n\n";
		for (Operation op : this.getOps()) {
			result = result + op.toString() + "\n";
		}
		result = result + "\n";
		if (this.assertionPresent) {
			result = result + assertionToString();
		}
		result = result + "measure " + this.getMeasureLo() + ".." + this.getMeasureHi();
		return result;
	}

	public String assertionToString() {
		String result = "";

		result = result + "assert state in span {\n  ";
		for (Qubit q : assertionv1) {
			result = result + q.toString();
		}
		result = result + ",\n  ";
		for (Qubit q : assertionv2) {
			result = result + q.toString();
		}
		result = result + "\n}\n\n";

		return result;
	}
}
