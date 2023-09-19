package staticanalysis;

import representation.*;
import matrix.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.function.Predicate;

public class StaticAnalysis {
	Program p;
	int k;

	boolean debug;
	boolean simplealpha;
	boolean timed;
	boolean cacheSubsets;
	boolean cacheSupersets;
	boolean validityCheck;
	boolean showGate;

	public static boolean whyWrong;

	int extensionBase;
	boolean rank;

	// for checking validity of abstract states:
	public static double epsilonZeroMatrix = 0.0000000001; // max tolerance for a zero-matrix
	public static double epsilonProjection = 0.0001; // max tolerance for a projection
	public static double epsilonHermetian = 0.000001; // max tolerance for a Hermetian

	// for computing Gram-Schmidt:
	public static double epsilonZeroVector = 0.000001; // max tolerance for a zero-vector
	public static double epsilonGramSchmidt = 0.0000000001; // threshold for division in G.S.

	// for checking overall correctness:
	public static double epsilonAbstractState = 0.0001; // threshold for (state - spec)

	Map<String, Map<List<Integer>, Complex[][]>> expansionMatrixCache;

	long timeSpentInGamma = 0;
	long timeSpentInTransform = 0;
	long timeSpentInAlpha = 0;

	public StaticAnalysis(Program p, int k, boolean debug, boolean simplealpha, boolean timed, boolean cacheSubsets,
			boolean cacheSupersets, boolean validityCheck, boolean showGate, boolean whyWrongArg, int extensionBase,
			boolean rank) {
		this.p = p;
		this.k = k;
		this.debug = debug;
		this.simplealpha = simplealpha;
		this.timed = timed;
		this.cacheSubsets = cacheSubsets;
		this.cacheSupersets = cacheSupersets;
		this.validityCheck = validityCheck;
		this.showGate = showGate;

		whyWrong = whyWrongArg;

		this.extensionBase = extensionBase;
		this.rank = rank;

		this.expansionMatrixCache = new HashMap<String, Map<List<Integer>, Complex[][]>>();
		this.expansionMatrixCache.put("H", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("T", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("X", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("Z", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("D", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("S", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("CNOT", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("CZ", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("CCNOT", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("NCCNOT", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("CNCNOT", new HashMap<List<Integer>, Complex[][]>());
		this.expansionMatrixCache.put("NCNCNOT", new HashMap<List<Integer>, Complex[][]>());
	}

	AbstractState abstractState;
	AbstractState projectedSpec;

	public static int exp2(int e) {
		int result = 1;
		for (int i = 0; i < e; i++) {
			result = 2 * result;
		}
		return result;
	}

	public void run() {
		Set<List<Integer>> ops = new HashSet<List<Integer>>();
		for (Operation op : p.getOps()) {
			List<Integer> args = op.args();
			if (2 <= args.size()) {
				ops.add(args);
			}
		}

		Predicate<List<Integer>> pred = (List<Integer> l) -> {
			if ((4 <= this.k) && (2 <= extensionBase) && (extensionBase <= 3)) {
				int[] found = new int[5]; // we will use found[2] .. found[4]
				for (List<Integer> op : ops) {
					int count = 0;
					for (int i : op) {
						if (l.contains(i)) {
							count = count + 1;
						}
					}
					if (count == 2) {
						found[2] = found[2] + 1;
					} else if (count >= 3) {
						found[3] = found[3] + 1;
					}
				}
				if (extensionBase == 3) {
					return (2 <= found[3]);
				} else { // (extensionBase == 2)
					return ((1 <= found[3]) || (2 <= found[2]));
				}
			} else {
				return true;
			}
		};

		Domain domain;
		if ((this.k == 5) && (this.extensionBase == 3)) {
			List<List<Integer>> unionsWithFive = new ArrayList<List<Integer>>();
			for (List<Integer> l1 : ops) {
				for (List<Integer> l2 : ops) {
					Set<Integer> s = new HashSet<Integer>();
					s.addAll(l1);
					s.addAll(l2);
					if ((l1.size() == 3) && (l2.size() == 3 && s.size() == 5)) {
						List<Integer> cand = new ArrayList<Integer>();
						cand.addAll(s);
						java.util.Collections.sort(cand);
						if (!unionsWithFive.contains(cand)) {
							unionsWithFive.add(cand);
						}
					}
				}
			}
			List<List<Integer>> dom = new ArrayList<List<Integer>>();
			dom.addAll(unionsWithFive);
			dom.sort(new IndexSetComparator());
			domain = new Domain(p.getSize(), this.k, dom, cacheSubsets, cacheSupersets);
			// System.out.println("The domain: " + dom);
		} else {
			domain = new Domain(p.getSize(), this.k, pred, cacheSubsets, cacheSupersets);
		}

		System.out.println("Domain size: " + domain.size());

		InitialValueGenerator constantGen = (int i) -> {
			List<Integer> indexSet = domain.get(i);
			int size = exp2(indexSet.size());
			Complex[][] m = ComplexMatrix.newSquareMatrix(size);
			m[0][0] = new Complex(1);
			return m;
		};

		this.abstractState = new AbstractState(domain, constantGen);

		// System.out.println("Assertion v1: " + p.getAssertionv1());
		// System.out.println("Assertion v2: " + p.getAssertionv2());
		ArrayList<Qubit> q1 = reverseArrayList(p.getAssertionv1());
		ArrayList<Qubit> q2 = reverseArrayList(p.getAssertionv2());

		if ((p.getSize() != q1.size()) || (p.getSize() != q2.size())) {
			throw new RuntimeException("\nThe assertion has a wrong length");
		}

		InitialValueGenerator localProjectionGen =
				// For
				// v = v_1 \otimes v_2 \otimes ... \otimes v_n
				// w = w_1 \otimes w_2 \otimes ... \otimes w_n
				//
				// the local projection of span{ v, w } is
				// p_{1,2} = span{v_1 \otimes v_2, w_1 \otimes w_2 }
				// p_{1,3} = span{v_1 \otimes v_3, w_1 \otimes w_3 }
				// ...
				// p_{n-1,n} = span{v_{n-1} \otimes v_n, w_{n-1} \otimes w_n }

				(int i) -> {
					List<Integer> indexSet = domain.get(i);
					int size = exp2(indexSet.size());

					Complex[] col1 = q1.get(indexSet.get(0)).asVector();
					Complex[] col2 = q2.get(indexSet.get(0)).asVector();
					for (int x = 1; x < indexSet.size(); x++) {
						col1 = tensormultiplyColCol(q1.get(indexSet.get(x)), col1);
						col2 = tensormultiplyColCol(q2.get(indexSet.get(x)), col2);
					}
					/*
					 * Complex[] col1 = q1.get(indexSet.get(indexSet.size()-1)).asVector();
					 * Complex[] col2 = q2.get(indexSet.get(indexSet.size()-1)).asVector(); for (int
					 * x=indexSet.size()-2; x>=0; x--) { col1 = tensormultiplyColCol( q1.get(
					 * indexSet.get(x) ), col1 ); col2 = tensormultiplyColCol( q2.get(
					 * indexSet.get(x) ), col2 ); }
					 */

					Complex[][] colMatrix = ComplexMatrix.newSquareMatrix(size);
					colMatrix[0] = col1;
					colMatrix[1] = col2;
					for (int q = 2; q < size; q++) {
						colMatrix[q] = ComplexMatrix.myZeroVec(size);
					}
					return ComplexMatrix.supp(ComplexMatrix.myTranspose(colMatrix));
				};

		this.projectedSpec = new AbstractState(domain, localProjectionGen);

		for (Operation op : p.getOps()) {
			if (this.debug) {
				System.out.println("\n" + abstractState);
			}
			if (this.showGate) {
				System.out.println("Apply " + op + ":");
			}
			abstractStep(op);
		}

		if (this.debug) {
			System.out.println(abstractState.toString());
		}

		if (!abstractState.valid()) {
			throw new RuntimeException("\n" + "The final abstract state is invalid");
		}

		if (p.getAssertionPresent()) {
			String s = abstractState.subsetOf(projectedSpec) ? "correct" : "wrong";
			System.out.println("The assertion is " + s);
		}
		if (this.debug) {
			System.out.println("Projected assertion:\n" + projectedSpec);
		}

    // BEGIN EDIT
    // Added logic to check if the last qubit is |0>

		// check last qubit
		int i = p.getSize() - 1;
		System.out.println("Checking: " + i);

		// check for |0>
		int value=0;

		// check in Z basis
		Complex[][] projection = ComplexMatrix.newSquareMatrix(2);
		projection[value][value] = new Complex(1);

		InitialValueGenerator m = measureState(domain, projection, i);
		AbstractState projectedSpec = new AbstractState(domain, m);
		boolean subset = abstractState.subsetOf(projectedSpec);
		if (subset) {
			System.out.println("Qubit state: Qubit " + i + " has value " + value + " in basis Z");
		}

    // END EDIT

		if (this.timed) {
			System.out.println("time spent in gamma:     " + timeSpentInGamma / 1000 + " seconds");
			System.out.println("time spent in transform: " + timeSpentInTransform / 1000 + " seconds");
			System.out.println("time spent in alpha:     " + timeSpentInAlpha / 1000 + " seconds");
		}
	}

  // BEGIN EDIT
  // Added function

	private InitialValueGenerator measureState(Domain domain, Complex[][] partialProjection, int qubitIndex) {
		InitialValueGenerator ret =
				(int i) -> {
					List<Integer> indexSet = domain.get(i);

					Complex[][] projection = new Complex[1][1];
					projection[0][0] = new Complex(1);
					for (int x = 0; x < indexSet.size(); x++) {
						Complex[][] next;
						if (indexSet.get(x) == qubitIndex) {
							next = partialProjection;
						} else {
							next = ComplexMatrix.myId(2);
						}
						Complex[][] result = ComplexMatrix.newSquareMatrix(2*projection.length);
						ComplexMatrix.tensormultiply(next, projection, result);
						projection = result;
					}

					return projection;
				};
		return ret;
	}

  // END EDIT

	private Complex[] tensormultiplyColCol(Qubit v1, Complex[] v2) {
		Complex[] result = new Complex[2 * v2.length];
		for (int i = 0; i < v2.length; i++) {
			result[i] = v1.amplitude0.mul(v2[i]);
			result[i + v2.length] = v1.amplitude1.mul(v2[i]);
		}
		return result;
	}

	private void abstractStep(Operation op) {
		List<Integer> expansion = op.args();
		if (this.timed) {
			long time1 = System.currentTimeMillis();
			AbstractState ias1 = gamma(expansion, this.abstractState);
			// System.out.println("State after gamma: " + ias1);
			long time2 = System.currentTimeMillis();
			timeSpentInGamma = timeSpentInGamma + (time2 - time1);
			AbstractState ias2 = transform(op, expansion, ias1);
			// System.out.println("State after transform: " + ias2);
			long time3 = System.currentTimeMillis();
			timeSpentInTransform = timeSpentInTransform + (time3 - time2);
			this.abstractState = alpha(expansion, ias2);
			long time4 = System.currentTimeMillis();
			timeSpentInAlpha = timeSpentInAlpha + (time4 - time3);
			if (this.validityCheck) {
				System.out.println("The abstract state after gamma is " + (ias1.valid() ? "valid" : "invalid"));
				System.out.println("The abstract state after transform is " + (ias2.valid() ? "valid" : "invalid"));
				System.out.println(
						"The abstract state after alpha is " + (this.abstractState.valid() ? "valid" : "invalid"));
			}
			if (this.rank) {
				List<Integer> ranks = this.abstractState.ranks();
				for (int i : ranks) {
					System.out.print(i + " ");
				}
				System.out.println();
				System.out.println("Geometric mean: " + rounded(geometricMean(ranks)));
				System.out.println("Standard deviation: " + rounded(stDev(ranks)));
			}
		} else {
			this.abstractState = alpha(expansion, transform(op, expansion, gamma(expansion, this.abstractState)));
		}
	}

	private AbstractState gamma(List<Integer> expansion, AbstractState inState) {
		InitialValueGenerator expansionGen = (int i) -> {
			// System.out.println("Gamma: " + inState.domain.get(i));
			List<Integer> indexesOfSubsets = inState.domain.indexesOfSubsets(i, expansion);
			List<Complex[][]> matrices = new ArrayList<Complex[][]>();
			for (int j = 0; j < indexesOfSubsets.size(); j++) {
				int indexOfSubsetj = indexesOfSubsets.get(j);
				matrices.add(ComplexMatrix.expand(inState.get(indexOfSubsetj), expansionSet(
						inState.domain.get(indexOfSubsetj), Domain.union(inState.domain.get(i), expansion))));
			}
			return ComplexMatrix.intersectProjections(matrices);
		};

		return new AbstractState(inState.domain, expansionGen);
	}

	private Complex[][] expandAndCache(String gate, List<Integer> ex, Complex[][] U) {
		Complex[][] Uex;
		if (expansionMatrixCache.get(gate).containsKey(ex)) {
			Uex = expansionMatrixCache.get(gate).get(ex);
		} else {
			Uex = ComplexMatrix.expand(U, ex);
			expansionMatrixCache.get(gate).put(ex, Uex);
		}
		return Uex;
	}

	private AbstractState transform(Operation op, List<Integer> expansion, AbstractState inState) {
		InitialValueGenerator transformationGen = (int i) -> {
			Complex[][] Q = inState.get(i);
			boolean descending = descendingChecker(expansion);
			Complex[][] U = ComplexMatrix.String2matrix(op.gate, descending);
			List<Integer> ex = expansionSet(op.args(), Domain.union(inState.domain.get(i), expansion));
			Complex[][] Uex = expandAndCache(op.gate, ex, U);
			return ComplexMatrix.myMul(Uex, Q, ComplexMatrix.myConjTrans(Uex));
		};

		return new AbstractState(inState.domain, transformationGen);
	}

	private AbstractState alpha(List<Integer> expansion, AbstractState inState) {
		Domain domain = inState.domain;

		InitialValueGenerator contractionGen = (int i) -> {
			List<Integer> indexesOfSupersets;
			if (this.simplealpha) {
				indexesOfSupersets = new ArrayList<Integer>();
				indexesOfSupersets.add(i);
			} else {
				indexesOfSupersets = domain.indexesOfSupersets(i, expansion);
			}

			List<Complex[][]> matrices = new ArrayList<Complex[][]>();
			for (int j = 0; j < indexesOfSupersets.size(); j++) {
				List<Integer> base = Domain.union(domain.get(indexesOfSupersets.get(j)), expansion);
				List<Integer> cs = contractionSet(domain.get(i), base);
				matrices.add(ComplexMatrix
						.supp(ComplexMatrix.traceout(inState.get(indexesOfSupersets.get(j)), cs, base.size())));
			}
			return ComplexMatrix.intersectProjections(matrices);
		};

		return new AbstractState(domain, contractionGen);
	}

	private List<Integer> expansionSet(List<Integer> base, List<Integer> superset) {
		// assume base subseteq superset
		// assume base is ordered in ascending order.
		// assume superset is ordering in ascending order.

		List<Integer> result = new ArrayList<Integer>();
		int multiplicity = 0;
		int basei = 0;
		int supersetj = 0;
		while (basei < base.size()) {
			if (superset.get(supersetj) < base.get(basei)) {
				multiplicity = multiplicity + 1;
				supersetj = supersetj + 1;
			} else if (base.get(basei) < superset.get(supersetj)) {
				throw new RuntimeException("expansionSet: not a superset");
			} else { // base.get(basei) == superset.get(supersetj)
				result.add(multiplicity);
				multiplicity = 0;
				basei = basei + 1;
				supersetj = supersetj + 1;
			}
		}
		result.add(superset.size() - supersetj);
		// System.out.println("Expansion, base: " + base);
		// System.out.println(" superset: " + superset);
		// System.out.println(" result: " + result);
		return result;
	}

	private List<Integer> contractionSet(List<Integer> subset, List<Integer> base) {
		// assume subset subseteq base
		// assume subset is ordered in ascending order.
		// assume base is ordering in ascending order.

		List<Integer> result = new ArrayList<Integer>();
		int index = 0;
		int subseti = 0;
		int basej = 0;
		while (basej < base.size()) {
			if (subseti < subset.size()) {
				if (base.get(basej) < subset.get(subseti)) {
					result.add(index);
					basej = basej + 1;
				} else if (subset.get(subseti) < base.get(basej)) {
					subseti = subseti + 1;
				} else { // base.get(basej) == subset.get(subseti)
					subseti = subseti + 1;
					basej = basej + 1;
				}
			} else {
				result.add(index);
				basej = basej + 1;
			}
			index = index + 1;
		}
		// System.out.println("Contraction, subset: " + subset);
		// System.out.println(" base: " + base);
		// System.out.println(" result: " + result);
		return result;
	}

	/**
	 * Calculates the geometric mean
	 * 
	 * The geometric mean is the product of all values in the array to the Nth root,
	 * where N is the total number of values in the array.
	 * 
	 * reference--http://en.wikipedia.org/wiki/Geometric_mean
	 * 
	 * @param values source of data for calculation
	 * @return geometric mean
	 */
	public static double geometricMean(List<Integer> values) {
		double geometricMean;

		// convert (all the values to log form and
		// send them to geometricMeanFromLog which
		// is a more efficient way of calculating
		// the geometric mean since it uses addition of small log values opposed
		// to multiplication of large non-log values

		List<Double> logValues = new ArrayList<Double>();
		for (int v : values) {
			logValues.add(Math.log(v));
		}

		geometricMean = geometricMeanFromLog(logValues);

		return geometricMean;
	}

	/**
	 * Calculates the geometric mean of log values.
	 * 
	 * The geometric mean of logarithmic values is simply the arithmethic mean
	 * converted to non-logarithmic values (exponentiated)
	 * 
	 * 
	 * @param logValues array of values in logarithmic form
	 * @return geometric mean
	 */
	public static double geometricMeanFromLog(List<Double> logValues) {
		double logArithmeticMean = arithmeticMean(logValues);
		double geometricMean = Math.exp(logArithmeticMean);
		return geometricMean;
	}

	/**
	 * calculate the arithmetic mean
	 * 
	 * The arithmetic mean is the sum of all values in the array divided by the
	 * total number of values in the array.
	 * 
	 * @param values source of data for mean calculation
	 * @return arithmetic mean
	 */
	public static double arithmeticMean(List<Double> values) {
		double arithmeticMean;

		int size = values.size();

		double sum = summation(values);

		arithmeticMean = sum / size;

		return arithmeticMean;
	}

	/**
	 * @param values source of data for summation calculation
	 * @return the sum of all values within the array
	 */
	public static double summation(List<Double> values) {
		double sum = 0.0;

		for (double v : values) {
			sum = sum + v;
		}

		return sum;
	}

	public double rounded(double d) {
		double epsilon = 0.00001;
		double factor = 10000.0;
		double rounded_d = Math.round(factor * d) / factor;
		return rounded_d;
	}

	public static double stDev(List<Integer> numArray) {
		double sum = 0.0, standardDeviation = 0.0;
		int length = numArray.size();

		for (double num : numArray) {
			sum += num;
		}

		double mean = sum / length;

		for (double num : numArray) {
			standardDeviation += Math.pow(num - mean, 2);
		}

		return Math.sqrt(standardDeviation / length);
	}

	private boolean descendingChecker(List<Integer> expansion) {
		boolean result = true;
		for (int i = 0; i < expansion.size() - 1; i++) {
			if (expansion.get(i) <= expansion.get(i + 1)) {
				result = false;
			}
		}
		return result;
	}

	public <T> ArrayList<T> reverseArrayList(ArrayList<T> alist) {
		ArrayList<T> result = new ArrayList<T>();
		for (int i = alist.size() - 1; i >= 0; i--) {
			T temp = alist.get(i);
			result.add(temp);
		}
		return result;
	}
}

interface InitialValueGenerator {
	Complex[][] operation(int i);
}
