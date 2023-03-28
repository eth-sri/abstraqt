package staticanalysis;

import representation.Qubit;
import matrix.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Comparator;
import java.util.function.Predicate;

public class Domain {

	// The indexSet management may be done as follows.
	// In the initializer, which is given n,k, precompute:
	// for helping with computing alpha:
	// for each expansion set and for each index of an indexset:
	// the indexes of indexsets in an expansion that are supersets
	// for helping with computing gamma:
	// for each expansion set and each indexset in an expansion:
	// the indexes of indexsets that are subsets of the expansion

	int n; // n is the number of qubits in the system
	int k; // k is the number of qubits in each local projection
	Predicate<List<Integer>> pred;
	List<List<Integer>> domain = new ArrayList<List<Integer>>();

	boolean cacheSubsets;
	boolean cacheSupersets;

	Map<Integer, Map<List<Integer>, List<Integer>>> indexesOfSubsetsCache;
	Map<Integer, Map<List<Integer>, List<Integer>>> indexesOfSupersetsCache;

	private void addDomain(ArrayList<Integer> indexSet, int low, int high, int count, Predicate<List<Integer>> pred) {
		if (count == 0) {
			if (pred.test(indexSet)) {
				this.domain.add(indexSet);
			}
		} else {
			for (int index = low; index < high; index++) {
				ArrayList<Integer> clonedIndexSet = (ArrayList<Integer>) indexSet.clone();
				clonedIndexSet.add(index);
				addDomain(clonedIndexSet, index + 1, high, count - 1, pred);
			}
		}
	}

	private void initCaching() {
		if (cacheSubsets) {
			indexesOfSubsetsCache = new HashMap<Integer, Map<List<Integer>, List<Integer>>>();
			for (int i = 0; i < this.domain.size(); i++) {
				indexesOfSubsetsCache.put(i, new HashMap<List<Integer>, List<Integer>>());
			}
		}

		if (cacheSupersets) {
			indexesOfSupersetsCache = new HashMap<Integer, Map<List<Integer>, List<Integer>>>();
			for (int i = 0; i < this.domain.size(); i++) {
				indexesOfSupersetsCache.put(i, new HashMap<List<Integer>, List<Integer>>());
			}
		}
	}

	public Domain(int n, int k, Predicate<List<Integer>> pred, boolean cacheSubsets, boolean cacheSupersets) {
		this.n = n;
		this.k = k;
		this.pred = pred;
		this.cacheSubsets = cacheSubsets;
		this.cacheSupersets = cacheSupersets;
		addDomain(new ArrayList<Integer>(), 0, n, k, pred);

		initCaching();
	}

	public Domain(int n, int k, List<List<Integer>> domain, boolean cacheSubsets, boolean cacheSupersets) {
		this.n = n;
		this.k = k;
		this.pred = pred;
		this.cacheSubsets = cacheSubsets;
		this.cacheSupersets = cacheSupersets;
		this.domain = domain;

		initCaching();
	}

	public List<Integer> get(int i) {
		return domain.get(i);
	}

	public int size() {
		return domain.size();
	}

	private int indexOf(List<Integer> key) {
		// assume key in this.domain
		// indexOf uses binary search because linear search is too slow
		int first = 0;
		int last = this.domain.size() - 1;
		int mid = (first + last) / 2;
		while (first <= last) {
			int compareResult = new IndexSetComparator().compare(this.domain.get(mid), key);
			if (compareResult == -1) {
				first = mid + 1;
			} else if (compareResult == 0) {
				return mid;
			} else if (compareResult == 1) {
				last = mid - 1;
			}
			mid = (first + last) / 2;
		}
		return -1;
	}

	private void myAdd(List<Integer> result, List<Integer> candidate) {
		int i = indexOf(candidate);
		if (0 <= i) {
			result.add(i);
		}
	}

	public List<Integer> indexesOfSubsets(int i, List<Integer> expansion) {
		if (cacheSubsets && indexesOfSubsetsCache.get(i).containsKey(expansion)) {
			return indexesOfSubsetsCache.get(i).get(expansion);
		} else {
			List<Integer> result = new ArrayList<Integer>();
			List<Integer> current = this.domain.get(i);

			if (expansion.size() == 1) {
				int x = expansion.get(0);
				if (current.contains(x)) {
					result.add(i);
				} else {
					List<Integer> superset = addSingle(current, x);
					for (int j = 0; j < superset.size(); j++) {
						int elem = superset.get(j);
						List<Integer> candidate = removeSingle(superset, elem);
						myAdd(result, candidate);
					}
				}
			} else if (expansion.size() == 2) {
				int x0 = expansion.get(0);
				int x1 = expansion.get(1);
				boolean b0 = current.contains(x0);
				boolean b1 = current.contains(x1);

				if (b0 && b1) {
					result.add(i);
				} else if (b0 && (!b1)) {
					List<Integer> superset = addSingle(current, x1);
					for (int j = 0; j < superset.size(); j++) {
						int elem = superset.get(j);
						List<Integer> candidate = removeSingle(superset, elem);
						myAdd(result, candidate);
					}
				} else if ((!b0) && b1) {
					List<Integer> superset = addSingle(current, x0);
					for (int j = 0; j < superset.size(); j++) {
						int elem = superset.get(j);
						List<Integer> candidate = removeSingle(superset, elem);
						myAdd(result, candidate);
					}
				} else {
					List<Integer> superset = addSingle(addSingle(current, x0), x1);
					for (int j0 = 0; j0 < superset.size(); j0++) {
						int elem0 = superset.get(j0);
						for (int j1 = j0 + 1; j1 < superset.size(); j1++) {
							int elem1 = superset.get(j1);
							List<Integer> candidate = removeSingle(removeSingle(superset, elem0), elem1);
							myAdd(result, candidate);
						}
					}
				}
			} else if (expansion.size() == 3) {
				List<Integer> superset = union(this.domain.get(i), expansion);
				for (int j = 0; j < this.domain.size(); j++) {
					if (new IndexSetSubsetChecker().compare(this.domain.get(j), superset)) {
						result.add(j);
					}
				}
			} else {
				throw new RuntimeException("Supports only 1,2,3-qubit gates");
			}

			if (cacheSubsets) {
				indexesOfSubsetsCache.get(i).put(expansion, result);
			}
			return result;
		}
	}

	public List<Integer> indexesOfSupersets(int i, List<Integer> expansion) {
		if (cacheSupersets && indexesOfSupersetsCache.get(i).containsKey(expansion)) {
			return indexesOfSupersetsCache.get(i).get(expansion);
		} else {
			List<Integer> result = new ArrayList<Integer>();
			List<Integer> subset = this.domain.get(i);

			if (expansion.size() == 1) {
				int x = expansion.get(0);
				if (subset.contains(x)) {
					List<Integer> subsubset = removeSingle(subset, x);
					for (int j = 0; j < this.n; j++) {
						if (!subsubset.contains(j)) {
							List<Integer> candidate = addSingle(subsubset, j);
							myAdd(result, candidate);
						}
					}
				} else {
					result.add(i);
				}
			} else if (expansion.size() == 2) {
				int x0 = expansion.get(0);
				int x1 = expansion.get(1);
				boolean b0 = subset.contains(x0);
				boolean b1 = subset.contains(x1);

				if (b0 && b1) {
					List<Integer> subsubset = removeSingle(removeSingle(subset, x0), x1);
					for (int j0 = 0; j0 < this.n; j0++) {
						if (!subsubset.contains(j0)) {
							for (int j1 = j0 + 1; j1 < this.n; j1++) {
								if ((!subsubset.contains(j1))) {
									List<Integer> candidate = addSingle(addSingle(subsubset, j0), j1);
									myAdd(result, candidate);
								}
							}
						}
					}
				} else if (b0 && (!b1)) {
					List<Integer> subsubset = removeSingle(subset, x0);
					for (int j = 0; j < this.n; j++) {
						if (!subsubset.contains(j)) {
							List<Integer> candidate = addSingle(subsubset, j);
							myAdd(result, candidate);
						}
					}
				} else if ((!b0) && b1) {
					List<Integer> subsubset = removeSingle(subset, x1);
					for (int j = 0; j < this.n; j++) {
						if (!subsubset.contains(j)) {
							List<Integer> candidate = addSingle(subsubset, j);
							myAdd(result, candidate);
						}
					}
				} else {
					result.add(i);
				}
			} else if (expansion.size() == 3) {
				for (int j = 0; j < this.domain.size(); j++) {
					if (new IndexSetSubsetChecker().compare(subset, union(this.domain.get(j), expansion))) {
						result.add(j);
					}
				}
			} else {
				throw new RuntimeException("Supports only 1,2,3-qubit gates");
			}

			if (cacheSupersets) {
				indexesOfSupersetsCache.get(i).put(expansion, result);
			}
			return result;
		}
	}

	public static List<Integer> union(List<Integer> l1, List<Integer> l2) {
		List<Integer> result = new ArrayList<Integer>();

		int i1 = 0;
		int i2 = 0;
		while ((i1 < l1.size()) && (i2 < l2.size())) {
			if (l1.get(i1) < l2.get(i2)) {
				result.add(l1.get(i1));
				i1 = i1 + 1;
			} else if (l2.get(i2) < l1.get(i1)) {
				result.add(l2.get(i2));
				i2 = i2 + 1;
			} else { // l1.get(i1) == l2.get(i2)
				result.add(l1.get(i1));
				i1 = i1 + 1;
				i2 = i2 + 1;
			}
		}
		if (i1 < l1.size()) {
			while (i1 < l1.size()) {
				result.add(l1.get(i1));
				i1 = i1 + 1;
			}
		} else { // (i2<l2.size())
			while (i2 < l2.size()) {
				result.add(l2.get(i2));
				i2 = i2 + 1;
			}
		}

		return result;
	}

	private static List<Integer> addSingle(List<Integer> l, int i) {
		List<Integer> singleton = new ArrayList<Integer>();
		singleton.add(i);
		return union(l, singleton);
	}

	private static List<Integer> setDifference(List<Integer> l1, List<Integer> l2) {
		List<Integer> result = new ArrayList<Integer>();

		int i1 = 0;
		int i2 = 0;
		while ((i1 < l1.size()) && (i2 < l2.size())) {
			if (l1.get(i1) < l2.get(i2)) {
				result.add(l1.get(i1));
				i1 = i1 + 1;
			} else if (l2.get(i2) < l1.get(i1)) {
				i2 = i2 + 1;
			} else { // l1.get(i1) == l2.get(i2)
				i1 = i1 + 1;
				i2 = i2 + 1;
			}
		}
		if (i1 < l1.size()) {
			while (i1 < l1.size()) {
				result.add(l1.get(i1));
				i1 = i1 + 1;
			}
		} else { // (i2<l2.size())
			while (i2 < l2.size()) {
				i2 = i2 + 1;
			}
		}

		return result;
	}

	private static List<Integer> removeSingle(List<Integer> l, int i) {
		List<Integer> singleton = new ArrayList<Integer>();
		singleton.add(i);
		return setDifference(l, singleton);
	}

	public String toString() {
		String result = "";
		for (List<Integer> l : domain) {
			result = result + l.toString() + " ";
		}
		return result;
	}
}

class IndexSetComparator implements Comparator<List<Integer>> {
	public int compare(List<Integer> o1, List<Integer> o2) {
		// assume that o1.size() == o2.size()
		for (int i = 0; i < o1.size(); i++) {
			int index1 = o1.get(i);
			int index2 = o2.get(i);
			if (index1 < index2) {
				return -1;
			} else if (index1 > index2) {
				return 1;
			}
		}
		return 0;
	}
}

class IndexSetSubsetChecker {
	public boolean compare(List<Integer> o1, List<Integer> o2) {
		int i1 = 0;
		int i2 = 0;
		boolean o1SubsetOfo2 = true;
		boolean o2SubsetOfo1 = true;

		while ((i1 < o1.size()) && (i2 < o2.size())) {
			if (o1.get(i1) < o2.get(i2)) {
				o1SubsetOfo2 = false;
				i1 = i1 + 1;
			} else if (o2.get(i2) < o1.get(i1)) {
				o2SubsetOfo1 = false;
				i2 = i2 + 1;
			} else { // o1.get(i1) == o2.get(i2)
				i1 = i1 + 1;
				i2 = i2 + 1;
			}
		}
		if (i1 < o1.size()) {
			o1SubsetOfo2 = false;
		} else if (i2 < o2.size()) {
			o2SubsetOfo1 = false;
		}

		return o1SubsetOfo2;
	}
}
