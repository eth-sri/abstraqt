Getting Started Guide 
------- Time needed: one minute.
1. check that you have java version 1.8 or higher: 
      java -version
2. unpack our tarfile.gz: 
      gunzip tarfile.gz; tar xvf tarfile
3. compile: 
      bash clean.sh; javac Main.java
4. run on the running example in the paper (Figure 1 on p.2):
      java Main static 2 < benchmarks/ghz003.q
   and get this output:
      Domain size: 3
      The assertion is correct
 


Step-by-Step Instructions for Evaluating the Software 
------- Time needed: Steps 1-2: three minutes; Step 3: one week.
1. Reproduce some of the measurements (Figure 4 on p.10):
      bash run.sh
   The expected output: output-static-analysis/run-all.txt 
   Notice that for Grover, the run was for n=15, which isn't it Figure 4.
   The output "The assertion is correct" is represented by a checkmark 
   in Figure 4.  
2. Reproduce the abstract states (Figure 2 on p.9):
      java Main static 2 gate debug < benchmarks/ghz003.q
   The expected output: output-static-analysis/ghz003-debug.txt
   The output is in the form of matrices rather than sums of outer products.
   If you calculate the sums of outer products, you will see that they match 
   the matrices in the output.
3. Reproduce all the measurements (Figure 4 on p.10) [time consuming!];
   in run.sh, uncomment all the lines (or as many lines as you like):
      bash run.sh
   The output will look as before, just more cases.
   Notice that for grover and each value of n, run.sh has two runs,
   as mentioned in the paper l.1168-1182.



Guide to Matching the Software with the Paper
------- Time needed: five minutes.
1. In the paper, Section 6, l.745, we have the definition of 
   an abstract step, U^#_F = alpha_{T->S} o U^{cg}_F o gamma_{S->T}
   In the file staticanalysis/StaticAnalysis.java, 
   in the method abstractStep, l.329-331, we have this code
          this.abstractState = alpha(expansion,
                               transform(op,expansion,
                               gamma(expansion,this.abstractState)));
   which matches the paper's definition.  Notice that the above code
   is in the else-branch of an if-then-else; the then-branch executes 
   the same behavior, but along with additional output.
2. For each run, we get output that says how much time was spent in
   gamma, transform, and alpha.  Figure 5 in the paper shows piecharts
   for programs with 300 qubits.  Even for smaller numbers of qubits,
   the runs will show that the time spent in gamma, transform, and alpha
   is much like what Figure 5 shows.
3. The software includes an interpreter that can produce the run 
   in Figure 1:
       java Main < benchmarks/ghz003.q
   Run a few times to see that the last line sometimes is
       Measure: 000  
   and sometimes is 
       Measure: 111
4. The claims in Section 9 of the paper (l.1073-1078):
       1. Is our approach scalable? Yes, it scales to programs with
          300 qubits.
       2. Is our approach useful? Yes, it checks the assertions in
          three families of benchmark programs.
       3. Is our approach flexible? Yes, it enables users to change
          the abstract domain easily.
   The file run.sh mentions programs with 300 qubits and the runs 
   with run.sh verify the assertions.  As an example of how we can change  
   the abstract domain, notice how some runs use k=2 and some use k=5.
   Notice how outputs such as 
       Domain size: 3
   are different for different runs, which illustrates how 
   the abstract domain varies.
