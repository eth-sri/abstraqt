use std::env;

use quizx::circuit::*;
use quizx::graph::*;
use quizx::vec_graph::Graph;
use quizx::decompose::Decomposer;


fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Please provide an argument");
        return;
    }

    let file_path = &args[1];

    simulate(file_path);
}


pub fn simulate(file_path: &str) {
    let c = Circuit::from_file(file_path).unwrap();
    let mut g: Graph = c.to_graph();
    
    let i = c.num_qubits() - 1;

    g.plug_inputs(&vec![BasisElem::Z0; c.num_qubits()]);
    g.plug_output(i, BasisElem::Z1);
    g.plug(&g.to_adjoint());
    let mut d = Decomposer::new(&g);
    d.with_full_simp();
    d.decomp_all();
    let amplitude = d.scalar.float_value();
    println!("<0..0|C|?..?1> = {}", amplitude);

    if amplitude.norm() < 0.0001 {
        println!("Output {} is definitively 0", i);
    }
}
