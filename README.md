# Abstraqt

## Dependencies

We provide a [docker](https://docs.docker.com/engine/install/) image providing
all dependencies for Abstraqt. To create the docker image and enter its
container, run (you may need `sudo`):

```bash
make -C docker run
```

Alternatively, to develop Abstraqt, install
[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
and the dependencies in the [Dockerfile](docker/Dockerfile).

## Installation

Inside the docker container, run:

```bash
time bash ./install-pip.sh  # install Abstraqt
rm -rf QuiZX/quizx && git clone https://github.com/Quantomatic/quizx.git QuiZX/quizx && cd QuiZX/quizx && git checkout 81e9e63 && cd - && make -C QuiZX/wrapper  # build QuiZX
cd baseline_yp21/v8; bash ./clean.sh; javac Main.java; cd -  # build YP21
```

This takes multiple hours, as it includes running extensive tests and some
pre-computation.

## Usage

```python
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus


state = AbstractStabilizerPlus.zero_state(2)

state = state.conjugate('X', 0)
state = state.conjugate('CNOT', 0, 1)
state = state.conjugate('T', 0)
state = state.compress(1)
state = state.measure('Z', 0)  # project to |0⟩
probability_range = state.get_trace()

print(probability_range)
```

## Evaluation

To reproduce the results from our evaluation of Abstraqt, simply run

```bash
time python abstraqt/experiments/run_experiments.py
python abstraqt/experiments/combine_results.py
```

This takes a few days, as it includes running multiple slow baselines.

### Memory Usage

To determine the memory usage of Abstraqt, run:

```bash
/usr/bin/time -v python abstraqt/experiments/run_experiments.py --tool abstraqt
```

### Generate Benchmarks

The benchmark circuits were generated using:

```bash
python abstraqt/experiments/generate_circuits.py
```
