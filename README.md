# PINN Digital Twin Framework

## Project status

This repository is an early-stage research prototype containing a small set of numerical and neural-network components for physics-informed surrogate modelling. It is not yet a complete, independently reproducible digital-twin or CFD benchmark implementation.

## Current implementation

- `code/teacher_model/pinn_architecture.py`: TensorFlow teacher/student network skeletons. A complete PDE residual/loss definition, training pipeline, checkpointed model, and dataset are not currently included.
- `code/geometry_encoding/sdf_computation.py`: approximate signed distance field for an ordered 2D polygon boundary. Distance magnitudes are based on sampled boundary points.
- `code/evaluation/metrics.py`: relative L2 error, mean absolute error, and root mean square error.
- `code/benchmarks/airfoil_cascade/run_benchmark.py`: illustrative placeholder output. It does not run a CFD solver or measure model inference.

**Important reproducibility note:** performance figures previously shown in this README (including sub-15 ms latency, 4,000× speedup, and less than 2.5% error) are not reproduced by the code or data currently included in this repository. They should not be treated as independently validated results until the trained model, reference dataset, benchmark procedure, and raw measurements are added.

## Environment

The CI baseline uses Python 3.12. Create a virtual environment and install the declared dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the lightweight numerical tests with:

```bash
python -m pip install numpy scipy pytest
pytest -q
```

The CI pipeline also performs a dependency-resolution dry run against the full requirements set and compiles all Python source files. This does not replace GPU training, convergence studies, numerical validation against a CFD reference, or reproducibility experiments.

## Contact

Wudneh Tilahun Mengist
