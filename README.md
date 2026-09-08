# PINN Digital Twin Framework

## Fast Physics-Informed Neural Network Surrogate Models for Near-Real-Time Fluid Simulation

### Overview

This repository contains the implementation of a Physics-Informed Neural Network (PINN) framework for near-real-time fluid simulation with digital twin applications. The framework achieves sub-15ms inference latency with 4,000× speedup over conventional CFD while maintaining <2.5% error.

**Key Features:**
- 🧠 **Geometry-Agnostic Encoding:** Single trained model handles multiple geometries without retraining
- 🚀 **Real-Time Inference:** Sub-15ms latency on commodity GPU hardware
- 📊 **4,000× Speedup:** Compared to high-fidelity DNS references
- 🔬 **Model Distillation:** Teacher-student compression for edge deployment

### Results

| Test Case | Error (%) | Latency (ms) | Speedup |
|-----------|-----------|--------------|---------|
| Airfoil Cascade | 2.3 | 14.2 | 4,200× |
| Stenotic Vessel | 1.8 | 11.8 | 4,800× |
| Stirred-Tank Reactor | 2.5 | 12.5 | 4,100× |

### Contact

Wudneh Tilahun Mengist - wud2hagie@gmail.com