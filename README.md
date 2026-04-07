# Amorphous Structure Generation via TDA-Optimized Perlin Noise by Temuujin Bayaraa

This project provides a novel method for generating realistic amorphous (glassy) atomic structures by perturbing crystalline precursors using Perlin and Simplex noise.

The key innovation is the use of Topological Data Analysis (TDA) to guide the displacement. Instead of relying on computationally expensive Molecular Dynamics (MD) or Monte Carlo (MC) simulations, this tool uses gradient descent to minimize the Wasserstein distance between the persistence diagrams of a generated structure and a reference amorphous structure.

Key Features

- Noise-Driven Displacement: Uses 3D Perlin and Simplex noise to create smooth, correlated atomic perturbations that preserve local density better than random displacement.
- TDA-Guided Optimization: Leverages the homcloud library to compute 1D and 2D persistence diagrams, ensuring the generated topology matches a target amorphous state.
- Gradient Descent with Momentum: Optimizes noise parameters (scale, amplitude, persistence, lacunarity) to iteratively refine the structure.
- High Performance: Supports parallel execution via ProcessPoolExecutor for generating large ensembles of structures across multiple CPU cores.
- VASP Integration: Reads and writes standard POSCAR files for easy integration into existing DFT or atomistic simulation workflows.

How It Works

1. Input: A crystalline POSCAR (starting point) and an amorphous POSCAR (topological reference).
2. Perturbation: Atoms are displaced according to a 3D noise field defined by a set of stochastic parameters.
3. Validation: A KD-Tree based check ensures no two atoms violate a user-defined minimum interatomic distance.
4. TDA Scoring: The persistence diagram of the new structure is calculated and compared to the reference using the Wasserstein metric.
5. Iteration: The optimizer adjusts the noise parameters until the topological "fingerprint" of the generated structure matches the reference within a specified threshold.

Dependencies

- homcloud: For Topological Data Analysis and persistence diagrams.
- noise: For Perlin and Simplex noise generation.
- numpy & scipy: For coordinate manipulation and KD-Tree distance calculations.
- concurrent.futures: For multi-core parallelization.

---
Example Usage

python get_amor_structures.py \
      --crystal POSCAR_crystal \
      --reference POSCAR_amorphous \
      --output_dir results \
      --num_structures 10 \
      --min_distance 2.0 \
      --num_workers 40
