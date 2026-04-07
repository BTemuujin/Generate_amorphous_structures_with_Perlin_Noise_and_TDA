import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import homcloud.interface as hc
import random
import math
import noise
import traceback
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.spatial import cKDTree
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Type aliases
Structure = Dict[str, Any]
PersistenceDiagram = Any
PerlinParams = Dict[str, float]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate amorphous structures using Perlin noise')
    parser.add_argument('--crystal', type=str, required=True, help='Path to crystal POSCAR file')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference amorphous POSCAR file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_structures', type=int, default=10, help='Number of structures to generate')
    parser.add_argument('--threshold', type=float, default=0.1, help='Wasserstein distance threshold')
    parser.add_argument('--min_distance', type=float, default=0.0, help='Minimum interatomic distance')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--delta', type=float, default=0.1, help='Delta parameter for Wasserstein distance')
    parser.add_argument('--p', type=float, default=2.0, help='p parameter for Wasserstein distance')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum iterations for gradient descent')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    return parser.parse_args()

def set_random_seed(seed: Optional[int]) -> None:
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # Note: noise library doesn't support setting global seed
        logger.info(f"Random seed set to {seed}")

def read_poscar(filename: str) -> Structure:
    """Read POSCAR file and return structure information."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Read scaling factor
    scale = float(lines[1].strip())
    
    # Read lattice vectors
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = np.array([float(x) for x in lines[i+2].split()])
    lattice *= scale
    
    # Read atom types and counts
    atom_types = lines[5].split()
    atom_counts = [int(x) for x in lines[6].split()]
    
    # Check if positions are direct or cartesian
    coord_type = lines[7].strip()[0].lower()
    
    # Read atom positions
    positions = []
    symbols = []
    line_index = 8
    
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            pos = np.array([float(x) for x in lines[line_index].split()[:3]])
            positions.append(pos)
            symbols.append(atom_type)
            line_index += 1
    
    positions = np.array(positions)
    
    # Convert to direct coordinates if needed
    if coord_type == 'c' or coord_type == 'k':
        # Convert Cartesian to direct
        inv_lattice = np.linalg.inv(lattice)
        positions = np.dot(positions, inv_lattice)
    
    return {
        'lattice': lattice,
        'positions': positions,
        'symbols': symbols
    }

def write_poscar(filename: str, structure: Structure) -> None:
    """Write structure to POSCAR file."""
    lattice = structure['lattice']
    positions = structure['positions']
    symbols = structure['symbols']
    
    # Count atoms of each type
    unique_symbols = []
    atom_counts = []
    
    for s in symbols:
        if s not in unique_symbols:
            unique_symbols.append(s)
            atom_counts.append(1)
        else:
            atom_counts[unique_symbols.index(s)] += 1
    
    with open(filename, 'w') as f:
        # Write header
        f.write("Generated structure\n")
        f.write("1.0\n")
        
        # Write lattice vectors
        for i in range(3):
            f.write(f"{lattice[i,0]:15.8f} {lattice[i,1]:15.8f} {lattice[i,2]:15.8f}\n")
        
        # Write atom types and counts
        f.write(' '.join(unique_symbols) + '\n')
        f.write(' '.join([str(count) for count in atom_counts]) + '\n')
        
        # Write positions (direct coordinates)
        f.write("Direct\n")
        
        # Sort atoms by type
        sorted_positions = []
        for symbol in unique_symbols:
            for i, s in enumerate(symbols):
                if s == symbol:
                    sorted_positions.append(positions[i])
        
        for pos in sorted_positions:
            f.write(f"{pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f}\n")

def compute_persistence_diagrams(structure: Structure) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Compute persistence diagrams from structure."""
    positions = structure['positions']
    lattice = structure['lattice']
    symbols = structure['symbols']
    
    # Convert to Cartesian coordinates
    cart_positions = np.dot(positions, lattice)
    
    try:
        # Compute persistence diagrams
        pd_list = hc.PDList.from_alpha_filtration(
            cart_positions,
            vertex_symbols=symbols,
            save_boundary_map=True
        )
        
        # Extract diagrams for dimensions 1 and 2
        pd1 = pd_list.dth_diagram(1)
        pd2 = pd_list.dth_diagram(2)
        
        return pd_list, pd1, pd2
    except Exception as e:
        logger.error(f"Error computing persistence diagrams: {e}")
        return None, None, None

def compute_wasserstein_distance(
    pd1_ref: PersistenceDiagram, 
    pd2_ref: PersistenceDiagram, 
    pd1_new: PersistenceDiagram, 
    pd2_new: PersistenceDiagram, 
    p: float = 2.0, 
    delta: float = 0.1
) -> Tuple[float, float, float]:
    """Compute Wasserstein distance between persistence diagrams."""
    try:
        # Compute distances for dimensions 1 and 2
        dist1 = hc.distance.wasserstein(pd1_ref, pd1_new, delta=delta, internal_p=p)
        dist2 = hc.distance.wasserstein(pd2_ref, pd2_new, delta=delta, internal_p=p)
        
        # Compute total distance
        total_dist = dist1 + dist2
        
        return total_dist, dist1, dist2
    except Exception as e:
        logger.error(f"Error computing Wasserstein distance: {e}")
        return float('inf'), float('inf'), float('inf')

def generate_perlin_parameters(exploration_factor: float = 1.0) -> PerlinParams:
    """Generate random parameters for Perlin noise with adjustable exploration factor."""
    # Higher exploration_factor means more diverse parameters
    # Start with more aggressive parameters to get away from crystal structure
    return {
        'seed': random.randint(1, 10000),
        'scale': random.uniform(1.0, 20.0) * exploration_factor,
        'amplitude': random.uniform(0.1, 0.4) * exploration_factor,  # Increased amplitude
        'octaves': random.randint(1, 5),
        'persistence': random.uniform(0.3, 0.8),
        'lacunarity': random.uniform(1.5, 3.0) * exploration_factor,
        'noise_type': random.choice(['perlin', 'simplex', 'combined'])  # Add different noise types
    }

def perturb_parameters(params: PerlinParams, key: str, step_size: float) -> PerlinParams:
    """Perturb a specific parameter by a step size."""
    new_params = params.copy()
    
    if key == 'noise_type':
        # Special handling for categorical parameter
        choices = ['perlin', 'simplex', 'combined']
        current_idx = choices.index(params[key])
        new_idx = (current_idx + 1) % len(choices)  # Just cycle through options
        new_params[key] = choices[new_idx]
    else:
        new_params[key] += step_size
        
        # Ensure parameters stay in valid ranges
        if key == 'scale':
            new_params[key] = max(0.1, min(30.0, new_params[key]))
        elif key == 'amplitude':
            new_params[key] = max(0.01, min(1.0, new_params[key]))
        elif key == 'persistence':
            new_params[key] = max(0.1, min(0.9, new_params[key]))
        elif key == 'lacunarity':
            new_params[key] = max(1.1, min(5.0, new_params[key]))
        elif key == 'octaves':
            new_params[key] = max(1, min(8, round(new_params[key])))
    
    return new_params

def apply_combined_noise(pos: np.ndarray, params: PerlinParams) -> np.ndarray:
    """Apply a combination of Perlin and Simplex noise."""
    seed = params['seed']
    scale = params['scale']
    amplitude = params['amplitude']
    octaves = params['octaves']
    persistence = params['persistence']
    lacunarity = params['lacunarity']

    # Use different seeds for different noise components
    perlin_seed = seed
    # simplex_seed = seed + 1000  # Not used since we can't seed simplex

    # Apply Perlin noise
    perlin_val = noise.pnoise3(
        pos[0] * scale,
        pos[1] * scale,
        pos[2] * scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        base=perlin_seed
    )

    # Apply Simplex noise (different scale) - removed the base parameter
    simplex_val = noise.snoise3(
        pos[0] * scale * 1.5,
        pos[1] * scale * 1.5,
        pos[2] * scale * 1.5,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity
    )

    # Combine with different weights
    return 0.7 * perlin_val + 0.3 * simplex_val

def apply_perlin_displacement(
    positions: np.ndarray,
    lattice: np.ndarray,
    params: PerlinParams
) -> np.ndarray:
    """Apply noise displacement to positions based on selected noise type."""
    # Create a copy of positions
    new_positions = positions.copy()

    noise_type = params['noise_type']
    seed = params['seed']
    scale = params['scale']
    amplitude = params['amplitude']
    octaves = params['octaves']
    persistence = params['persistence']
    lacunarity = params['lacunarity']

    # Apply noise displacement
    for i, pos in enumerate(positions):
        # Apply noise displacement
        displacement = np.zeros(3)
        for dim in range(3):
            # Offset each dimension to get different noise patterns
            offset = dim * 1000

            if noise_type == 'perlin':
                # Use Perlin noise
                noise_val = noise.pnoise3(
                    pos[0] * scale,
                    pos[1] * scale,
                    pos[2] * scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=seed + offset
                )
            elif noise_type == 'simplex':
                # Use Simplex noise without the base parameter
                noise_val = noise.snoise3(
                    pos[0] * scale,
                    pos[1] * scale,
                    pos[2] * scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity
                    # No base parameter
                )
            else:  # 'combined'
                # Use combined noise
                noise_val = apply_combined_noise(
                    np.array([pos[0], pos[1], pos[2] + offset/1000]),
                    params
                )

            displacement[dim] = noise_val * amplitude

        # Apply displacement
        new_positions[i] = pos + displacement

        # Ensure positions stay within [0, 1] range (periodic boundary conditions)
        new_positions[i] = new_positions[i] % 1.0

    return new_positions

def build_kdtree(
    positions: np.ndarray, 
    lattice: np.ndarray
) -> Tuple[cKDTree, np.ndarray]:
    """Build KD-tree for efficient distance queries."""
    # Convert to Cartesian coordinates
    cart_positions = np.dot(positions, lattice)
    
    # Build KD-tree
    kdtree = cKDTree(cart_positions, boxsize=None)  # No periodic boundary yet
    
    return kdtree, cart_positions

def check_min_distance_kdtree(
    positions: np.ndarray, 
    min_distance: float, 
    lattice: np.ndarray
) -> bool:
    """Check if all atoms satisfy minimum distance constraint using KD-tree."""
    if min_distance <= 0:
        return True
    
    # Convert to Cartesian coordinates
    cart_positions = np.dot(positions, lattice)
    
    # Get number of atoms
    n_atoms = len(positions)
    
    # Create KD-tree
    kdtree = cKDTree(cart_positions)
    
    # Query pairs with distance less than min_distance
    # We add a small buffer to account for numerical precision
    pairs = kdtree.query_pairs(min_distance * 1.001)
    
    # If any pairs found, minimum distance constraint is violated
    return len(pairs) == 0

def get_min_distance_matrix(structure: Structure) -> np.ndarray:
    """Calculate minimum distance matrix between all atoms."""
    positions = structure['positions']
    lattice = structure['lattice']
    
    # Convert to Cartesian coordinates
    cart_positions = np.dot(positions, lattice)
    
    # Get number of atoms
    n_atoms = len(positions)
    
    # Initialize distance matrix
    dist_matrix = np.zeros((n_atoms, n_atoms))
    
    # Create KD-tree for efficient distance calculation
    kdtree = cKDTree(cart_positions)
    
    # Calculate all pairwise distances
    for i in range(n_atoms):
        # Query distances to all other points
        distances, indices = kdtree.query(cart_positions[i:i+1], k=n_atoms)
        distances = distances[0]
        indices = indices[0]
        
        # Fill distance matrix
        for j, idx in enumerate(indices):
            if i != idx:  # Skip self
                dist_matrix[i, idx] = distances[j]
                dist_matrix[idx, i] = distances[j]
    
    return dist_matrix

def try_generate_structure(
    crystal_structure: Structure, 
    params: PerlinParams, 
    min_distance: float, 
    max_attempts: int = 50
) -> Optional[Structure]:
    """Try to generate a structure with noise that satisfies constraints."""
    try:
        lattice = crystal_structure['lattice']
        positions = crystal_structure['positions']
        symbols = crystal_structure['symbols']
        
        # Gradually reduce amplitude if we can't satisfy min_distance
        current_amplitude = params['amplitude']
        amplitude_factor = 0.9
        
        for attempt in range(max_attempts):
            # Apply noise displacement
            new_positions = apply_perlin_displacement(positions, lattice, params)
            
            # Check minimum distance constraint with KD-tree (much faster)
            if min_distance <= 0 or check_min_distance_kdtree(new_positions, min_distance, lattice):
                # Create new structure
                return {
                    'lattice': lattice,
                    'positions': new_positions,
                    'symbols': symbols
                }
            else:
                # Reduce amplitude and try again
                params['amplitude'] *= amplitude_factor
                
                # If amplitude gets too small, break
                if params['amplitude'] < 0.001:
                    break
        
        # If we couldn't generate a valid structure, return None
        return None
    except Exception as e:
        logger.error(f"Error generating structure: {e}")
        traceback.print_exc()
        return None

def adaptive_step_size(current_dist: float, threshold: float, param_key: str) -> float:
    """Calculate adaptive step size based on current distance from threshold."""
    # Base step sizes for each parameter
    base_steps = {
        'scale': 0.5,
        'amplitude': 0.05,
        'persistence': 0.05,
        'lacunarity': 0.2,
        'octaves': 1.0,
        'noise_type': 1.0  # For categorical, this is just a flag
    }
    
    # Calculate distance ratio (higher when far from threshold)
    dist_ratio = max(1.0, current_dist / max(0.1, threshold))
    
    # Cap the ratio to prevent too large steps
    dist_ratio = min(dist_ratio, 5.0)
    
    # Return adaptive step size
    return base_steps[param_key] * dist_ratio

def save_intermediate_result(
    structure: Structure, 
    params: PerlinParams, 
    dist: float, 
    dist1: float, 
    dist2: float, 
    iteration: int, 
    output_dir: str
) -> None:
    """Save intermediate result to files."""
    # Create intermediate directory if it doesn't exist
    intermediate_dir = os.path.join(output_dir, "intermediates")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Save structure to file
    output_file = os.path.join(intermediate_dir, f"structure_iter_{iteration}.vasp")
    write_poscar(output_file, structure)
    
    # Save parameters to file
    params_file = os.path.join(intermediate_dir, f"params_iter_{iteration}.txt")
    with open(params_file, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"total_distance: {dist}\n")
        f.write(f"distance_dim1: {dist1}\n")
        f.write(f"distance_dim2: {dist2}\n")
    
    logger.info(f"Saved intermediate result at iteration {iteration} with distance {dist:.4f}")

def gradient_descent_with_momentum(args: Tuple) -> Tuple:
    """Worker function using gradient descent with momentum to minimize Wasserstein distance."""
    try:
        crystal_structure, pd1_ref, pd2_ref, threshold, min_distance, max_iterations, delta, p, output_dir, worker_id = args
        
        logger.info(f"Worker {worker_id}: Starting gradient descent optimization...")
        
        # Get initial min_distance from crystal structure to understand constraints
        dist_matrix = get_min_distance_matrix(crystal_structure)
        min_dist_crystal = np.min(dist_matrix[dist_matrix > 0])
        
        logger.info(f"Worker {worker_id}: Minimum distance in crystal: {min_dist_crystal:.4f} Å")
        logger.info(f"Worker {worker_id}: Requested minimum distance: {min_distance:.4f} Å")
        
        # Adjust min_distance if it's too strict compared to crystal
        if min_distance > 0.8 * min_dist_crystal:
            adjusted_min_distance = 0.8 * min_dist_crystal
            logger.info(f"Worker {worker_id}: Adjusting minimum distance to {adjusted_min_distance:.4f} Å (80% of crystal minimum)")
            min_distance = adjusted_min_distance
        
        # Initial parameters with higher amplitude to get away from crystal structure
        params = generate_perlin_parameters(exploration_factor=1.0)
        
        # Generate initial structure
        logger.info(f"Worker {worker_id}: Generating initial structure...")
        current_structure = try_generate_structure(crystal_structure, params, min_distance, max_attempts=100)
        
        if current_structure is None:
            # Try again with different parameters
            logger.info(f"Worker {worker_id}: Initial structure generation failed, trying with different parameters...")
            for attempt in range(50):  # Increased attempts
                # Gradually decrease exploration factor to make smaller perturbations
                exploration_factor = 1.0 * (1.0 - attempt/50)
                params = generate_perlin_parameters(exploration_factor=exploration_factor)
                current_structure = try_generate_structure(crystal_structure, params, min_distance, max_attempts=100)
                if current_structure is not None:
                    logger.info(f"Worker {worker_id}: Found valid initial structure after {attempt+1} attempts")
                    break
            
            if current_structure is None:
                logger.error(f"Worker {worker_id}: Failed to generate initial structure after multiple attempts")
                # As a last resort, try with a much smaller minimum distance
                last_resort_min_distance = min_distance * 0.5
                logger.info(f"Worker {worker_id}: Trying with reduced minimum distance: {last_resort_min_distance:.4f} Å")
                params = generate_perlin_parameters(exploration_factor=0.5)
                current_structure = try_generate_structure(crystal_structure, params, last_resort_min_distance, max_attempts=100)
                
                if current_structure is None:
                    return None, None, None, float('inf'), float('inf'), float('inf')
        
        # Compute initial persistence diagrams and distance
        logger.info(f"Worker {worker_id}: Computing initial persistence diagrams...")
        pd_list_current, pd1_current, pd2_current = compute_persistence_diagrams(current_structure)
        
        if pd1_current is None or pd2_current is None:
            logger.error(f"Worker {worker_id}: Failed to compute initial persistence diagrams")
            return None, None, None, float('inf'), float('inf'), float('inf')
        
        current_dist, dist1, dist2 = compute_wasserstein_distance(
            pd1_ref, pd2_ref, pd1_current, pd2_current, p=p, delta=delta
        )
        
        logger.info(f"Worker {worker_id}: Initial Wasserstein distance: {current_dist:.4f}")
        
        # Store best result
        best_structure = current_structure
        best_pd_list = pd_list_current
        best_params = params.copy()
        best_dist = current_dist
        best_dist1 = dist1
        best_dist2 = dist2
        
        # Save initial best result
        save_intermediate_result(
            best_structure, best_params, best_dist, best_dist1, best_dist2, 0, 
            os.path.join(output_dir, f"worker_{worker_id}")
        )
        
        # Initial learning rate and momentum
        learning_rate = 0.2
        lr_decay = 0.997
        momentum = 0.8
        
        # Parameters to optimize
        param_keys = ['scale', 'amplitude', 'persistence', 'lacunarity', 'octaves', 'noise_type']
        
        # Initialize momentum vectors
        momentum_vectors = {key: 0.0 for key in param_keys}
        
        # For tracking progress
        no_improvement_count = 0
        plateau_count = 0
        last_best_dist = best_dist
        last_save_iteration = 0
        
        # Main optimization loop
        for iteration in range(max_iterations):
            improved = False
            gradients = {key: 0.0 for key in param_keys}
            
            # Report progress every 10 iterations
            if iteration % 10 == 0:
                logger.info(f"Worker {worker_id}: Iteration {iteration}, current distance: {current_dist:.4f}, best distance: {best_dist:.4f}")
            
            # Estimate gradients for each parameter
            for key in param_keys:
                # Skip noise_type for gradient calculation (will be handled separately)
                if key == 'noise_type':
                    continue
                
                # Determine adaptive step size
                step_size = adaptive_step_size(current_dist, threshold, key)
                
                if key == 'octaves':
                    # Special handling for integer parameter
                    step_size = 1
                
                # Compute gradient by finite difference
                params_plus = perturb_parameters(params, key, step_size)
                params_minus = perturb_parameters(params, key, -step_size)
                
                # Try positive direction
                structure_plus = try_generate_structure(crystal_structure, params_plus, min_distance)
                if structure_plus is not None:
                    pd_list_plus, pd1_plus, pd2_plus = compute_persistence_diagrams(structure_plus)
                    if pd1_plus is not None and pd2_plus is not None:
                        dist_plus, _, _ = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_plus, pd2_plus, p=p, delta=delta
                        )
                    else:
                        dist_plus = float('inf')
                else:
                    dist_plus = float('inf')
                
                # Try negative direction
                structure_minus = try_generate_structure(crystal_structure, params_minus, min_distance)
                if structure_minus is not None:
                    pd_list_minus, pd1_minus, pd2_minus = compute_persistence_diagrams(structure_minus)
                    if pd1_minus is not None and pd2_minus is not None:
                        dist_minus, _, _ = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_minus, pd2_minus, p=p, delta=delta
                        )
                    else:
                        dist_minus = float('inf')
                else:
                    dist_minus = float('inf')
                
                # Calculate approximate gradient
                if dist_plus != float('inf') and dist_minus != float('inf'):
                    gradients[key] = (dist_plus - dist_minus) / (2 * step_size)
                elif dist_plus != float('inf'):
                    gradients[key] = (dist_plus - current_dist) / step_size
                elif dist_minus != float('inf'):
                    gradients[key] = (current_dist - dist_minus) / step_size
                else:
                    gradients[key] = 0.0
                
                # Directly check if either direction is better
                if dist_plus < current_dist:
                    params = params_plus.copy()
                    current_structure = structure_plus
                    pd_list_current = pd_list_plus
                    current_dist = dist_plus
                    improved = True
                    
                    if dist_plus < best_dist:
                        best_structure = structure_plus
                        best_pd_list = pd_list_plus
                        best_params = params_plus.copy()
                        best_dist = dist_plus
                        best_dist1, best_dist2 = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_plus, pd2_plus, p=p, delta=delta
                        )[1:]
                        
                        # Save intermediate result when we find a better solution
                        save_intermediate_result(
                            best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                            os.path.join(output_dir, f"worker_{worker_id}")
                        )
                        last_save_iteration = iteration
                        
                        # If below threshold, we can return early
                        if best_dist < threshold:
                            logger.info(f"Worker {worker_id}: Found solution below threshold: {best_dist:.4f}")
                            return best_structure, best_pd_list, best_params, best_dist, best_dist1, best_dist2
                    
                if dist_minus < current_dist:
                    params = params_minus.copy()
                    current_structure = structure_minus
                    pd_list_current = pd_list_minus
                    current_dist = dist_minus
                    improved = True
                    
                    if dist_minus < best_dist:
                        best_structure = structure_minus
                        best_pd_list = pd_list_minus
                        best_params = params_minus.copy()
                        best_dist = dist_minus
                        best_dist1, best_dist2 = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_minus, pd2_minus, p=p, delta=delta
                        )[1:]
                        
                        # Save intermediate result when we find a better solution
                        save_intermediate_result(
                            best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                            os.path.join(output_dir, f"worker_{worker_id}")
                        )
                        last_save_iteration = iteration
                        
                        # If below threshold, we can return early
                        if best_dist < threshold:
                            logger.info(f"Worker {worker_id}: Found solution below threshold: {best_dist:.4f}")
                            return best_structure, best_pd_list, best_params, best_dist, best_dist1, best_dist2
            
            # Try different noise type (categorical parameter)
            if not improved and iteration % 5 == 0:  # Try every 5 iterations
                new_params = perturb_parameters(params, 'noise_type', 0)  # Just cycles through options
                new_structure = try_generate_structure(crystal_structure, new_params, min_distance)
                
                if new_structure is not None:
                    pd_list_new, pd1_new, pd2_new = compute_persistence_diagrams(new_structure)
                    if pd1_new is not None and pd2_new is not None:
                        new_dist, new_dist1, new_dist2 = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_new, pd2_new, p=p, delta=delta
                        )

                        if new_dist < current_dist:
                            params = new_params.copy()
                            current_structure = new_structure
                            pd_list_current = pd_list_new
                            current_dist = new_dist
                            improved = True

                            if new_dist < best_dist:
                                best_structure = new_structure
                                best_pd_list = pd_list_new
                                best_params = new_params.copy()
                                best_dist = new_dist
                                best_dist1 = new_dist1
                                best_dist2 = new_dist2

                                # Save intermediate result
                                save_intermediate_result(
                                    best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                                    os.path.join(output_dir, f"worker_{worker_id}")
                                )
                                last_save_iteration = iteration

                                # If below threshold, we can return early
                                if best_dist < threshold:
                                    logger.info(f"Worker {worker_id}: Found solution below threshold: {best_dist:.4f}")
                                    return best_structure, best_pd_list, best_params, best_dist, best_dist1, best_dist2

            # If no direct improvement, apply gradient descent with momentum
            if not improved:
                # Update momentum vectors
                for key in param_keys:
                    if key != 'noise_type':  # Skip categorical parameter
                        momentum_vectors[key] = momentum * momentum_vectors[key] - learning_rate * gradients[key]

                # Apply momentum update
                new_params = params.copy()
                for key in param_keys:
                    if key != 'noise_type':  # Skip categorical parameter
                        if key == 'octaves':
                            # Special handling for integer parameter
                            new_params[key] = max(1, min(8, round(params[key] + momentum_vectors[key])))
                        else:
                            new_params[key] = params[key] + momentum_vectors[key]

                # Try new parameters
                new_structure = try_generate_structure(crystal_structure, new_params, min_distance)

                if new_structure is not None:
                    pd_list_new, pd1_new, pd2_new = compute_persistence_diagrams(new_structure)
                    if pd1_new is not None and pd2_new is not None:
                        new_dist, new_dist1, new_dist2 = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_new, pd2_new, p=p, delta=delta
                        )

                        if new_dist < current_dist:
                            params = new_params.copy()
                            current_structure = new_structure
                            pd_list_current = pd_list_new
                            current_dist = new_dist
                            improved = True

                            if new_dist < best_dist:
                                best_structure = new_structure
                                best_pd_list = pd_list_new
                                best_params = new_params.copy()
                                best_dist = new_dist
                                best_dist1 = new_dist1
                                best_dist2 = new_dist2

                                # Save intermediate result
                                save_intermediate_result(
                                    best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                                    os.path.join(output_dir, f"worker_{worker_id}")
                                )
                                last_save_iteration = iteration

                                # If below threshold, we can return early
                                if best_dist < threshold:
                                    logger.info(f"Worker {worker_id}: Found solution below threshold: {best_dist:.4f}")
                                    return best_structure, best_pd_list, best_params, best_dist, best_dist1, best_dist2

            # Adaptive learning rate schedule
            learning_rate *= lr_decay

            # Track improvement
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # Check if we're stuck in a plateau
            if abs(best_dist - last_best_dist) < 0.01:
                plateau_count += 1
            else:
                plateau_count = 0
                last_best_dist = best_dist

            # If stuck for too long, try random restart
            if no_improvement_count >= 20 or plateau_count >= 30:
                logger.info(f"Worker {worker_id}: No improvement for {no_improvement_count} iterations or plateau for {plateau_count} iterations, trying random restart")

                # Save current best if not saved recently
                if iteration - last_save_iteration > 20:
                    save_intermediate_result(
                        best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                        os.path.join(output_dir, f"worker_{worker_id}")
                    )
                    last_save_iteration = iteration

                # Try more aggressive exploration
                exploration_factor = 1.0 + (no_improvement_count / 20.0)  # Increase exploration with stagnation
                params = generate_perlin_parameters(exploration_factor=exploration_factor)

                # Generate new structure
                new_structure = try_generate_structure(crystal_structure, params, min_distance)

                if new_structure is not None:
                    pd_list_new, pd1_new, pd2_new = compute_persistence_diagrams(new_structure)
                    if pd1_new is not None and pd2_new is not None:
                        new_dist, new_dist1, new_dist2 = compute_wasserstein_distance(
                            pd1_ref, pd2_ref, pd1_new, pd2_new, p=p, delta=delta
                        )

                        # Always accept the new point to escape local minimum
                        current_structure = new_structure
                        pd_list_current = pd_list_new
                        current_dist = new_dist

                        # Update best if better
                        if new_dist < best_dist:
                            best_structure = new_structure
                            best_pd_list = pd_list_new
                            best_params = params.copy()
                            best_dist = new_dist
                            best_dist1 = new_dist1
                            best_dist2 = new_dist2

                            # Save intermediate result
                            save_intermediate_result(
                                best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                                os.path.join(output_dir, f"worker_{worker_id}")
                            )
                            last_save_iteration = iteration

                # Reset counters and momentum
                no_improvement_count = 0
                plateau_count = 0
                momentum_vectors = {key: 0.0 for key in param_keys}
                learning_rate = 0.2  # Reset learning rate

            # Occasionally try a robust noise approach (every 50 iterations)
            if iteration % 50 == 0 and iteration > 0:
                logger.info(f"Worker {worker_id}: Trying robust noise approach at iteration {iteration}")

                # Apply a more robust noise with gradually increasing amplitude
                robust_params = params.copy()
                robust_params['amplitude'] = min(0.4, params['amplitude'] * 1.5)  # Increase amplitude
                robust_params['scale'] = params['scale'] * 0.8  # Decrease scale for finer details

                # Try multiple noise types
                for noise_type in ['perlin', 'simplex', 'combined']:
                    robust_params['noise_type'] = noise_type

                    # Generate new structure
                    robust_structure = try_generate_structure(crystal_structure, robust_params, min_distance)

                    if robust_structure is not None:
                        pd_list_robust, pd1_robust, pd2_robust = compute_persistence_diagrams(robust_structure)
                        if pd1_robust is not None and pd2_robust is not None:
                            robust_dist, robust_dist1, robust_dist2 = compute_wasserstein_distance(
                                pd1_ref, pd2_ref, pd1_robust, pd2_robust, p=p, delta=delta
                            )

                            if robust_dist < current_dist:
                                params = robust_params.copy()
                                current_structure = robust_structure
                                pd_list_current = pd_list_robust
                                current_dist = robust_dist

                                if robust_dist < best_dist:
                                    best_structure = robust_structure
                                    best_pd_list = pd_list_robust
                                    best_params = robust_params.copy()
                                    best_dist = robust_dist
                                    best_dist1 = robust_dist1
                                    best_dist2 = robust_dist2

                                    # Save intermediate result
                                    save_intermediate_result(
                                        best_structure, best_params, best_dist, best_dist1, best_dist2, iteration,
                                        os.path.join(output_dir, f"worker_{worker_id}")
                                    )
                                    last_save_iteration = iteration

                                    # If below threshold, we can return early
                                    if best_dist < threshold:
                                        logger.info(f"Worker {worker_id}: Found solution below threshold with robust noise: {best_dist:.4f}")
                                        return best_structure, best_pd_list, best_params, best_dist, best_dist1, best_dist2

                                # Break the loop if we found an improvement
                                break

        # Final save if not saved recently
        if max_iterations - last_save_iteration > 10:
            save_intermediate_result(
                best_structure, best_params, best_dist, best_dist1, best_dist2, max_iterations,
                os.path.join(output_dir, f"worker_{worker_id}")
            )

        logger.info(f"Worker {worker_id}: Finished optimization. Best distance: {best_dist:.4f}")
        return best_structure, best_pd_list, best_params, best_dist, best_dist1, best_dist2

    except Exception as e:
        logger.error(f"Worker {worker_id}: Error during optimization: {e}")
        traceback.print_exc()
        return None, None, None, float('inf'), float('inf'), float('inf')

def main():
    """Main function to generate amorphous structures."""
    # Parse command line arguments
    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read crystal and reference structures
    logger.info(f"Reading crystal structure from {args.crystal}")
    crystal_structure = read_poscar(args.crystal)

    logger.info(f"Reading reference amorphous structure from {args.reference}")
    reference_structure = read_poscar(args.reference)

    # Compute persistence diagrams for reference structure
    logger.info("Computing persistence diagrams for reference structure")
    pd_list_ref, pd1_ref, pd2_ref = compute_persistence_diagrams(reference_structure)

    if pd1_ref is None or pd2_ref is None:
        logger.error("Failed to compute persistence diagrams for reference structure")
        return

    # Generate structures in parallel
    logger.info(f"Generating {args.num_structures} amorphous structures using {args.num_workers} workers")

    # Prepare worker arguments
    worker_args = []
    for i in range(args.num_structures):
        worker_args.append((
            crystal_structure, pd1_ref, pd2_ref, args.threshold, args.min_distance,
            args.max_iterations, args.delta, args.p, args.output_dir, i
        ))

    # Run workers in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(gradient_descent_with_momentum, worker_args))

    # Process results
    valid_results = []
    for i, (structure, pd_list, params, dist, dist1, dist2) in enumerate(results):
        if structure is not None:
            valid_results.append((structure, pd_list, params, dist, dist1, dist2, i))

    # Sort by distance
    valid_results.sort(key=lambda x: x[3])

    # Save results
    for i, (structure, _, params, dist, dist1, dist2, worker_id) in enumerate(valid_results):
        # Save structure
        output_file = os.path.join(args.output_dir, f"structure_{i+1}.vasp")
        write_poscar(output_file, structure)

        # Save parameters
        params_file = os.path.join(args.output_dir, f"params_{i+1}.txt")
        with open(params_file, 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"total_distance: {dist}\n")
            f.write(f"distance_dim1: {dist1}\n")
            f.write(f"distance_dim2: {dist2}\n")
            f.write(f"worker_id: {worker_id}\n")

        logger.info(f"Saved structure {i+1} with distance {dist:.4f}")

    logger.info(f"Generated {len(valid_results)} valid structures out of {args.num_structures} attempts")

    if valid_results:
        best_dist = valid_results[0][3]
        logger.info(f"Best Wasserstein distance: {best_dist:.4f}")

        if best_dist < args.threshold:
            logger.info("Found structure below threshold!")
        else:
            logger.info(f"No structure below threshold ({args.threshold})")
    else:
        logger.warning("No valid structures generated")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
