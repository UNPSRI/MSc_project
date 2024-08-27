import torch
from scipy.spatial import ConvexHull
import time
import sys
import torch.nn as nn
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from Bio import PDB
import numpy as np
from utilities import marsaglia_polar
warnings.filterwarnings('ignore')

class ObjFuncModule(nn.Module):
    def __init__(self, obj_func):
        super(ObjFuncModule, self).__init__()
        self.obj_func = obj_func

    def forward(self, position):
        # Convert tensor to numpy, apply the objective function, then convert back to tensor
        result = self.obj_func(position)
        return result
    
def marsaglia_method(dim):
    if dim == 2:
        while True:
            u1 = random.uniform(-1, 1)
            u2 = random.uniform(-1, 1)
            s = u1**2 + u2**2
            if s >= 1:
                continue
        
            #print (s)
            factor = np.sqrt(-2.0 * np.log(s) / s)
            z0 = u1 * factor
            z1 = u2 * factor
            return np.array( [ z0, z1 ] )
        
    if dim == 3:
        while True:
            u1 = random.uniform(-1, 1)
            u2 = random.uniform(-1, 1)
            s = u1**2 + u2**2
            if s >= 1:
                continue
            z0 = 2 * u1 * np.sqrt( 1 - u1**2 - u2**2 )
            z1 = 2 * u2 * np.sqrt( 1 - u1**2 - u2**2 )
            z2 = 1 - 2*( u1**2 + u2**2 )
            
            return np.array( [ z0, z1, z2 ] )
            
def random_perturbation(protein_structure, device):
    # Apply a small random perturbation to both positions and masses
    random_amino = random.choice(range(len(protein_structure)))
    perturbed_structure = protein_structure.clone()
    perturbed = torch.tensor(marsaglia_method(protein_structure.shape[1]),device=device, dtype=torch.float64)
    perturbed *= np.random.rand()
    perturbed_structure[random_amino] = protein_structure[random_amino].clone() + perturbed

    return perturbed_structure 

def simulated_annealing(obj_func, initial_structure, initial_temp, final_temp, alpha, \
                        max_iterations):
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    current_structure = torch.tensor(initial_structure, device=device, dtype=torch.float64, requires_grad=False)

    # Wrapping the objective function in a module and parallelizing it
    obj_func_module = ObjFuncModule(obj_func).to(device)
    obj_func_parallel = torch.nn.DataParallel(obj_func_module).to(device)
        
        
    current_loss = obj_func_parallel(current_structure.cpu().numpy())
    temperature = initial_temp
    #torch.tensor(initial_temp, device=device, dtype=torch.float64, requires_grad=False)
    steps = 1
    iteration = 0
    position = [current_structure]
    values = [current_loss]
    accept = 0
    start_time = time.time()
    while temperature > final_temp and iteration < max_iterations:
        
        
        # Metropolis criterion
        for step in range (steps):
            new_structure = random_perturbation(current_structure, device)
            new_loss = obj_func_parallel(new_structure.cpu().numpy())
            # Calculate loss difference
            delta_loss = new_loss - current_loss
            if delta_loss < 0 or np.random.rand() < np.exp(-delta_loss / temperature):
                accept += 1
                current_structure = new_structure
                current_loss = new_loss
        
        step_end_time = time.time()
        
        # Decrease the temperature
        temperature *= alpha
        iteration += 1
        
        # Print progress (optional)
        if iteration % 100 == 0 and iteration > 0:  # Print every 10 steps
            values.append(current_loss)
            position.append(current_structure)
            elapsed_time = step_end_time - start_time
            steps_remaining = max_iterations - iteration
            estimated_total_time = elapsed_time / iteration * max_iterations
            estimated_remaining_time = estimated_total_time - elapsed_time
            sys.stdout.write(f"\rStep {iteration}/{max_iterations} - Elapsed Time: {elapsed_time:.2f}s, "
                  f"Estimated Total Time: {estimated_total_time:.2f}s, "
                  f"Estimated Remaining Time: {estimated_remaining_time:.2f}s                              ")
            sys.stdout.write(f"Best Loss = {current_loss}, acceptance {accept/iteration}")
            sys.stdout.flush()



        
        
    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s")
        
        

    return current_structure, current_loss, values, position

