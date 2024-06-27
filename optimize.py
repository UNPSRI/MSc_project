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

    def forward(self, position, mass):
        # Convert tensor to numpy, apply the objective function, then convert back to tensor
        result = self.obj_func(position.cpu().numpy(), mass.cpu().numpy())
        return torch.tensor(result, dtype=torch.float32).cuda()


def obj_func_wrapper(obj_func, position):
    return torch.tensor(obj_func(position.cpu().numpy()), dtype=torch.float32).cuda()

def random_perturbation(protein_structure, masses, proposal_std):
    # Apply a small random perturbation to both positions and masses
    mass_res = list(mass_residue("7cv0.pdb").values())[:]
    hull = ConvexHull(protein_structure)
    vertice = hull.vertices
    #print (vertice)
    random_amino = random.choice(vertice)
    #print (random_amino)
    perturbed_structure = protein_structure.clone()
    perturbed = torch.normal(0, 0.1, size=(protein_structure.shape[1],))
    perturbed_structure[random_amino] = protein_structure[random_amino] + perturbed
    #print ((protein_structure == perturbed_structure).all())
    perturbed_masses = masses.clone()
    perturbed_masses[random_amino] = masses[random_amino] + random.choice(mass_res)

    return perturbed_structure, np.clip(perturbed_masses, 0.1, 10), perturbed  # Ensure masses remain positive

def metropolis_monte_carlo_backbone(obj_func, initial_position, initial_backbone, initial_mass, \
                                       steps, proposal_std, pdb_filename, temperature, gpu_ids=None):
    # Set default GPU device
    if gpu_ids is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    initial_position = torch.tensor(initial_position, device=device, dtype=torch.float32)
    current_position = torch.tensor(initial_position, device=device, dtype=torch.float32)
    current_backbone = torch.tensor(initial_backbone, device=device, dtype=torch.float32)
    current_mass = torch.tensor(initial_mass, device=device, dtype=torch.float32)
    temperature = torch.tensor(temperature, device=device, dtype=torch.float32)
    
    # Wrapping the objective function in a module and parallelizing it
    obj_func_module = ObjFuncModule(obj_func)
    if gpu_ids is not None:
        obj_func_parallel = torch.nn.DataParallel(obj_func_module, device_ids=gpu_ids).to(device)
    else:
        obj_func_parallel = torch.nn.DataParallel(obj_func_module).to(device)
        
    current_loss = obj_func_parallel(current_position,current_mass)

    best_position = current_position.clone()
    best_loss = current_loss.clone()
    best_backbone = current_backbone.clone()

    positions = [current_position.cpu().numpy()]
    values = [current_loss.cpu().item()]
    start_time = time.time()

    for step in range(steps):
        proposed_position, proposed_mass, perturbed = random_perturbation(current_position, current_mass, proposal_std)
        proposed_loss = obj_func_parallel(proposed_position.clone(),proposed_mass.clone() )
        proposed_backbone = current_backbone + perturbed
        delta_loss = torch.tensor(proposed_loss - current_loss, device=device)

        if delta_loss < 0 \
                or torch.rand(1, dtype=torch.float32, device=device) < torch.exp(- delta_loss / temperature):
            current_position = proposed_position
            current_loss = proposed_loss
            current_backbone = proposed_backbone

        
            best_loss = current_loss.clone()

        positions.append(current_position.cpu().numpy())
        values.append(current_loss.cpu().item())
        step_end_time = time.time()
        
        if step % 10 == 0 and step > 0:  # Print every 10 steps
            elapsed_time = step_end_time - start_time
            steps_remaining = steps - step
            estimated_total_time = elapsed_time / step * steps
            estimated_remaining_time = estimated_total_time - elapsed_time
            sys.stdout.write(f"\rStep {step}/{steps} - Elapsed Time: {elapsed_time:.2f}s, "
                  f"Estimated Total Time: {estimated_total_time:.2f}s, "
                  f"Estimated Remaining Time: {estimated_remaining_time:.2f}s                              ")
            sys.stdout.write(f"Minimum lost: {best_loss: 2f} ")
            sys.stdout.flush()
            
    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s")

    return current_position.cpu().numpy(), best_loss.cpu().item(), current_backbone.cpu().numpy()
        

def metropolis_monte_carlo_backbone_anim(obj_func, initial_position, initial_backbone, initial_mass, \
                                       steps, proposal_std, pdb_filename, temperature, gpu_ids=None):
    # Set default GPU device
    if gpu_ids is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    initial_position = torch.tensor(initial_position, device=device, dtype=torch.float32)
    current_position = torch.tensor(initial_position, device=device, dtype=torch.float32)
    current_backbone = torch.tensor(initial_backbone, device=device, dtype=torch.float32)
    current_mass = torch.tensor(initial_mass, device=device, dtype=torch.float32)
    temperature = torch.tensor(temperature, device=device, dtype=torch.float32)
    
    # Wrapping the objective function in a module and parallelizing it
    obj_func_module = ObjFuncModule(obj_func)
    if gpu_ids is not None:
        obj_func_parallel = torch.nn.DataParallel(obj_func_module, device_ids=gpu_ids).to(device)
    else:
        obj_func_parallel = torch.nn.DataParallel(obj_func_module).to(device)
        
    current_loss = obj_func_parallel(current_position,current_mass)

    best_position = current_position.clone()
    best_loss = current_loss.clone()
    best_backbone = current_backbone.clone()

    positions = [current_position.cpu().numpy()]
    values = [current_loss.cpu().item()]
    start_time = time.time()

    fig, ax = plt.subplots()
    scatter = ax.scatter(current_position.cpu().numpy()[:, 0], current_position.cpu().numpy()[:, 1])
    hull_lines = []
    hull_vertices = ax.plot([], [], 'ro', label='Hull Vertices')[0]

    ax.set_xlim(-10, 10)  # Set appropriate limits based on your data
    ax.set_ylim(-10, 10)
    
    def update_plot(frame):
        nonlocal current_position, current_loss, best_loss, current_backbone
        proposed_position, proposed_mass, perturbed = random_perturbation(current_position, current_mass, proposal_std)
        proposed_loss = obj_func_parallel(proposed_position.clone(),proposed_mass.clone() )
        proposed_backbone = current_backbone + perturbed
        delta_loss = torch.tensor(proposed_loss - current_loss, device=device)

        if delta_loss < 0 \
                or torch.rand(1, dtype=torch.float32, device=device) < torch.exp(- delta_loss / temperature):
            current_position = proposed_position
            current_loss = proposed_loss
            current_backbone = proposed_backbone

        
            best_loss = current_loss.clone()

        positions.append(current_position.cpu().numpy())
        values.append(current_loss.cpu().item())
        step_end_time = time.time()
        
        scatter.set_offsets(current_position.cpu().numpy())
        ax.set_title(f"Step {frame+1}/{steps}, Loss: {current_loss.cpu().item():.2f}")
        step = frame
        if step % 10 == 0 and step > 0:  # Print every 10 steps
            elapsed_time = step_end_time - start_time
            steps_remaining = steps - step
            estimated_total_time = elapsed_time / step * steps
            estimated_remaining_time = estimated_total_time - elapsed_time
            sys.stdout.write(f"\rStep {step}/{steps} - Elapsed Time: {elapsed_time:.2f}s, "
                  f"Estimated Total Time: {estimated_total_time:.2f}s, "
                  f"Estimated Remaining Time: {estimated_remaining_time:.2f}s                              ")
            sys.stdout.write(f"Minimum lost: {best_loss: 2f} ")
            sys.stdout.flush()
            
        for line in hull_lines:
            line.remove()
        hull_lines.clear()

        hull = ConvexHull(current_position.cpu().numpy())
        for simplex in hull.simplices:
            line, = ax.plot(current_position.cpu().numpy()[simplex, 0], current_position.cpu().numpy()[simplex, 1], 'r-')
            hull_lines.append(line)
        hull_vertices.set_data(current_position.cpu().numpy()[hull.vertices, 0], current_position.cpu().numpy()[hull.vertices, 1])
        
        return scatter, *hull_lines, hull_vertices

    ani = animation.FuncAnimation(fig, update_plot, frames=[i for i in range(steps)], blit=True, repeat=False)
    
    # Save the animation
    #ani.save('monte_carlo_animation-3d.gif', writer='pillow', fps=10)
    #plt.show()

    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s")

    return current_position.cpu().numpy(), best_loss.cpu().item(), current_backbone.cpu().numpy()

