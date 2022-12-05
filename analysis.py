import os
import time

import numpy as np
import pandas as pd

from team30_A_phi import solve_team30
from generate_team30_meshes import generate_team30_mesh, convert_mesh, mesh_parameters


# Solver Options
single = False
three = True
apply_torque = False
num_phases = 1
omegaU = 0
degree = 1
steps = 40
plot= False
progress = False
output = False
# petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

# CSV output filepath
path = "/Users/sam/Documents/4th Year Eng/Project/results"

# Generate Mesh Options
L = 1
res = 0.002
# single = False
# three = True
# depth = mesh_parameters["r5"]

for type, pc in [("preonly", "lu"), ("cg", "gamg")]:

    petsc_options = {"ksp_type": type, "pc_type": pc}

    data = []

    for i in range(12):

        res = 0.0005 / (1.2**i)
        
        os.system("mkdir -p meshes")

        if single:
            fname = "meshes/single_phase"
            generate_team30_mesh(fname, True, res, L)
            convert_mesh(fname, "triangle", prune_z=True)
            convert_mesh(fname, "line", prune_z=True, ext="facets")
        if three:
            fname = "meshes/three_phase"
            generate_team30_mesh(fname, False, res, L)
            convert_mesh(fname, "triangle", prune_z=True)
            convert_mesh(fname, "line", prune_z=True, ext="facets")


        start = time.time()

        def T_ext(t):
            T = num_phases * 1 / 60
            if t > 0.5 * T:
                return 1
            else:
                return 0

        if single:
            outdir = f"TEAM30_{omegaU}_single"
            os.system(f"mkdir -p {outdir}")
            its, dof = solve_team30(True, num_phases, omegaU, degree, petsc_options=petsc_options,
                            apply_torque=apply_torque, T_ext=T_ext, outdir=outdir, steps_per_phase=steps,
                            plot=plot, progress=progress, save_output=output)
        if three:
            outdir = f"TEAM30_{omegaU}_three"
            os.system(f"mkdir -p {outdir}")
            its, dof = solve_team30(False, num_phases, omegaU, degree, petsc_options=petsc_options,
                            apply_torque=apply_torque, T_ext=T_ext, outdir=outdir, steps_per_phase=steps,
                            plot=plot, progress=progress, save_output=output)
        
        data.append({"type": type, "pc": pc, "dof": dof, "its": its, "time": time.time()-start})
        print(f'Problem size {dof}: complete')

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f'{path}/team30_2D_{type}_{pc}.csv')

