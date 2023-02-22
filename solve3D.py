import argparse
import os
import sys
from io import TextIOWrapper
from typing import Callable, Optional, TextIO, Union

import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import ufl
from dolfinx import cpp, fem, io
from dolfinx.cpp.io import VTXWriter
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                   interpolation_matrix)
from mpi4py import MPI
from petsc4py import PETSc

from generate_team30_meshes import (domain_parameters, model_parameters,
                                    surface_map)
from utils import DerivedQuantities2D, MagneticField2D, update_current_density



## -- Parameters -- ##

single = False
single_phase = single
three = True
apply_torque = False
num_phases = 1
steps_per_phase = 100
omega_u = 0
degree = 1
steps = 40
plot= False
progress = True
output = False
mesh_dir = "meshes"


freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1 / steps_per_phase * 1 / freq
mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

ext = "single" if single_phase else "three"
fname = f"{mesh_dir}/{ext}_phase3D"



## -- Load Mesh -- ##

with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, 0)
with io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")



## -- Functions and Spaces -- ##

domains, currents = domain_parameters(single_phase)

DG0 = fem.FunctionSpace(mesh, ("DG", 0))
mu_R = fem.Function(DG0)
sigma = fem.Function(DG0)
density = fem.Function(DG0)
for (material, domain) in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]


cell = mesh.ufl_cell()

FE_scalar = ufl.FiniteElement("Lagrange", cell, 1)
FE_vector = ufl.FiniteElement("N1curl", cell, 1)
ME = ufl.MixedElement([FE_vector, FE_scalar])
VQ = fem.FunctionSpace(mesh, ME)

A, V = ufl.TrialFunctions(VQ)
v, q = ufl.TestFunctions(VQ)
AnVn = fem.Function(VQ)
An, _ = ufl.split(AnVn)  # Solution at previous time step
J0z = fem.Function(DG0)  # FIXME: Vector current in future


# Create integration sets
Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

# Create integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])

# Define temporal and spatial parameters
n = ufl.FacetNormal(mesh)
dt = fem.Constant(mesh, dt_)
x = ufl.SpatialCoordinate(mesh)

omega = fem.Constant(mesh, PETSc.ScalarType(omega_u))



## -- Weak Form -- ##

a00 = dt / mu_R * ufl.inner(ufl.curl(A), ufl.curl(v)) * dx(Omega_n + Omega_c) # curl curl
a00 += dt / mu_R * ufl.inner(v, ufl.cross(n, ufl.curl(A))) * ds
a00 += mu_0 * sigma * ufl.inner(A, v) * dx(Omega_c)
a01 = dt * mu_0 * sigma * ufl.inner(ufl.grad(V), v) * dx(Omega_c)
a11 = dt * mu_0 * sigma * ufl.inner(ufl.grad(V), ufl.grad(q)) * dx(Omega_c) # poisson
L = dt * mu_0 * J0z * v[2] * dx(Omega_n)

# Motion voltage term
u = omega * ufl.as_vector((-x[1], x[0], 0)) # try various omega to verify this is right

a00 += dt * mu_0 * sigma * ufl.inner(ufl.cross(u, ufl.curl(A)), v) * dx(Omega_c)

a = a00 + a01 + a11 # treat these seperately



## -- Boundary Condition Stuff -- ##

# Find all dofs in Omega_n for Q-space
cells_n = np.hstack([ct.find(domain) for domain in Omega_n])
Q, _ = VQ.sub(1).collapse()
deac_dofs = fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

# Create zero condition for V in Omega_n
zeroQ = fem.Function(Q)
zeroQ.x.array[:] = 0
bc_Q = fem.dirichletbc(zeroQ, deac_dofs, VQ.sub(1))

# Create external boundary condition for V space
V_, _ = VQ.sub(0).collapse()
tdim = mesh.topology.dim

def boundary(x):
    return np.full(x.shape[1], True)

boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, boundary)
bndry_dofs = fem.locate_dofs_topological((VQ.sub(0), V_), tdim - 1, boundary_facets)
zeroV = fem.Function(V_)
zeroV.x.array[:] = 0
bc_V = fem.dirichletbc(zeroV, bndry_dofs, VQ.sub(0))
bcs = [bc_V, bc_Q]



## -- Assembly with Sparsity Stuff -- ##

form_compiler_options = {}
jit_parameters = {}

cpp_a = fem.form(a, form_compiler_options=form_compiler_options, jit_options=jit_parameters)

pattern = fem.create_sparsity_pattern(cpp_a)
block_size = VQ.dofmap.index_map_bs
deac_blocks = deac_dofs[0] // block_size
pattern.insert_diagonal(deac_blocks)
pattern.assemble()

# Create matrix based on sparsity pattern
A = cpp.la.petsc.create_matrix(mesh.comm, pattern)
A.zeroEntries()
if not apply_torque:
    A.zeroEntries()
    fem.petsc.assemble_matrix(A, cpp_a, bcs=bcs)  # type: ignore
    A.assemble()

# Create inital vector for LHS
cpp_L = fem.form(L, form_compiler_options=form_compiler_options, jit_options=jit_parameters)
b = fem.petsc.create_vector(cpp_L)



## -- Solver Set Up -- ##

petsc_options = {"ksp_type": "cg", "pc_type": "ams"}

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)

if petsc_options["pc_type"] == "ams":

    k = 1

    solver.setOptionsPrefix(f"ksp_{id(solver)}")
    pc = solver.getPC()
    M = VQ.sub(0).collapse()[0]

    # Build discrete gradient
    G = ufl.FiniteElement("Lagrange", cell, k)
    V_CG = fem.FunctionSpace(mesh, G)
    G = discrete_gradient(V_CG._cpp_object, M._cpp_object)
    G.assemble()

    qel = ufl.VectorElement("CG", mesh.ufl_cell(), k)
    Q3 = fem.FunctionSpace(mesh, qel)
    Pi = interpolation_matrix(Q3._cpp_object, M._cpp_object)
    Pi.assemble()

    pc.setType("hypre")
    pc.setHYPREType("ams")
    pc.setHYPREDiscreteGradient(G)
    pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)

else:

    prefix = "AV_"
    # Give PETSc solver options a unique prefix
    solver_prefix = "TEAM30_solve_{}".format(id(solver))
    solver.setOptionsPrefix(solver_prefix)

    # Set PETSc options
    opts = PETSc.Options()
    opts.prefixPush(solver_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    solver.setFromOptions()
    solver.setOptionsPrefix(prefix)
    solver.setFromOptions()



## -- Solve -- ##

save_output = True
outdir = "team30_3D_output"

# Function for containg the solution
AV = fem.Function(VQ)
A_out = AV.sub(0).collapse()
V_out = AV.sub(1).collapse()

# Post-processing function for projecting the magnetic field potential
post_B = MagneticField2D(AV)
# compare A and V only since this needs adapting for 3D

# Class for computing torque, losses and induced voltage
# derived = DerivedQuantities2D(AV, AnVn, u, sigma, domains, ct, ft)
# TODO: Not implemented in 3D yet
A_out.name = "A"
post_B.B.name = "B"

W = fem.VectorFunctionSpace(mesh, ("Discontinuous Lagrange", 1))
post_A = fem.Function(W)

if save_output:
    A_vtx = VTXWriter(mesh.comm, f"{outdir}/A.bp", [post_A._cpp_object])

# Computations needed for adding addiitonal torque to engine
x = ufl.SpatialCoordinate(mesh)
r = ufl.sqrt(x[0]**2 + x[1]**2)
L = 1  # Depth of domain
num_steps = int(T / float(dt.value))
times = np.zeros(num_steps + 1, dtype=PETSc.ScalarType)
# Generate initial electric current in copper windings
t = 0.
update_current_density(J0z, omega_J, t, ct, currents)

if MPI.COMM_WORLD.rank == 0 and progress:
    progressbar = tqdm.tqdm(desc="Solving time-dependent problem",
                            total=int(T / float(dt.value)))

iters = np.zeros(num_steps)

for i in range(num_steps):
    # Update time step and current density
    if MPI.COMM_WORLD.rank == 0 and progress:
        progressbar.update(1)
    t += float(dt.value)
    update_current_density(J0z, omega_J, t, ct, currents)

    # Reassemble LHS
    if apply_torque:
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, cpp_a, bcs=bcs)  # type: ignore
        A.assemble()

    # Reassemble RHS
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, cpp_L)
    fem.petsc.apply_lifting(b, [cpp_a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve problem
    solver.solve(b, AV.vector)
    iters[i] = solver.its
    AV.x.scatter_forward()

    times[i + 1] = t

    # Update previous time step
    AnVn.x.array[:] = AV.x.array
    AnVn.x.scatter_forward()

    # Write solution to file
    if save_output:
        post_A.interpolate(AV.sub(0).collapse())
        A_vtx.write(t)

if save_output:
    A_vtx.close()