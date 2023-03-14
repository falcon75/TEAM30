import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import cpp, fem, io
from dolfinx.common import Timer, timing
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                interpolation_matrix)
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                        dirichletbc, form, locate_dofs_topological, petsc)
from dolfinx.mesh import Mesh, locate_entities_boundary
from dolfinx.mesh import Mesh, create_unit_cube, CellType
from dolfinx.io import VTXWriter
from ufl import TestFunction, TrialFunction, VectorElement, curl, dx, ds, inner, cross, grad
import ufl
from ufl import SpatialCoordinate, as_vector, cos, pi, curl

from utils import DerivedQuantities2D, MagneticField2D, update_current_density
from util import save_function
from generate_team30_meshes import (domain_parameters, model_parameters,
                                    surface_map)

# Questions
# - proper way to set options for petsc, prefixes (currently a mess)

# Objectives
# - investigate time complexity of hyper and higher degrees on TEAM30 irregular mesh
# - derived quantities and time steps for complete 3D Team 30 preconditioned and verified
# - research coloumb gauge in 3D (maybe)

# ## -- Parameters -- ##

single = False
single_phase = single
three = True
apply_torque = False
num_phases = 1
steps_per_phase = 100

freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1 / steps_per_phase * 1 / freq
mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

mesh_dir = "meshes"
ext = "single" if single_phase else "three"
fname = f"{mesh_dir}/{ext}_phase3D"

domains, currents = domain_parameters(single_phase)

degree = 1



## -- Load Mesh -- ##

with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, 0)
with io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")



## -- Functions and Spaces -- ##

x = SpatialCoordinate(mesh)
cell = mesh.ufl_cell()
# n = ufl.FacetNormal(mesh)
dt = fem.Constant(mesh, dt_)

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

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
# ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])

lagrange_elem = ufl.VectorElement("Lagrange", cell, degree)
nedelec_elem = ufl.FiniteElement("N1curl", cell, degree)
A_space = FunctionSpace(mesh, nedelec_elem)

A = TrialFunction(A_space)
v = TestFunction(A_space)

A_prev = fem.Function(A_space)
J0z = fem.Function(DG0)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs



## -- Weak Form -- ##

# form_compiler_options = {}
# jit_options = {}

a = dt * 1 / mu_R * inner(curl(A), curl(v)) * dx(Omega_n + Omega_c) 
# a += 1 / mu_R * inner(v, cross(n, curl(A))) * ds(Omega_n + Omega_c) # FIXME: Zero?
a += mu_0 * sigma * inner(A, v) * dx(Omega_c)
a = form(a)
L = form(dt * mu_0 * inner(J0z, v[2]) * dx(Omega_n))



## -- Boundary Condition Stuff -- ##

def boundary_marker(x):
    return np.full(x.shape[1], True)

boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs = locate_dofs_topological(A_space, entity_dim=tdim - 1, entities=boundary_facets)

zeroA = fem.Function(A_space)
zeroA.x.array[:] = 0
bc = fem.dirichletbc(zeroA, boundary_dofs)



## -- Assembly -- ##

t = 0
update_current_density(J0z, omega_J, t, ct, currents) # must be reassembled after update

A_out = Function(A_space)
A = petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])



## -- Solver Set Up -- ##

preconditioner = "ams"
petsc_options = {"ksp_type": "cg"}
ams_options = {"pc_hypre_ams_cycle_type": 1,
                "pc_hypre_ams_tol": 1e-8,
                "ksp_atol": 1e-6, "ksp_rtol": 4e-1,
                "ksp_initial_guess_nonzero": True,
                "ksp_max_iter": 100,
                "ksp_type": "gmres"}

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")

ksp.setOperators(A)

pc = ksp.getPC()
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in petsc_options.items():
    opts[option] = value
opts.prefixPop()

if preconditioner == "ams":
    
    pc.setType("hypre")
    pc.setHYPREType("ams")

    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in ams_options.items():
        opts[option] = value
    opts.prefixPop()

    W = FunctionSpace(mesh, ("CG", degree))
    G = discrete_gradient(W._cpp_object, A_space._cpp_object)
    G.assemble()

    X = VectorElement("CG", mesh.ufl_cell(), degree)
    Q = FunctionSpace(mesh, X)
    Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
    Pi.assemble()

    ksp.pc.setHYPREDiscreteGradient(G)
    ksp.pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)

    pc.setHYPRESetBetaPoissonMatrix(None) # mass term may be close to zero, be careful

elif preconditioner == "gamg":
    pc.setType("gamg")



## -- Solve -- ##

def monitor(ksp, its, rnorm):
    if mesh.comm.rank == 0:
        print("Iteration: {}, rel. residual: {}".format(its, rnorm))

ksp.setMonitor(monitor)
ksp.setFromOptions()
pc.setUp()
ksp.setUp()
ksp.view()

with Timer(f"~{degree}, {ndofs}: Solve Problem"):
    ksp.solve(b, A_out.vector)
    A_out.x.scatter_forward()

print("Max U: ", A_out.x.array.max())
reason = ksp.getConvergedReason()
print(f"Convergence reason {reason}")
A_out.name = "A"
save_function(A_out, "team30_3D_output")

print((A_out, {"ndofs": ndofs,
            "solve_time": timing(f"~{degree}, {ndofs}: Solve Problem")[1],
            "iterations": ksp.its}))
