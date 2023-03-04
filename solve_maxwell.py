# Solver for the problem details in [1]. NOTE: If beta = 0, problem is
# only semi-definite and right hand side must satisfy compatability
# conditions. This can be ensured by prescribing an impressed magnetic
# field, see[2]

# TODO Could construct alpha and beta poisson matrices and pass to AMS,
# see [1] for details.

# References:
# [1] https://hypre.readthedocs.io/en/latest/solvers-ams.html
# [2] Oszkar Biro, "Edge element formulations of eddy current problems"

from typing import Dict
from util import save_function
import numpy as np
from dolfinx.common import Timer, timing
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                interpolation_matrix)
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                        dirichletbc, form, locate_dofs_topological, petsc)
from dolfinx.mesh import Mesh, locate_entities_boundary
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, VectorElement, curl, dx, inner
from ufl.core.expr import Expr


from dolfinx import cpp, fem, io
from generate_team30_meshes import (domain_parameters, model_parameters,
                                    surface_map)
from mpi4py import MPI


from dolfinx.mesh import Mesh, create_unit_cube, CellType
from ufl import SpatialCoordinate, as_vector, cos, pi, curl




# ## -- Parameters -- ##

# single = False
# single_phase = single
# three = True
# apply_torque = False
# num_phases = 1
# steps_per_phase = 100
# omega_u = 0
# degree = 1
# steps = 40
# plot= False
# progress = True
# output = True
# mesh_dir = "meshes"


# freq = model_parameters["freq"]
# T = num_phases * 1 / freq
# dt_ = 1 / steps_per_phase * 1 / freq
# mu_0 = model_parameters["mu_0"]
# omega_J = 2 * np.pi * freq

# ext = "single" if single_phase else "three"
# fname = f"{mesh_dir}/{ext}_phase3D"

degree = 1




## -- Load Mesh -- ##

# with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
#         mesh = xdmf.read_mesh(name="Grid")
#         ct = xdmf.read_meshtags(mesh, name="Grid")

# tdim = mesh.topology.dim
# mesh.topology.create_connectivity(tdim - 1, 0)
# with io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
#     ft = xdmf.read_meshtags(mesh, name="Grid")

hexa = False
h = 1/16
n = int(round(1 / h))
mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=CellType.hexahedron if hexa else CellType.tetrahedron)
x = SpatialCoordinate(mesh)
u_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
f = curl(curl(u_e)) + u_e




## -- Functions and Spaces -- ##

V = FunctionSpace(mesh, ("N1curl", degree))
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

u = TrialFunction(V)
v = TestFunction(V)




## -- Weak Form -- ##

form_compiler_options = {}
jit_options = {}

alpha = Constant(mesh, 1.)
beta = Constant(mesh, 1.)
a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx,
            form_compiler_options=form_compiler_options, jit_options=jit_options)

L = form(inner(f, v) * dx,
            form_compiler_options=form_compiler_options, jit_options=jit_options)




## -- Boundary Condition Stuff -- ##

def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])

tdim = mesh.topology.dim
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)
u_bc_expr = Expression(u_e, V.element.interpolation_points())

with Timer(f"~{degree}, {ndofs}: BC interpolation"):
    u_bc = Function(V)
    u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, boundary_dofs)




## -- Assembly -- ##

u = Function(V)

# TODO More steps needed here for Dirichlet boundaries
with Timer(f"~{degree}, {ndofs}: Assemble LHS and RHS"):
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.assemble_vector(L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])





## -- Solver Set Up -- ##

preconditioner = "ams"
petsc_options = {}
ams_options = {"pc_hypre_ams_cycle_type": 1,
                "pc_hypre_ams_tol": 1e-8,
                "solver_atol": 1e-8, "solver_rtol": 1e-8,
                "solver_initial_guess_nonzero": True,
                "solver_type": "gmres"}

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

    # Build discrete gradient
    with Timer(f"~{degree}, {ndofs}: Build discrete gradient"):
        V_CG = FunctionSpace(mesh, ("CG", degree))._cpp_object
        G = discrete_gradient(V_CG, V._cpp_object)
        G.assemble()
        pc.setHYPREDiscreteGradient(G)

    W = FunctionSpace(mesh, ("CG", degree))
    G = discrete_gradient(W._cpp_object, V._cpp_object)
    G.assemble()

    qel = VectorElement("CG", mesh.ufl_cell(), degree)
    Q3 = FunctionSpace(mesh, qel)
    Pi = interpolation_matrix(Q3._cpp_object, V._cpp_object)
    Pi.assemble()

    ksp.pc.setHYPREDiscreteGradient(G)
    ksp.pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)


    # If we are dealing with a zero conductivity problem (no mass
    # term),need to tell the preconditioner
    if np.isclose(beta.value, 0):
        pc.setHYPRESetBetaPoissonMatrix(None)

elif preconditioner == "gamg":
    pc.setType("gamg")

# Set matrix operator




## -- Solve -- ##

def monitor(ksp, its, rnorm):
    if mesh.comm.rank == 0:
        print("Iteration: {}, rel. residual: {}".format(its, rnorm))
ksp.setMonitor(monitor)
ksp.setFromOptions()
pc.setUp()
ksp.setUp()

# Compute solution
with Timer(f"~{degree}, {ndofs}: Solve Problem"):
    ksp.solve(b, u.vector)
    u.x.scatter_forward()

reason = ksp.getConvergedReason()
print(f"Convergence reason {reason}")
if reason < 0:
    u.name = "A"
    save_function(u, "error.bp")
    raise RuntimeError("Solver did not converge. Output at error.bp")
ksp.view()
print((u, {"ndofs": ndofs,
            "solve_time": timing(f"~{degree}, {ndofs}: Solve Problem")[1],
            "iterations": ksp.its}))
