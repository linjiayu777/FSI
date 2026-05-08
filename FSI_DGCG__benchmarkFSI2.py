import numpy as np
from mpi4py import MPI
import gmsh
from dolfinx import fem, io, default_real_type
from petsc4py import PETSc
import ufl
from basix.ufl import element, mixed_element
from dolfinx.io import gmsh as gmshio
from pathlib import Path
import dolfinx 
from dolfinx.fem.petsc import (assemble_matrix, apply_lifting, set_bc)
from dolfinx.mesh import create_submesh, locate_entities_boundary, meshtags
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem

# ====================== Mesh Generation in Python ======================

# ================ Geometric Parameters ================
L  = 2.5
H  = 0.41
xc = 0.2
yc = 0.2
r  = 0.05
fl = 0.35
ft = 0.02
half = ft / 2.0

gmsh.model.add("FSI_TurekHron")

fluid_marker     = 1
solid_marker     = 2
inlet_marker     = 10
outlet_marker    = 11
top_marker       = 12
bottom_marker    = 13
cylinder_marker  = 14
interface_marker = 15   # Flag fluid-structure interface (fluid side)
flag_base_marker = 16
# ====================== Channel ======================
p1 = gmsh.model.geo.addPoint(0, 0, 0)
p2 = gmsh.model.geo.addPoint(L, 0, 0)
p3 = gmsh.model.geo.addPoint(L, H, 0)
p4 = gmsh.model.geo.addPoint(0, H, 0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# ====================== Cylinder + Flag Connection Points ======================
p5 = gmsh.model.geo.addPoint(xc, yc, 0)           # Cylinder center

dx = np.sqrt(r*r - half*half)
p14 = gmsh.model.geo.addPoint(xc + dx, yc - half, 0)   # Lower connection point
p15 = gmsh.model.geo.addPoint(xc + dx, yc + half, 0)   # Upper connection point

p6  = gmsh.model.geo.addPoint(xc + r, yc, 0)      # Rightmost point
p7  = gmsh.model.geo.addPoint(xc, yc + r, 0)
p8  = gmsh.model.geo.addPoint(xc - r, yc, 0)
p9  = gmsh.model.geo.addPoint(xc, yc - r, 0)

# Cylinder Arcs
c5  = gmsh.model.geo.addCircleArc(p6,  p5, p15)   # Rightmost → upper connection
c6  = gmsh.model.geo.addCircleArc(p15, p5, p7)
c7  = gmsh.model.geo.addCircleArc(p7,  p5, p8)
c8  = gmsh.model.geo.addCircleArc(p8,  p5, p9)
c9  = gmsh.model.geo.addCircleArc(p9,  p5, p14)
c10 = gmsh.model.geo.addCircleArc(p14, p5, p6)

# ====================== Flag ======================
p10 = gmsh.model.geo.addPoint(xc + r + fl, yc - half, 0)  # Right lower
p11 = gmsh.model.geo.addPoint(xc + r + fl, yc + half, 0)  # Right upper

l11 = gmsh.model.geo.addLine(p14, p10)   # Flag lower edge
l12 = gmsh.model.geo.addLine(p10, p11)   # Flag right end
l13 = gmsh.model.geo.addLine(p11, p15)   # Flag upper edge

# ====================== Surface Definitions ======================
# solid_area
loop_flag = gmsh.model.geo.addCurveLoop([l11, l12, l13, -c5, -c10])
s2 = gmsh.model.geo.addPlaneSurface([loop_flag])

# fluid_area
loop_channel = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
loop_cylinder = gmsh.model.geo.addCurveLoop([c5, c6, c7, c8, c9, c10])
loop_flag2   = gmsh.model.geo.addCurveLoop([l11, l12, l13, -c5, -c10])

s1 = gmsh.model.geo.addPlaneSurface([loop_channel, loop_cylinder, loop_flag2])
gmsh.model.geo.synchronize()
# =================== Physical Groups ======================
gmsh.model.addPhysicalGroup(2, [s1], fluid_marker)
gmsh.model.setPhysicalName(2, fluid_marker, "Fluid")
gmsh.model.addPhysicalGroup(2, [s2], solid_marker)
gmsh.model.setPhysicalName(2, solid_marker, "Solid")

gmsh.model.addPhysicalGroup(1, [l4], inlet_marker)
gmsh.model.addPhysicalGroup(1, [l2], outlet_marker)
gmsh.model.addPhysicalGroup(1, [l3], top_marker)
gmsh.model.addPhysicalGroup(1, [l1], bottom_marker)
gmsh.model.addPhysicalGroup(1, [c5,c6,c7,c8,c9,c10], cylinder_marker)
gmsh.model.addPhysicalGroup(1, [l11,l12,l13], interface_marker)
# ====================== Mesh Setup ======================
obstacle_edges = [c5, c6, c7, c8, c9, c10, l11, l12, l13]

# Distance Field
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "EdgesList", obstacle_edges)

# Threshold Field
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", ft / 3.0)      # 0.01
gmsh.model.mesh.field.setNumber(2, "LcMax", 0.1 * H)    # 0.082
gmsh.model.mesh.field.setNumber(2, "DistMin", r/2)
gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * H)

# Background Field
gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
gmsh.model.mesh.field.setAsBackgroundMesh(3)

# ====================== Mesh Generation ======================

gmsh.model.mesh.generate(2)        # Generate 2D mesh

#gmsh.write("mesh1.msh")

# ====================== Import DOLFINx ======================
mesh_data = gmshio.model_to_mesh(
    gmsh.model,
    MPI.COMM_WORLD,
    rank=0,
    gdim=2
)

parent_mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

#------------------------Create Submesh-----------------------
tdim = parent_mesh.topology.dim 

fluid_cells = cell_tags.indices[cell_tags.values == fluid_marker]
solid_cells = cell_tags.indices[cell_tags.values == solid_marker]

fluid_mesh, fluid_to_parent_cells, _, _ = create_submesh(parent_mesh, tdim, fluid_cells)
solid_mesh, solid_to_parent_cells, _, _ = create_submesh(parent_mesh, tdim, solid_cells)

#------------------------------------------------------------
def create_facet_tags(mesh, markers):
    fdim = mesh.topology.dim - 1
    facets = []
    values = []
    for marker, locator in markers.items():
        entities = locate_entities_boundary(mesh, fdim, locator)
        facets.append(entities)
        values.append(np.full(len(entities), marker, dtype=np.int32))
    if facets:
        facets = np.hstack(facets)
        values = np.hstack(values)
        sort_idx = np.argsort(facets)
        facet_tags = meshtags(mesh, fdim, facets[sort_idx], values[sort_idx])
    else:
        facet_tags = meshtags(mesh, fdim, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
    return facet_tags

fluid_markers = {
    inlet_marker:     lambda x: np.isclose(x[0], 0.0, atol=1e-6),
    outlet_marker:    lambda x: np.isclose(x[0], L, atol=1e-6),
    top_marker:       lambda x: np.isclose(x[1], H, atol=1e-6),
    bottom_marker:    lambda x: np.isclose(x[1], 0.0, atol=1e-6),
    cylinder_marker:  lambda x: np.abs(np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r) < 1e-4,
    interface_marker: lambda x: (np.abs(x[1] - (yc + half)) < 1e-4) |
                                (np.abs(x[1] - (yc - half)) < 1e-4) |
                                (np.abs(x[0] - (xc + r + fl)) < 1e-4)
}

facet_tags_fluid = create_facet_tags(fluid_mesh, fluid_markers)

attachment_x = xc + np.sqrt(r*r - half*half)
solid_markers = {
    interface_marker: lambda x: (np.abs(x[1] - (yc + half)) < 1e-4) |
                                (np.abs(x[1] - (yc - half)) < 1e-4) |
                                (np.abs(x[0] - (xc + r + fl)) < 1e-4),
    flag_base_marker: lambda x: (
    np.abs(np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r) < 1e-5) & 
    (np.abs(x[1] - yc) < half + 1e-5)
}

facet_tags_solid = create_facet_tags(solid_mesh, solid_markers)

#----------------------------------------------------------------------
output_dir = Path("output_turek_submesh")
output_dir.mkdir(exist_ok=True)
with io.XDMFFile(fluid_mesh.comm, output_dir / "fluid_mesh.xdmf", "w") as f:
    f.write_mesh(fluid_mesh)
with io.XDMFFile(solid_mesh.comm, output_dir / "solid_mesh.xdmf", "w") as f:
    f.write_mesh(solid_mesh)


################################### ALE #########################################
gdim = 2

ale_order = 2 
V_ale = fem.functionspace(fluid_mesh, ("Lagrange", ale_order, (gdim,)))

d = fem.Function(V_ale, name="ale_displacement")
d_old = fem.Function(V_ale, name="ale_displacement")
v = fem.Function(V_ale, name="ale_velocity")

zero_disp = fem.Function(V_ale)
zero_disp.x.array[:] = 0.0

scale_det = True
scale_exp = 1.0

dx_ale = ufl.Measure("dx", domain=fluid_mesh)

# Material parameters (recommended to increase to adapt to amplitude=0.05)
a0 = fem.Constant(fluid_mesh, PETSc.ScalarType(15.0))
b0 = fem.Constant(fluid_mesh, PETSc.ScalarType(15.0))    #
kappa = fem.Constant(fluid_mesh, PETSc.ScalarType(40.0)) # 

I = ufl.Identity(gdim)
F = I + ufl.grad(d)
F = ufl.variable(F)                     

#C = F.T * F
J = ufl.det(F)
Ic_bar = J**(-2.0/gdim) * ufl.tr(F.T * F)
#I1bar = ufl.tr(J**(-2.0/gdim) * C)

psi_dev = (a0 / (2.0 * b0)) * (ufl.exp(b0 * (Ic_bar - gdim)) - 1.0)
psi_vol = (kappa / 2.0) * (J - 1)**2
psi = psi_dev + psi_vol

if scale_det:
    fac = (1.0 / J) ** scale_exp
else:
    fac = 1.0

P = ufl.diff(psi, F) * fac                # First Piola-Kirchhoff stress
# #***********************************************************************

d_test = ufl.TestFunction(V_ale)
F_ale = ufl.inner(P, ufl.grad(d_test)) * dx_ale

# #-----------------------------------------------------------------------
ale_bcs = []
fdim = fluid_mesh.topology.dim - 1

# Inlet (fixed)
inlet_dofs = fem.locate_dofs_topological(V_ale, fdim, facet_tags_fluid.find(inlet_marker))
# Walls (fixed)
top_dofs = fem.locate_dofs_topological(V_ale, fdim, facet_tags_fluid.find(top_marker))
bottom_dofs = fem.locate_dofs_topological(V_ale, fdim, facet_tags_fluid.find(bottom_marker))
# Outlet (fixed)
outlet_dofs = fem.locate_dofs_topological(V_ale, fdim, facet_tags_fluid.find(outlet_marker))
# Obstacle (prescribed motion)
cylinder_dofs = fem.locate_dofs_topological(V_ale, fdim, facet_tags_fluid.find(cylinder_marker))
#obstacle_dofs = fem.locate_dofs_topological(V_ale, fdim, facet_tags_fluid.find(interface_marker))

bc_ale_inlet   = fem.dirichletbc(zero_disp, inlet_dofs)      # displacement = 0 #should we fix the inlet ?
bc_ale_top    = fem.dirichletbc(zero_disp, top_dofs)       # displacement = 0
bc_ale_bottom = fem.dirichletbc(zero_disp, bottom_dofs)
bc_ale_outlet  = fem.dirichletbc(zero_disp, outlet_dofs)  
bc_ale_cylinder = fem.dirichletbc(zero_disp, cylinder_dofs)

bcs_ale = [bc_ale_inlet, bc_ale_outlet, bc_ale_bottom, bc_ale_top, bc_ale_cylinder]

petsc_options_ALE = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",        
    "snes_stol": 1e-12,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-8,
    "snes_max_it": 20,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",  
    "snes_monitor": None,                   
}

ALE_problem = NonlinearProblem(
    F_ale, 
    d, 
    bcs=bcs_ale,
    petsc_options=petsc_options_ALE,
    petsc_options_prefix = "ALE_solver"
)

################################## Fluid ######################################
k = 2
ve = element("Discontinuous Lagrange", fluid_mesh.basix_cell(), k, shape=(gdim,))
pe = element("Discontinuous Lagrange", fluid_mesh.basix_cell(), k-1)
me = mixed_element([ve, pe])
VQ = fem.functionspace(fluid_mesh, me)

up = ufl.TrialFunction(VQ)
vq = ufl.TestFunction(VQ)
u, p = ufl.split(up)
v, q = ufl.split(vq)

dt = fem.Constant(fluid_mesh, default_real_type(0.05))
rho = fem.Constant(fluid_mesh, PETSc.ScalarType(1000.0))
nu = fem.Constant(fluid_mesh, PETSc.ScalarType(1))
alpha = fem.Constant(fluid_mesh, PETSc.ScalarType(50*k**2))   # shall we increase penalty for stability
alpha_bd = alpha * 10
h = ufl.CellDiameter(fluid_mesh)
n_vec = ufl.FacetNormal(fluid_mesh)

ds_fluid = ufl.Measure("ds", domain=fluid_mesh, subdomain_data=facet_tags_fluid)
dS_fluid = ufl.Measure("dS", domain=fluid_mesh)
dx_fluid = ufl.Measure("dx", domain = fluid_mesh)

ds_inlet = ds_fluid(inlet_marker)
ds_top = ds_fluid(top_marker)
ds_bot = ds_fluid(bottom_marker)
ds_interface = ds_fluid(interface_marker)
ds_cylinder = ds_fluid(cylinder_marker)
ds_outlet = ds_fluid(outlet_marker)

V0, _ = VQ.sub(0).collapse()

class InletVelocity:
    def __init__(self):
        self.t = 0.0                     

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        U_bar = 1.0                 
        ramp_time = 1            
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(self.t, ramp_time) / ramp_time)) if self.t < ramp_time else 1.0
        factor = 1.5 * U_bar * 4.0 * x[1] * (H - x[1]) / (H ** 2)
        values[0] = factor * ramp
        values[1] = 0.0
        return values

class ZeroVelocity:
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        return values

u_inlet_func = fem.Function(V0)
u_inlet_func.interpolate(InletVelocity())

zero_vel_func = fem.Function(V0)
zero_vel_func.interpolate(ZeroVelocity())

u_elem = fem.Function(V0, name="u_elem")

#------------------------Diffuse Term----------------------------
def jump(phi, n):
    return ufl.outer(phi("+"), n("+")) + ufl.outer(phi("-"), n("-"))

# Interior face terms (interior penalty)
a_diffusion = nu * (
    ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_fluid
    - ufl.inner(ufl.avg(ufl.grad(u)), jump(v, n_vec)) * dS_fluid
    - ufl.inner(jump(u, n_vec), ufl.avg(ufl.grad(v))) * dS_fluid
    + (alpha / ufl.avg(h)) * ufl.inner(jump(u, n_vec), jump(v, n_vec)) * dS_fluid
)

# Dirichlet boundary terms (inlet + wall + obstacle) — only for trial u
boundary_ds = ds_inlet + ds_bot + ds_top + ds_interface + ds_cylinder
a_diffusion += nu * (
    - ufl.inner(ufl.grad(u), ufl.outer(v, n_vec)) * boundary_ds
    - ufl.inner(ufl.outer(u, n_vec), ufl.grad(v)) * boundary_ds
    + (alpha_bd / h) * ufl.inner(ufl.outer(u, n_vec), ufl.outer(v, n_vec)) * boundary_ds   #always pay attention to the alpha_bd we used here!!!
)

a_pressure = -ufl.inner(p, ufl.div(v)) * dx_fluid + ufl.inner(ufl.avg(p), ufl.jump(v, n_vec)) * dS_fluid
alpha_p = fem.Constant(fluid_mesh, PETSc.ScalarType(3))   # or 0.1~5
a_pressure_plty = +  alpha_p / ufl.avg(h) * ufl.dot(ufl.jump(p, n_vec), ufl.jump(q, n_vec)) * dS_fluid
a_pressure += a_pressure_plty

a_continuity = -ufl.inner(ufl.div(u), q) * dx_fluid + ufl.inner(ufl.jump(u, n_vec), ufl.avg(q)) * dS_fluid
a_continuity_bd = ufl.inner(ufl.dot(u, n_vec), q) * boundary_ds
a_continuity += a_continuity_bd

a_stokes = a_stokes = a_diffusion + a_pressure + a_continuity

# 
L_continuity_bd = (
      ufl.inner(ufl.dot(u_inlet_func, n_vec), q) * ds_inlet          # inlet
    - ufl.inner(ufl.dot(zero_vel_func, n_vec), q) * (ds_top + ds_bot + ds_cylinder)        # wall
    + ufl.inner(ufl.dot(u_elem, n_vec), q) * ds_interface            # obstacle: u = w → +u_elem
)

L_diffusion_bd = nu * (
    # inlet (g = u_inlet_func)
    - ufl.inner(ufl.outer(u_inlet_func, n_vec), ufl.grad(v)) * ds_inlet
    + (alpha_bd / h) * ufl.inner(ufl.outer(u_inlet_func, n_vec), ufl.outer(v, n_vec)) * ds_inlet

    # wall (g = 0)
    - ufl.inner(ufl.outer(zero_vel_func, n_vec), ufl.grad(v)) * (ds_top + ds_bot)
    + (alpha_bd / h) * ufl.inner(ufl.outer(zero_vel_func, n_vec), ufl.outer(v, n_vec)) * (ds_bot + ds_top + ds_cylinder)

    # obstacle (g = u_elem = w)   ← must be +u_elem
    - ufl.inner(ufl.outer(u_elem, n_vec), ufl.grad(v)) * ds_interface
    + (alpha_bd / h) * ufl.inner(ufl.outer(u_elem, n_vec), ufl.outer(v, n_vec)) * ds_interface
)

L_stokes  = L_continuity_bd + L_diffusion_bd

#----------------------Convection and Time-----------------------
u_n  = fem.Function(V0, name="u_n")      # u^n
u_nm1 = fem.Function(V0, name="u_nm1")

u_star = fem.Function(V0, name="u_star") #extrapolation velocity
u_star_rel = fem.Function(V0, name="rel_u_star")

#u_rel = fem.Function(V0, name="rel_u")

u_n.x.array[:] = 0.0

#u_rel = u - u_elem
#u_star.x.array[:] = u_star.x.array[:] - u_elem.x.array[:]

u_star_rel = u_star - u_elem

lmbda = ufl.conditional(ufl.gt(ufl.dot(u_star_rel, n_vec), 0), 1, 0)

u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")

term_conv_vol = -rho * ufl.inner(u, ufl.div(ufl.outer(v, u_star_rel))) * dx_fluid
               
term_conv_surf = rho * ufl.inner(ufl.dot(u_star_rel, n_vec)("+") * u_uw, v("+")) * dS_fluid + \
                  rho * ufl.inner(ufl.dot(u_star_rel, n_vec)("-") * u_uw, v("-")) * dS_fluid 

term_conv_bd = rho * ufl.inner(ufl.dot(u_star_rel, n_vec) * lmbda * u, v) * (boundary_ds + ds_outlet)

a_conv = term_conv_vol  + term_conv_surf + term_conv_bd

#-----------------------------------------------------------------

L_conv = - rho * ufl.inner(ufl.dot(u_star_rel, n_vec) * (1 - lmbda) * u_inlet_func, v) * ds_inlet \
         - rho * ufl.inner(ufl.dot(u_star_rel, n_vec) * (1 - lmbda) *  u_elem, v) * ds_interface \
         - rho * ufl.inner(ufl.dot(u_star_rel, n_vec) * (1 - lmbda) * zero_vel_func, v) * (ds_bot + ds_top + ds_cylinder)  

#--------------------------Time Discretization---------------------------

term_time_lhs_bdf1 = rho * ufl.inner(u/dt, v) * ufl.dx 
term_time_rhs_bdf1 = rho * ufl.inner(u_n/dt, v) * ufl.dx 
# Backward Euler

# BDF2: (3u^n+1 - 4u^n + u^{n-1}) / (2*dt)
term_time_lhs_bdf2 = rho * (3.0/ (2.0 * dt)) * ufl.inner(u,v) * ufl.dx 
term_time_rhs_bdf2 = rho * (4.0/ (2.0 * dt)) * ufl.inner(u_n, v) * ufl.dx \
                    - rho * (1.0 / (2.0 * dt)) * ufl.inner(u_nm1, v) * ufl.dx

#-------------------------Form Assembly---------------------------
a_ns_bdf1 = a_stokes  + term_time_lhs_bdf1 + a_conv
L_ns_bdf1 = L_stokes  + term_time_rhs_bdf1 + L_conv

a_ns_bdf2 = a_stokes + a_conv + term_time_lhs_bdf2
L_ns_bdf2 = L_stokes + L_conv + term_time_rhs_bdf2


a_ns_form_bdf1 = fem.form(a_ns_bdf1)
L_ns_form_bdf1 = fem.form(L_ns_bdf1)
a_ns_form_bdf2 = fem.form(a_ns_bdf2)
L_ns_form_bdf2 = fem.form(L_ns_bdf2)

#-----------------------------Solver Setup-----------------------------

up_h = fem.Function(VQ)
u_h, p_h = up_h.sub(0), up_h.sub(1)

def create_solver(a_form):
    A = dolfinx.fem.petsc.create_matrix(a_form)
    ksp = PETSc.KSP().create(fluid_mesh.comm)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setOperators(A)
    pc.getFactorMatrix().setMumpsIcntl(14, 80)
    pc.getFactorMatrix().setMumpsIcntl(24, 1)
    pc.getFactorMatrix().setMumpsIcntl(25, 0)
    return A, ksp

A1, ksp1 = create_solver(a_ns_form_bdf1)
A2, ksp2 = create_solver(a_ns_form_bdf2)

a_form = fem.form(a_ns_bdf1)
L_form = fem.form(L_ns_bdf1)
A,ksp = create_solver(a_form)

################################# SOLID ##################################

V_solid = fem.functionspace(solid_mesh, ("Lagrange", 2, (gdim,)))

d_solid = fem.Function(V_solid, name = "solid_displacement")
v_solid = fem.Function(V_solid, name = "solid_velocity")
a_solid = fem.Function(V_solid, name = "solid_acceleration")

d_solid_old = fem.Function(V_solid)
v_solid_old = fem.Function(V_solid)
a_solid_old = fem.Function(V_solid)

d_solid_test = ufl.TestFunction(V_solid)

E_mod = 1.4e6
nu_val = 0.4

mu_s = fem.Constant(solid_mesh, E_mod / (2 * (1 + nu_val)))
#kappa_s = fem.Constant(solid_mesh, E_mod / (3 * (1 - 2 * nu_val)))
lambda_s = E_mod * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

# Measures
dx_solid = ufl.Measure("dx", domain = solid_mesh)
# dx_solid = ufl.Measure("dx", domain=solid_mesh, 
#                        metadata={"quadrature_degree": 2})
ds_solid = ufl.Measure("ds", domain = solid_mesh, subdomain_data=facet_tags_solid)

# Hyperelastic
I_s = ufl.Identity(gdim)
F_s = I_s + ufl.grad(d_solid)
C_s = F_s.T * F_s
E_s  = 0.5 * (C_s - I_s)

S_s = lambda_s * ufl.tr(E_s) * I_s + 2.0 * mu_s * E_s 
P_s = F_s * S_s 

rho_s = fem.Constant(solid_mesh, PETSc.ScalarType(10000.0))
n_solid = ufl.FacetNormal(solid_mesh)
dt_solid = dt

beta  = fem.Constant(solid_mesh, PETSc.ScalarType(0.25))
gamma = fem.Constant(solid_mesh, PETSc.ScalarType(0.55))

#gamma = fem.Constant(solid_mesh, PETSc.ScalarType(0.6))   # Introduce low-frequency dissipation
beta  = fem.Constant(solid_mesh, PETSc.ScalarType(0.25 * (gamma + 0.5)**2))

n_solid = ufl.FacetNormal(solid_mesh)

d_solid_test = ufl.TestFunction(V_solid)

a_expr_s = (1.0 / (beta * dt_solid**2)) * (d_solid - d_solid_old 
            - dt_solid * v_solid_old - (0.5 - beta) * dt_solid**2 * a_solid_old)

#F_solid = ufl.inner(P_s, ufl.grad(d_solid_test)) * dx_solid 

F_solid = (rho_s * ufl.inner(a_expr_s, d_solid_test) * dx_solid +
           ufl.inner(P_s, ufl.grad(d_solid_test)) * dx_solid)

#----------------------------Force Application----------------------------
I_f = ufl.Identity(gdim)

V_tensor_f = fem.functionspace(fluid_mesh, ("CG", 1, (gdim, gdim)))
sigma_f_exact = -p_h * I_f + nu * (ufl.grad(u_h) + ufl.grad(u_h).T)

u_sig = ufl.TrialFunction(V_tensor_f)
v_sig = ufl.TestFunction(V_tensor_f)

a_sig = ufl.inner(u_sig, v_sig) * dx_fluid
L_sig = ufl.inner(sigma_f_exact, v_sig) * dx_fluid

import dolfinx.fem.petsc
stress_prob = fem.petsc.LinearProblem(a_sig, L_sig, bcs=[], 
                                     petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
                                     petsc_options_prefix="stress_solver")
sigma_f_smooth = stress_prob.solve()

# ====================== Transfer to Solid Side ======================
dx_solid = ufl.Measure("dx", domain=solid_mesh)
ds_solid = ufl.Measure("ds", domain=solid_mesh, subdomain_data=facet_tags_solid)

V_tensor_s = fem.functionspace(solid_mesh, ("CG", 1, (gdim, gdim)))
sigma_s_smooth = fem.Function(V_tensor_s, name="solid_stress_tensor")

# Perform non-matching interpolation
cells_solid = np.arange(solid_mesh.topology.index_map(solid_mesh.topology.dim).size_local, dtype=np.int32)
interp_data = dolfinx.fem.create_interpolation_data(V_tensor_s, V_tensor_f, cells=cells_solid, padding=1e-5)

sigma_s_smooth.interpolate_nonmatching(sigma_f_smooth, cells_solid, interp_data)
sigma_s_smooth.x.scatter_forward()


n_s = ufl.FacetNormal(solid_mesh)
n_f = ufl.FacetNormal(fluid_mesh)
traction_f_expr = -ufl.dot(sigma_f_smooth, n_f)
traction_s_expr = ufl.dot(sigma_s_smooth, n_s)

#------------------------------------------------------------
# J_s = ufl.det(F_s)                                    # Solid deformation gradient determinant
# traction_ref = J_s * ufl.dot(traction_s_expr, ufl.inv(F_s).T)   # First Piola-Kirchhoff traction
# F_solid -= ufl.inner(traction_ref, d_solid_test) * ds_solid(interface_marker)
#_solid -= ufl.inner(traction_s, d_solid_test) * ds_solid(interface_marker)
J_s = ufl.det(F_s)
F_inv_T = ufl.inv(F_s).T
P_fluid = J_s * ufl.dot(sigma_s_smooth, F_inv_T)
traction_ref = ufl.dot(P_fluid, n_s)
F_solid -= ufl.inner(traction_ref, d_solid_test) * ds_solid(interface_marker)
#-----------------------------Dirichlet BC Application-----------------------
# Specifically check the base marker
base_facets = facet_tags_solid.find(flag_base_marker)
print(f"flag_base_marker (value {flag_base_marker}) found {len(base_facets)} facets")
zero = fem.Function(V_solid)
zero.x.array[:] = 0.0
base_dofs = fem.locate_dofs_topological(V_solid, fdim, facet_tags_solid.find(flag_base_marker))
bc_base = fem.dirichletbc(zero, base_dofs)
#--------------------------------------------------------------------
petsc_options = {
    "snes_type": "newtonls", 
    "snes_linesearch_type": "bt",         #bt?
    "snes_stol": 1e-12,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-8,
    "snes_max_it": 20,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,                  
}

solid_problem = NonlinearProblem(
    F_solid, 
    d_solid, 
    bcs=[bc_base],
    petsc_options=petsc_options,
    petsc_options_prefix = "solid_solver"
)

############################### Assistance #####################################

d_interface = fem.Function(V_ale, name = "d_interface_fluid")

cells_fluid = np.arange(
    fluid_mesh.topology.index_map(fluid_mesh.topology.dim).size_local,
    dtype = np.int32)

interp_data_solid_to_ale = dolfinx.fem.create_interpolation_data(
    V_ale,           # target: fluid ALE space
    V_solid,         # source: solid space
    cells=cells_fluid,
    padding=1e-5     
)

#------------------------------------------------------------------

def update_ale_interface_bc():

    d_interface.interpolate_nonmatching(
        d_solid, 
        cells_fluid, 
        interp_data_solid_to_ale
    )
    d_interface.x.scatter_forward()

    interface_dofs = fem.locate_dofs_topological(
        V_ale, fdim, facet_tags_fluid.find(interface_marker)
    )
    
    bc_ale_interface = fem.dirichletbc(d_interface, interface_dofs)
    
    bcs_ale_new = [bc_ale_inlet, bc_ale_top, bc_ale_bottom, bc_ale_outlet, bc_ale_cylinder, bc_ale_interface]
    
    return bcs_ale_new

#------------------------------------------------------------------
V0_fluid = V0 #fem.functionspace(fluid_mesh, ("Lagrange", 1, (gdim,)))
#u_elem = fem.Function(V0_fluid, name="mesh_velocity")   # u_mesh = (d - d_old)/dt
d_elem = fem.Function(fem.functionspace(fluid_mesh, ("Lagrange", 1, (gdim,))))
def project_mesh_velocity():
    u_expr = (d - d_old) / float(dt.value)
    trial = ufl.TrialFunction(V0_fluid)
    test = ufl.TestFunction(V0_fluid)

    dx_proj = ufl.Measure("dx", domain = fluid_mesh)

    a_proj = ufl.inner(trial, test) * dx_proj
    L_proj = ufl.inner(u_expr, test) * dx_proj
    proj_problem = fem.petsc.LinearProblem(
        a_proj, 
        L_proj, 
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix = "ale_velocity_projector")
    return proj_problem.solve()

#----------------------------------------------------------------

folder = Path("results_FSI_submesh")
folder.mkdir(exist_ok=True, parents=True)

V_u_vis = fem.functionspace(fluid_mesh, ("CG", 2, (gdim,)))
V_p_vis = fem.functionspace(fluid_mesh, ("CG", 1))
V_d_vis = fem.functionspace(fluid_mesh, ("CG", 2, (gdim,)))
V_solid_vis = fem.functionspace(solid_mesh, ("CG", 2, (gdim,)))
Vf_fluid_vis = fem.functionspace(fluid_mesh, ("CG", 1, (gdim,)))
Vf_solid_vis = fem.functionspace(solid_mesh, ("CG", 1, (gdim,)))

u_vis = fem.Function(V_u_vis, name="velocity")
p_vis = fem.Function(V_p_vis, name="pressure")
d_vis = fem.Function(V_d_vis, name="ALE_displacement")
d_solid_vis = fem.Function(V_solid_vis, name="solid_displacement")
T_fluid_vis = fem.Function(Vf_fluid_vis, name="traction_f")
T_solid_vis = fem.Function(Vf_solid_vis, name="traction_s")

vtx_u = io.VTXWriter(fluid_mesh.comm, folder / "u.bp", u_vis)
vtx_p = io.VTXWriter(fluid_mesh.comm, folder / "p.bp", p_vis)
vtx_combined = io.VTXWriter(fluid_mesh.comm, folder / "fluid_ale.bp", [u_vis, d_vis], engine="BP4")
vtx_solid = io.VTXWriter(solid_mesh.comm, folder / "solid.bp", d_solid_vis)
vtx_T_f = io.VTXWriter(fluid_mesh.comm, folder / "T_f.bp", T_fluid_vis)
vtx_T_s = io.VTXWriter(solid_mesh.comm, folder / "T_s.bp", T_solid_vis)

#-------------------------History ux uy-------------------------
from dolfinx import geometry
tip_x = xc + r + fl
tip_y = yc 
tip_point = np.array([[tip_x, tip_y, 0.0]], dtype=PETSc.ScalarType) # Note 3D coordinates
tdim = solid_mesh.topology.dim
bb_tree = geometry.bb_tree(solid_mesh, tdim)

cell_tree = geometry.bb_tree(solid_mesh, tdim)
midpoint_tree = geometry.bb_tree(solid_mesh, tdim)
tip_cell = geometry.compute_closest_entity(cell_tree, midpoint_tree, solid_mesh, tip_point)

solid_tip_history = []
interface_force = []
flag_force = []

inertia_force_vector = rho_s * a_expr_s
external_force_vector = traction_ref
inertia_mag_form = fem.form(ufl.sqrt(ufl.inner(inertia_force_vector, inertia_force_vector)) * dx_solid)
external_mag_form = fem.form(ufl.sqrt(ufl.inner(external_force_vector, external_force_vector)) * ds_solid(interface_marker))
inertia_ratio_history = []

################################### Time Loop ###############################
max_fsi_iter = 50
fsi_tol = 1e-6

inlet_vel = InletVelocity()          # Instance with .t attribute
x_ref_fluid = fluid_mesh.geometry.x.copy()
u_n.interpolate(fem.Function(V0))
d_old.x.array[:] = 0.0
d_solid_old.x.array[:] = 0.0

t = 0.0
step = 0
dt.value = 0.05
dt_val = float(dt.value)
#T = dt_val * 50 #40 is the end of the ramp
T = 15.0

while t < T:
    
    inlet_vel.t = float(t)                    # inlet_vel is the instance created below
    u_inlet_func.interpolate(inlet_vel)
    #print(step,t)
    if step == 0:
        a_form = a_ns_form_bdf1
        L_form = L_ns_form_bdf1
        A, ksp = A1, ksp1
    else:
        a_form = a_ns_form_bdf2
        L_form = L_ns_form_bdf2
        A, ksp = A2, ksp2
    
    A.zeroEntries()
    assemble_matrix(A, a_form)
    A.assemble()
    ksp.setOperators(A)

    b = dolfinx.fem.assemble_vector(L_form)

    ksp.solve(b.petsc_vec, up_h.x.petsc_vec)
    up_h.x.scatter_forward()

    #********************************************************************
    if t >= 2.0:
        
        # change of the timestep has impact on the stability keep same will be better
        if t < 5.0: 
            dt.value = 0.02
        elif t < 7.5:
            dt.value = 0.01
        else:
            dt.value = 0.005

        aitken_residual_old = None 
        aitken_omega = 0.5
        d_interface_old = d_interface.x.array.copy()
        d_solid_prev_iter = d_solid_old.x.array.copy()

        #---------------------fsi  Aitken iter for the add_mass-------------------
        for fsi_iter in range(max_fsi_iter):
            print(f"  FSI iter {fsi_iter+1}/{max_fsi_iter} at t={t:.4f}")

            sigma_f_smooth = stress_prob.solve()
            sigma_s_smooth.interpolate_nonmatching(sigma_f_smooth, cells_solid, interp_data)
            sigma_s_smooth.x.scatter_forward()

        #-------------------------------------------------------------------------
            solid_nit = solid_problem.solve()
            d_solid.x.scatter_forward()
            d_solid_solver = d_solid.x.array.copy()

       
            residual = d_solid_solver - d_solid_prev_iter
            if fsi_iter == 0:
                d_solid.x.array[:] = aitken_omega * d_solid_solver + (1.0 - aitken_omega) * d_solid_prev_iter 
                d_solid_prev_iter = d_solid.x.array.copy()
                aitken_residual_old = residual.copy()
            else:
                residual_old = aitken_residual_old
                delta_r = residual - residual_old 
                norm_delta_r = np.dot(delta_r, delta_r)

                if norm_delta_r > 1e-14:
                    aitken_omega *= -np.dot(residual_old, delta_r) / norm_delta_r 
                    aitken_omega = max(0.1, min(aitken_omega, 1.0))

                else:
                    aitken_omega = 0.5

                d_solid.x.array[:] = d_solid_prev_iter + aitken_omega * residual
                aitken_residual_old = residual.copy() 

            d_solid_prev_iter = d_solid.x.array.copy()

            # ALE mesh motion
            fluid_mesh.geometry.x[:] = x_ref_fluid
            bcs_ale_new = update_ale_interface_bc()
            ALE_problem = NonlinearProblem(
                F_ale, 
                d, 
                bcs=bcs_ale_new,
                petsc_options=petsc_options_ALE,
                petsc_options_prefix="ALE_solver"
            )
            ale_nit = ALE_problem.solve()
            d.x.scatter_forward()

            d_elem.interpolate(d)
            new_u_elem = project_mesh_velocity()
            u_elem.x.array[:] = new_u_elem.x.array[:]
            u_elem.x.scatter_forward()

            fluid_mesh.geometry.x[:, :gdim] = x_ref_fluid[:, :gdim] + d_elem.x.array.reshape((-1, gdim))
        #-----------------------------------------------

            A.zeroEntries()
            assemble_matrix(A, a_form)
            A.assemble()
            ksp.setOperators(A)
            b = dolfinx.fem.assemble_vector(L_form)
            ksp.solve(b.petsc_vec, up_h.x.petsc_vec)
            up_h.x.scatter_forward()

            #---------------------------------------------------------------------
            traction_f_expr_step = -ufl.dot(sigma_f_smooth, n_f)
            traction_s_expr_step = ufl.dot(sigma_s_smooth, n_s)
            total_force_f_x = fem.assemble_scalar(fem.form(
                ufl.inner(traction_f_expr_step, e_x) * ds_fluid(interface_marker)
            ))
            total_force_f_y = fem.assemble_scalar(fem.form(
                ufl.inner(traction_f_expr_step, e_y) * ds_fluid(interface_marker)
            ))

            total_force_s_x = fem.assemble_scalar(fem.form(
                ufl.inner(traction_s_expr_step, e_x) * ds_solid(interface_marker)
            ))
            total_force_s_y = fem.assemble_scalar(fem.form(
                ufl.inner(traction_s_expr_step, e_y) * ds_solid(interface_marker)
            ))

            force_res = np.sqrt(
                (total_force_f_x - total_force_s_x)**2 +
                (total_force_f_y - total_force_s_y)**2
            )
            #--------------------------------------------------------------------
            # Normalization (optional)
            norm_force = np.sqrt(total_force_f_x**2 + total_force_f_y**2) + 1e-10
            rel_force_res = force_res / norm_force

            # Convergence check
            d_diff = np.linalg.norm(d_interface.x.array - d_interface_old) / (np.linalg.norm(d_interface.x.array) + 1e-12)
            print(f"    interface disp change = {d_diff:.2e}")
            print(f"    force change = {rel_force_res:.2e}")
            if d_diff < fsi_tol:
                print(f"    FSI converged in {fsi_iter+1} iterations at t={t:.4f}")
                break

            d_interface_old[:] = d_interface.x.array.copy()
        
        # Newmark time integration (velocity / acceleration update)
        a_new = (1.0 / (beta * dt_solid**2)) * (d_solid.x.array - d_solid_old.x.array 
                - float(dt_solid) * v_solid_old.x.array 
                - (0.5 - beta) * float(dt_solid)**2 * a_solid_old.x.array)
        a_solid.x.array[:] = a_new

        v_new = v_solid_old.x.array + float(dt_solid) * ((1 - gamma) * a_solid_old.x.array 
                + gamma * a_solid.x.array)
        v_solid.x.array[:] = v_new


    u_nm1.x.array[:] = u_n.x.array[:] 
    u_n.interpolate(u_h)    
    
    if step >= 1:
        u_star.x.array[:] = 2.0 * u_n.x.array[:] - u_nm1.x.array[:]
    else:
        u_star.x.array[:] = u_n.x.array[:]  

    d_old.x.array[:] = d.x.array[:]
    d_solid_old.x.array[:] = d_solid.x.array.copy()
    v_solid_old.x.array[:] = v_solid.x.array.copy()
    a_solid_old.x.array[:] = a_solid.x.array.copy()

    fluid_mesh.geometry.x[:, :gdim] = x_ref_fluid[:, :gdim] + d_elem.x.array.reshape((-1, gdim))

    # Record flag tip displacement (as in original FSI2)
    tip_disp = d_solid.eval(tip_point, tip_cell)
    ux_tip = tip_disp[0]
    uy_tip = tip_disp[1]
    solid_tip_history.append([float(t), ux_tip, uy_tip])
    print(f"Flag tip t = {t:.4f} | ux = {ux_tip:.6e} uy = {uy_tip:.6e}")

    #--------------------------------------------------------------------------------
    #------------------------History FD, FL-----------------------------
    ds_obstacle = ds_fluid(interface_marker) + ds_fluid(cylinder_marker)

    I_f = ufl.Identity(gdim)
    sigma_f = -p_h * I_f + nu * (ufl.grad(u_h) + ufl.grad(u_h).T)
    n_f = ufl.FacetNormal(fluid_mesh)
    force_on_body = - sigma_f * n_f

    # Total force (unit: N)
    drag_form = ufl.inner(- sigma_f * n_f, ufl.as_vector([1.0, 0.0])) * ds_obstacle
    lift_form = ufl.inner(- sigma_f * n_f, ufl.as_vector([0.0, 1.0])) * ds_obstacle

    F_D = dolfinx.fem.assemble_scalar(dolfinx.fem.form(drag_form))
    F_L = dolfinx.fem.assemble_scalar(dolfinx.fem.form(lift_form))


    drag_form1 = ufl.inner(- sigma_f * n_f, ufl.as_vector([1.0, 0.0])) * ds_fluid(interface_marker) 
    lift_form1= ufl.inner(- sigma_f * n_f, ufl.as_vector([0.0, 1.0])) * ds_fluid(interface_marker) 

    F_D1 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(drag_form1))
    F_L1 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(lift_form1))

    e_x = ufl.as_vector([1.0, 0.0])
    e_y = ufl.as_vector([0.0, 1.0])

    traction_f_expr_step = -ufl.dot(sigma_f_smooth, n_f)
    traction_s_expr_step = ufl.dot(sigma_s_smooth, n_s)

    total_force_f_x = fem.assemble_scalar(fem.form(
        ufl.inner(traction_f_expr_step, e_x) * ds_fluid(interface_marker)
    ))
    total_force_f_y = fem.assemble_scalar(fem.form(
        ufl.inner(traction_f_expr_step, e_y) * ds_fluid(interface_marker)
    ))

    total_force_s_x = fem.assemble_scalar(fem.form(
        ufl.inner(traction_s_expr_step, e_x) * ds_solid(interface_marker)
    ))
    total_force_s_y = fem.assemble_scalar(fem.form(
        ufl.inner(traction_s_expr_step, e_y) * ds_solid(interface_marker)
    ))
    total_force_f = fem.assemble_scalar(fem.form(
        ufl.inner(traction_f_expr_step, n_f) * ds_fluid(interface_marker)
    ))
    total_force_s = fem.assemble_scalar(fem.form(
        ufl.inner(traction_s_expr_step, n_s) * ds_solid(interface_marker)
    ))
    #--------------------------------------------------------------------------------
    
    interface_force.append([float(t), F_D, F_L, F_D1, F_L1])
    print(f"FD = {F_D:.6e} | FL = {F_L:.6e} |FD1 = {F_D1:.6e} | FL1 = {F_L1:.6e} ")

    flag_force.append([float(t), total_force_f_x, total_force_f_y, total_force_s_x, total_force_s_y])

    print(f"total_force_f = {total_force_f:.6e} | total_force_s = {total_force_s:.6e} ")
    print(f"Interface force (fluid)  Fx={total_force_f_x:.4f}  Fy={total_force_f_y:.4f}")
    print(f"Interface force (solid)  Fx={total_force_s_x:.4f}  Fy={total_force_s_y:.4f}")

    mag_inertia = solid_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(inertia_mag_form), op=MPI.SUM)
    mag_external = solid_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(external_mag_form), op=MPI.SUM)
    if mag_external > 1e-12:
        inertia_ratio = mag_inertia / mag_external
    else:
        inertia_ratio = 0.0
    print(f"intertia_ratio = {inertia_ratio}")

    step += 1
    t += float(dt.value)
    print(step, t)

    history_array = np.array(solid_tip_history)
    inertia_ratio_history.append([float(t), mag_inertia, mag_external, inertia_ratio])


    np.savetxt(folder / "solid_tip_history.txt", 
               history_array, 
               header="time ux uy", 
               comments='', 
               fmt='%.8f')
    np.savetxt(folder / "interface_history.txt", 
               interface_force, 
               header="time F_D F_L F_D1 F_L1", 
               comments='', 
               fmt='%.8f')
    np.savetxt(folder / "flagforce_history.txt", 
               flag_force, 
               header="time total_force_f_x total_force_f_y total_force_s_x total_force_s_y", 
               comments='', 
               fmt='%.8f')
    np.savetxt(folder / "inertia_ratio_history.txt", 
               np.array(inertia_ratio_history), 
               header="time mag_inertia mag_external ratio", 
               comments='', 
               fmt='%.8f')
    
    #--------------------Output--------------------------
    if step % 5 == 0 or step == 1:
        u_vis.interpolate(u_h.collapse())
        p_vis.interpolate(p_h.collapse())
        d_vis.interpolate(d)
        d_solid_vis.interpolate(d_solid)
        #T_fluid_vis.interpolate(traction_f_smooth)
        #T_solid_vis.interpolate(traction_s)
        vtx_u.write(t)
        vtx_p.write(t)
        vtx_solid.write(t)
        vtx_combined.write(t)
        vtx_T_f.write(t)
        vtx_T_s.write(t)

#-------------------------------Static Recheck----------------------------------

# traction_f_smooth = stress_prob.solve()
# traction_s.interpolate_nonmatching(traction_f_smooth, cells_solid, interp_data)
# traction_s.x.scatter_forward()

# # Static equilibrium residual (hyperelastic internal force balances traction)
# # → NO inertia term, NO Newmark acceleration/velocity
# d_solid_test_static = ufl.TestFunction(V_solid)

# F_solid_static = (total_force_f_x = fem.assemble_scalar(fem.form(
#     ufl.inner(traction_f_smooth, e_x) * ds_fluid(interface_marker)
# ))
# total_force_f_y = fem.assemble_scalar(fem.form(
#     ufl.inner(traction_f_smooth, e_y) * ds_fluid(interface_marker)
# ))
#     ufl.inner(P_s, ufl.grad(d_solid_test_static)) * dx_solid
#     - ufl.inner(traction_ref, d_solid_test_static) * ds_solid(interface_marker)
# )

# # Create dedicated nonlinear problem
# static_problem = NonlinearProblem(
#     F_solid_static,
#     d_solid,
#     bcs=[bc_base],
#     petsc_options=petsc_options,
#     petsc_options_prefix="static_solid_solver"
# )

# # Solve
# static_nit = static_problem.solve()
# d_solid.x.scatter_forward()

# print(f"✓ Static solid solve converged in {static_nit} Newton iterations.")

# # Print tip displacement (no files saved)
# tip_disp_static = d_solid.eval(tip_point, tip_cell)
# ux_static = tip_disp_static[0]
# uy_static = tip_disp_static[1]

# print(f"Static flag tip displacement → ux = {ux_static:.8e} m")
# print(f"Static flag tip displacement → uy = {uy_static:.8e} m")
# print("="*80)
#-------------------------------------------History and Plot--------------------------------------------
# if len(solid_tip_history) > 0:

#     history_array = np.array(solid_tip_history)
#     np.savetxt(folder / "solid_tip_history.txt", 
#                history_array, 
#                header="time ux uy", 
#                comments='', 
#                fmt='%.8f')

#     t_hist = history_array[:, 0]
#     ux_hist = history_array[:, 1]
#     uy_hist = history_array[:, 2]

#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(12, 8))

#     plt.subplot(2, 1, 1)
#     plt.plot(t_hist, ux_hist*1000, 'b-', linewidth=2, label='ux (mm)')
#     plt.ylabel('Tip Displacement ux [mm]')
#     plt.grid(True, alpha=0.5)
#     plt.legend()
#     plt.title('FSI1 - Flag Tip Displacement History (x-direction)')

#     plt.subplot(2, 1, 2)
#     plt.plot(t_hist, uy_hist*1000, 'r-', linewidth=2, label='uy (mm)')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Tip Displacement uy [mm]')
#     plt.grid(True, alpha=0.5)
#     plt.legend()

#     plt.suptitle('Turek & Hron FSI1 - Flag Tip Displacement over Time', fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])

#     plt.savefig(folder / 'tip_displacement_history.png', dpi=300, bbox_inches='tight')
#     plt.show()