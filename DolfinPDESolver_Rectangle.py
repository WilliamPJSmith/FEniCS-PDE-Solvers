"""
DolfinumpyDESolver_Rectangle.py
==================

A python class structure written to interface CellModeller4 with the FEniCs/Dolfin finite element library.
This is a analogue of the original DolfinumpyDESolver class, this time designed for radial simulations.
Key changes:
	- Meshes are instances of the FEniCS CircleMesh class
	- Elementwise computations are adapted to the 2D unstructured mesh type
	- Some of the clumsy features of the cuboidal moving boundary meshes can be removed, since we've dropped the restriction to canonical elements
	
At some point, it'd be useful to unite the various Solver class versions into one all-purpose system.

Created: W. P. J. Smith, 19.08.15
"""

try:
    from dolfin import *
except ImportError:
    print "Error: could not import dolfin library."
    print "Try calling $ source /Applications/FEniCS.app/Contents/Resources/share/fenics/TestFenicsPath.conf "
import numpy
import math
from pyopencl.array import vec
import time

class RectangleDolfinSolver:

	def __init__(self, solverParams):
		""" 
		Initialise the dolfin solver using a dictionary of params.
		"""
		# extract fixed params from params dictionary
		self.pickleSteps = solverParams['pickleSteps']
		self.h = solverParams['h']
		self.origin = solverParams['origin']
		self.N_x = int(solverParams['N_x'])
		self.L_x = solverParams['L_x']
		self.u0 = solverParams['u0']
		self.K = solverParams['K']
		self.mu_eff = solverParams['mu_eff'] 
		self.rel_tol = solverParams['rel_tol']
		self.mesh_type = solverParams['mesh_type']
		self.delta = solverParams['delta']

		# some attributes that we'll update on the fly: set them to None for now
		self.boundaryCondition = None
		self.mesh = None
		self.V = None
		self.solution = None
		
		
	def SolvePDE(self, centers, areas, filename, dir, stepNum=0):
		"""
		High-level function to be called during the function.
		"""
		global L_y
		global L_x
		
		L_x = self.L_x
		
		#get height of highest cell in domain
		max_height = 0.0
		for center in centers:
			hy = center[1] 
			if hy > max_height:
				max_height = hy
		print 'max height is %f' % max_height
		L_y = max_height + self.delta

		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(L_y)
		self.V = FunctionSpace(self.mesh, "CG", 1)
		
		# choose boundary conditions
		self.set_bcs()
		#self.set_bcs_surround()
		#self.set_bcs_reverse()
		
		# Use cell centres to evaluate volume occupancy of mesh
		self.GetVolFracsCrossMesh(centers, areas)
		
		# call a solver and save the solution
		self.NewtonIterator()
		if stepNum % self.pickleSteps == 0:
			self.WriteFieldToFile(dir+filename+'.pvd', self.solution)
		
		# interpolate solution to cell centres
		u_local = self.InterpolateToCenters2D(centers)
		
		return u_local
		
		
	def SolvePDE_x_periodic(self, centers, areas, filename, dir, stepNum=0):
		"""
		High-level function to be called during the function.
		/!\ Exports both L_x and L_y as global variables.
		"""
		global L_y
		global L_x
		
		# get height of highest cell in domain
		max_height = 0.0
		for center in centers:
			hy = center[1] 
			if hy > max_height:
				max_height = hy
		print 'max height is %f' % max_height
		
		# global variables
		L_y = max_height + self.delta 	# needed for top BC
		L_x = self.L_x				  	# needed for periodic BC

		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(L_y)
		
		# constrain domain to be x-periodic
		pbc = XPeriodicDomain()
		self.V = FunctionSpace(self.mesh, "CG", 1, constrained_domain=pbc)
		
		# set bcs
		self.set_bcs()
		#self.set_bcs_reverse()
		
		# Use cell centres to evaluate volume occupancy of mesh
		self.GetVolFracsCrossMesh(centers, areas)
		
		# call a solver and save the solution
		self.NewtonIterator()
		if stepNum % self.pickleSteps == 0:
			self.WriteFieldToFile(dir+filename+'.pvd', self.solution)
		
		# interpolate solution to cell centres
		u_local = self.InterpolateToCenters2D(centers)
		
		return u_local
		
	
	def NewtonIterator(self):
		"""
		A Newton iterator for solving non-linear problems.
		/!\ Assumes that function space (V), boundaryCondition, vol_fracs are up-to-date.
		"""
		# Define variational problem
		u = Function(self.V, name = "Nutrient")
		v = TestFunction(self.V)
		F = dot(grad(u), grad(v))*dx - self.MonodNutrientSink(u)*v*dx

		# Call built-in Newton solver
		#set_log_level(PROGRESS) # very detailed info for testing
		set_log_level(WARNING) # near-silent info for simulations
		
		solve(F == 0, u, self.boundaryCondition, solver_parameters = {"newton_solver":
																	 {"relative_tolerance": self.rel_tol}})															 
		self.solution = u	
		
		
	def InterpolateToCenters2D(self, centers):
		"""
		Interpolate a solution object u onto a list of cell coordinates
		"""
		u = self.solution
		data_t = tuple(map(tuple, centers)) 	   		     # Convert to tuple format
		u_local = numpy.zeros((len(data_t),),numpy.float64)  # preallocate solution array
		for i in range(0,len(data_t)):				  		 # loop over all cells
			u_local[i] = numpy.max(u(data_t[i][0:2]), 0.0)					 # extrapolate solution value at cell centre
		return u_local												 		
		
		
	def SetRectangularMesh(self, L_y):
		"""
		Given a disk radius and center, return an unstructured mesh on that dish
		Uses h as the mesh element size parameter. Center must be a Dolfin Point() instance.
		/!\ Exports N_y as a global variable
		"""
		global N_y
		
		mesh_type = self.mesh_type
		N_y = int(round(L_y/self.h))
		self.mesh = RectangleMesh(0, 0, self.L_x, L_y, self.N_x, N_y, mesh_type)
		
		
	def set_bcs(self):
		"""
		Initialise boundary conditions on the mesh.
		/!\ Assumes that the function V is up-to-date
		"""
		dbc = TopDirichletBoundary()
		self.boundaryCondition = DirichletBC(self.V, Constant(self.u0), dbc)
		
		
	def set_bcs_reverse(self):
		"""
		Initialise boundary conditions on the mesh.
		/!\ Assumes that the function V is up-to-date
		"""
		tbc = TopDirichletBoundary()
		bbc = BaseDirichletBoundary()
		bc0 = DirichletBC(self.V, Constant(self.u0), bbc)
		bc1 = DirichletBC(self.V, Constant(0.0), tbc)
		bcs = [bc0, bc1]
		
		self.boundaryCondition = bcs
		
	def set_bcs_surround(self):
		"""
		Initialise boundary conditions on the mesh.
		/!\ Assumes that the function V is up-to-date
		"""
		tbc = TopDirichletBoundary()
		bbc = BaseDirichletBoundary()
		lbc = LeftDirichletBoundary()
		rbc = RightDirichletBoundary()
		
		bc0 = DirichletBC(self.V, Constant(self.u0), bbc)
		bc1 = DirichletBC(self.V, Constant(self.u0), tbc)
		bc2 = DirichletBC(self.V, Constant(self.u0), lbc)
		bc3 = DirichletBC(self.V, Constant(self.u0), rbc)
		
		bcs = [bc0, bc1, bc2, bc3]

		self.boundaryCondition = bcs		
		
		
		
	def AssignElementsToData2D(self, centers):
		"""
		Modified cell assignment function using the mesh cell 'collides' method.
		Unlike ASSIGNELEMENTSTODATA2D, this will work on unstructured [2D] meshes.
		"""
		N = centers.shape[0]
		elements = numpy.zeros((N), numpy.int32)
	
		for i in range(0,N):
			for j in range(0,self.mesh.num_cells()):
				if Cell(self.mesh, j).collides(Point(float(centers[i][0]),\
													 float(centers[i][1]),\
													 0)):
					elements[i] = j
					break
		return elements
		
		
	def WriteFieldToFile(self, filename, u):
		"""
		Export the PDE solution as a pvd mesh.
		"""
		print "Writing fields..."
		File(filename) << u
		print 'Done.'


	def MonodNutrientSink(self, u):
		""" 
		Monod function with which to build RHS.
		"""
		a = Constant(self.mu_eff)
		b = Constant(self.K)
		
		return -1 * a * u * VolumeFraction() / (b + u)
		
		
	def GetTriangleIndexCrossMesh(self, Point, Origin):
		"""
		Given mesh square, assign which triangle a point is in.
		/!\ Assumes that we're using a crossed rectilinear mesh.
		/!\ Assumes L_y and N_y are supplied as global variables.
		"""
		global L_y, N_y
		
		p_x = Point[0]; p_y = Point[1]
		a_x = Origin[0]; a_y = Origin[1]
		dx = p_x - a_x
		dy = p_y - a_y
		hx = self.L_x / self.N_x
		hy = L_y / N_y
		gr = hy / hx; 
		
		return 1*(dy-gr*dx>0) + 2*(dy+gr*dx-hy>0);


	def GetSquareIndex(self, Point):
		"""
		Given mesh dimensions, assign which square a point is in.
		/!\ Assumes L_y and N_y are supplied as global variables.
		"""
		global L_y, N_y
		
		p_x = Point[0]; p_y = Point[1]
		p = int(numpy.floor(p_x*self.N_x / float(self.L_x)))  		# index along x
		q = int(numpy.floor(p_y*N_y / float(L_y)))   		# index along y

		s = p + q*self.N_x 											# global index of this square
		sqOrigin = [p*self.L_x / float(self.N_x),\
					q*L_y / float(N_y)]						# coordinates of this square's origin			    
		
		return int(s), sqOrigin 


	def GetElementIndexCrossMesh(self, point):
		"""
		Get tetrahedron and cube indices and calculate global element index.
		"""
		[s, sqOrigin] = self.GetSquareIndex(point)
		t = self.GetTriangleIndexCrossMesh(point,sqOrigin)
		
		return t + 4*s

	
	def AssignElementsToDataCrossMesh(self, centers):
		"""
		Sort cell centres into their respective mesh elements.
		"""
		N = centers.shape[0]
		elements = numpy.zeros((N), numpy.int32)
		for i in range(0,N):
			point = centers[i]
			elements[i] = self.GetElementIndexCrossMesh(point)
			
		return elements
		
		
	def GetVolFracsCrossMesh(self, centers, vols):
		"""
		Create a global list of the cell volume fractions in mesh elements.
		Assumes that self.mesh is up to date.
		'Volumes' are equivalent to areas in 2D.
		/!\ Exports the array vol_fracs as a global array, for use by VolumeFraction.
		/!\ Takes 
		"""
		global vol_fracs, L_y, N_y

		# assign elements of cells
		elements = self.AssignElementsToDataCrossMesh(centers)
		
		# need to define volume fraction for every element in the mesh
		# (not just the canonical elements for counting)
		num_elements = self.mesh.num_cells()

		# sum cell volumes over each element
		a = (self.L_x*L_y) / (4.0*float(self.N_x)*float(N_y))
		vol_fracs = numpy.bincount(elements,vols,num_elements) / a
			
			
	def GetVolFracs2D(self, centers, vols):
		"""
		Create a global list of the cell volume fractions in mesh elements.
		Assumes that self.mesh and self.h are up-to-date.
		'Volumes' are equivalent to areas in 2D.
		NOTE: this is pretty slow for >1000 cells, so I've written GetVolFracsCrossMechs as a faster alternative.
		/!\ Exports the array vol_fracs as a global array, for use by VolumeFraction.
		"""
		global vol_fracs

		# assign elements of cells
		elements = self.AssignElementsToData2D(centers)
	
		# compute element volumes
		num_elements = self.mesh.num_cells()
		element_vols = numpy.zeros((num_elements,)) 
		for i in range(0,num_elements):
			element_vols[i] = Cell(self.mesh,i).volume()
		
		# Here, we need to:
		#  >  define volume fraction for every element in the mesh
		#  >  normalise volume totals by mesh element volumes
		binned_vol_fracs = numpy.bincount(elements,vols,num_elements)
		vol_fracs = numpy.divide(binned_vol_fracs, element_vols)
	
	
	def VolumeFractionOnElements(self):
		""" 
		Monod function with which to build RHS.
		"""
		return VolumeFraction()
		
		
	def TestProblem_A(self, dir, filename, height):
		"""
		Solves the homogenised reaction-diffusion problem on a standard mesh. 
		Imaginary cells are placed at the centroids of each element, so that vol_fracs should evaluate to 1 everywhere.
		You can check this by eye, since we export the volume fraction function too.
		This is meant to test the main SolvePDE method, so keep the two similar!
		"""
		global L_y
		
		L_y = height
		
		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(height)	
		self.V = FunctionSpace(self.mesh, "CG", 1)
		self.set_bcs()	

		# create an imaginary cell filling each element
		N = self.mesh.num_cells() 
		centers = numpy.zeros((N,), vec.float4) 
		areas = numpy.zeros((N,)) 
		for cell_no in range(N):
			thisEl = Cell(self.mesh, cell_no)
			centers[cell_no][0] = thisEl.midpoint().x()
			centers[cell_no][1] = thisEl.midpoint().y()
			centers[cell_no][2] = thisEl.midpoint().z()
			areas[cell_no] = thisEl.volume()

		# compute area fractions (should evaluate to 1.0 everywhere)
		#self.GetVolFracs2D(centers, areas)  # Too slow
		self.GetVolFracsCrossMesh(centers, areas)
				
		# Export a meshfile showing element occupancies
		G = self.VolumeFractionOnElements()
		g = Function(self.V, name = "Area fraction")
		g.interpolate(G)
		self.WriteFieldToFile(dir+filename+'_VolFracs'+'.pvd', g)
		
		# call a solver and save the solution
		set_log_level(PROGRESS)
		self.NewtonIterator()
		
		# save the solution so we can inspect it elsewhere
		self.WriteFieldToFile(dir+filename+'_Solution'+'.pvd', self.solution)

	def TestProblem_A_periodic(self, dir, filename, height):
		"""
		Solves the homogenised reaction-diffusion problem on a standard mesh. 
		Imaginary cells are placed at the centroids of each element, so that vol_fracs should evaluate to 1 everywhere.
		You can check this by eye, since we export the volume fraction function too.
		This is meant to test the main SolvePDE method, so keep the two similar!
		"""
		global L_y
		global L_x
		
		L_y = height
		L_x = self.L_x
		
		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(L_y)
		
		# constrain domain to be x-periodic
		pbc = XPeriodicDomain()
		self.V = FunctionSpace(self.mesh, "CG", 1, constrained_domain=pbc)
		self.set_bcs()	

		# create an imaginary cell filling each element
		N = self.mesh.num_cells() 
		centers = numpy.zeros((N,), vec.float4) 
		areas = numpy.zeros((N,)) 
		for cell_no in range(N):
			thisEl = Cell(self.mesh, cell_no)
			centers[cell_no][0] = thisEl.midpoint().x()
			centers[cell_no][1] = thisEl.midpoint().y()
			centers[cell_no][2] = thisEl.midpoint().z()
			areas[cell_no] = thisEl.volume()

		# compute area fractions (should evaluate to 1.0 everywhere)
		#self.GetVolFracs2D(centers, areas)  # Too slow
		self.GetVolFracsCrossMesh(centers, areas)
				
		# Export a meshfile showing element occupancies
		G = self.VolumeFractionOnElements()
		g = Function(self.V, name = "Area fraction")
		g.interpolate(G)
		self.WriteFieldToFile(dir+filename+'_VolFracs'+'.pvd', g)
		
		# call a solver and save the solution
		set_log_level(PROGRESS)
		self.NewtonIterator()
		
		# save the solution so we can inspect it elsewhere
		self.WriteFieldToFile(dir+filename+'_Solution'+'.pvd', self.solution)


	def TestProblem_B(self, dir, filename, height):
		"""
		Solves the homogenised reaction-diffusion problem on a standard mesh. 
		Imaginary cells are placed at the centroids of each element, so that vol_fracs should evaluate to 1 everywhere.
		You can check this by eye, since we export the volume fraction function too.
		This is meant to test the main SolvePDE method, so keep the two similar!
		"""
		global L_y
		
		L_y = height
		
		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(height)	
		self.V = FunctionSpace(self.mesh, "CG", 1)
		self.set_bcs()	

		# create an imaginary cell filling the first 50% of elements (assuming regular indexing from bottom)
		N = int(0.5 * self.mesh.num_cells())
		centers = numpy.zeros((N,), vec.float4) 
		areas = numpy.zeros((N,)) 
		for cell_no in range(N):
			thisEl = Cell(self.mesh, cell_no)
			centers[cell_no][0] = thisEl.midpoint().x()
			centers[cell_no][1] = thisEl.midpoint().y()
			centers[cell_no][2] = thisEl.midpoint().z()
			areas[cell_no] = thisEl.volume()

		# compute area fractions (should evaluate to 1.0 everywhere)
		#self.GetVolFracs2D(centers, areas)  # Too slow
		self.GetVolFracsCrossMesh(centers, areas)
				
		# Export a meshfile showing element occupancies
		G = self.VolumeFractionOnElements()
		g = Function(self.V, name = "Area fraction")
		g.interpolate(G)
		self.WriteFieldToFile(dir+filename+'_VolFracs'+'.pvd', g)
		
		# call a solver and save the solution
		set_log_level(PROGRESS)
		self.NewtonIterator()
		
		# save the solution so we can inspect it elsewhere
		self.WriteFieldToFile(dir+filename+'_Solution'+'.pvd', self.solution)
		
	def TestProblem_B_periodic(self, dir, filename, height):
		"""
		Solves the homogenised reaction-diffusion problem on a standard mesh. 
		Imaginary cells are placed at the centroids of each element, so that vol_fracs should evaluate to 1 everywhere.
		You can check this by eye, since we export the volume fraction function too.
		This is meant to test the main SolvePDE method, so keep the two similar!
		"""
		global L_y
		global L_x
		
		L_y = height
		L_x = self.L_x
		
		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(height)	

		# constrain domain to be x-periodic
		pbc = XPeriodicDomain()
		self.V = FunctionSpace(self.mesh, "CG", 1, constrained_domain=pbc)
		self.set_bcs()	

		# create an imaginary cell filling the first 50% of elements (assuming regular indexing from bottom)
		N = int(0.5 * self.mesh.num_cells())
		centers = numpy.zeros((N,), vec.float4) 
		areas = numpy.zeros((N,)) 
		for cell_no in range(N):
			thisEl = Cell(self.mesh, cell_no)
			centers[cell_no][0] = thisEl.midpoint().x()
			centers[cell_no][1] = thisEl.midpoint().y()
			centers[cell_no][2] = thisEl.midpoint().z()
			areas[cell_no] = thisEl.volume()

		# compute area fractions (should evaluate to 1.0 everywhere)
		#self.GetVolFracs2D(centers, areas)  # Too slow
		self.GetVolFracsCrossMesh(centers, areas)
				
		# Export a meshfile showing element occupancies
		G = self.VolumeFractionOnElements()
		g = Function(self.V, name = "Area fraction")
		g.interpolate(G)
		self.WriteFieldToFile(dir+filename+'_VolFracs'+'.pvd', g)
		
		# call a solver and save the solution
		set_log_level(PROGRESS)
		self.NewtonIterator()
		
		# save the solution so we can inspect it elsewhere
		self.WriteFieldToFile(dir+filename+'_Solution'+'.pvd', self.solution)
		
	def TestProblem_C(self, dir, filename, centers, areas):
		"""
		2-D Version of Test problem C
		"""

		global L_y
		
		# get height of highest cell in domain
		max_height = 0.0
		for center in centers:
			hy = center[1] 
			if hy > max_height:
				max_height = hy
		print 'max height is %f' % max_height
		L_y = max_height + self.delta

		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(L_y)
		
		# constrain domain to be x-periodic
		pbc = XPeriodicDomain()
		self.V = FunctionSpace(self.mesh, "CG", 1, constrained_domain=pbc)
		self.set_bcs()	
		
		# Use cell centres to evaluate volume occupancy of mesh
		self.GetVolFracsCrossMesh(centers, areas)
		
		# Export a meshfile showing element occupancies
		G = self.VolumeFractionOnElements()
		g = Function(self.V, name = "Area fraction")
		g.interpolate(G)
		self.WriteFieldToFile(dir+filename+'_VolFracs'+'.pvd', g)		
		
		
		# call a solver and save the solution
		set_log_level(PROGRESS)
		self.NewtonIterator()
		
		# save the solution so we can inspect it elsewhere
		self.WriteFieldToFile(dir+filename+'_Solution'+'.pvd', self.solution)

	def TestProblem_C_periodic(self, dir, filename, centers, areas):
		"""
		2-D Version of Test problem C
		"""

		global L_y
		global L_x
		
		L_x = self.L_x
		
		# get height of highest cell in domain
		max_height = 0.0
		for center in centers:
			hy = center[1] 
			if hy > max_height:
				max_height = hy
		print 'max height is %f' % max_height
		L_y = max_height + self.delta

		# we'll need to extend this to something more general later
		# also: clear up the file I/O stuff, it's messy!
		self.SetRectangularMesh(L_y)
		
		self.V = FunctionSpace(self.mesh, "CG", 1)
		self.set_bcs()
		#self.set_bcs_reverse()
		
		# Use cell centres to evaluate volume occupancy of mesh
		self.GetVolFracsCrossMesh(centers, areas)
		
		# Export a meshfile showing element occupancies
		G = self.VolumeFractionOnElements()
		g = Function(self.V, name = "Area fraction")
		g.interpolate(G)
		self.WriteFieldToFile(dir+filename+'_VolFracs'+'.pvd', g)		
		
		
		# call a solver and save the solution
		set_log_level(PROGRESS)
		self.NewtonIterator()
		
		# save the solution so we can inspect it elsewhere
		self.WriteFieldToFile(dir+filename+'_Solution'+'.pvd', self.solution)
		
	
"""
Some supporting classes
"""
	
class VolumeFraction(Expression):
	"""
	Supporting class defining element-wise volume fraction function, for nutrient PDEs.
	"""

	def eval_cell(self, value, x, ufc_cell):
		"""
		Evaluate the cell volume fraction for this mesh element.
		/!\ Assumes vol_fracs is being supplied as a global variable.
		"""
		global vol_fracs
		value[0] = vol_fracs[ufc_cell.index]


class TopDirichletBoundary(SubDomain):
	"""
	Supporting class defining top boundary of rectangular mesh
	"""

	def inside(self, x, on_boundary):
		"""
		Determine whether point x lies on the Dirchlet Boundary subdomain.
		/!\ Assumes L_y is being supplied as a global variable.
		"""
		global L_y
		return bool(near(x[1], L_y) and on_boundary)
		
		
class BaseDirichletBoundary(SubDomain):
	"""
	Supporting class defining base boundary of rectangular mesh
	"""

	def inside(self, x, on_boundary):
		"""
		Determine whether point x lies on the Dirchlet Boundary subdomain.
		"""
		return bool(near(x[1], 0.0) and on_boundary)
		
		
class LeftDirichletBoundary(SubDomain):
	"""
	Supporting class defining left boundary of rectangular mesh
	"""

	def inside(self, x, on_boundary):
		"""
		Determine whether point x lies on the Dirchlet Boundary subdomain.
		/!\ Assumes L_y is being supplied as a global variable.
		"""
		return bool(near(x[0], 0.0) and on_boundary)
		
		
class RightDirichletBoundary(SubDomain):
	"""
	Supporting class defining right boundary of rectangular mesh
	"""

	def inside(self, x, on_boundary):
		"""
		Determine whether point x lies on the Dirchlet Boundary subdomain.
		/!\ Assumes L_y is being supplied as a global variable.
		"""
		global L_x
		return bool(near(x[0], L_x) and on_boundary)
		
		
		
		
		
class XPeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
    	"""
    	Return true if we are on the master boundary at x=0
    	/!\ Assumes that origin = [0,0,z]!
    	"""
        return bool(near(x[0], 0.0) and on_boundary)

    def map(self, x, y):
		""" 
		Map points on the slave boundary to the corresponding master boundary.
		/!\ Takes L_x as a global variable.
		"""
		global L_x
		      
		if near(x[0], L_x):
			y[0] = x[0] - L_x
			y[1] = x[1]           
		else:
			y[0] = x[0]
			y[1] = x[1]
		