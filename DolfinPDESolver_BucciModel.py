"""
DolfinPDESolver_BucciModel.py
==================
A python class structure, written to interface CellModeller4 with the FEniCs/Dolfin finite element library.
This code is an adaptation of the DolfinPDESolver class, designed as an implementation of Bucci et al's model of bacterial interaction via bacteriocin secretion.
As before, the model is solved in non-dimensional form, and is controlled by dimensionless parameter clusters fed in as arguments.
The solver is based on a 2-D domain; extension to 3-D to follow.

This work is explored in detail in Thesis Chapter 7.

Written by William P J Smith, August-September 2016
"""

try:
    from dolfin import *
except ImportError:
    print "Error: could not import dolfin library."
    print "Try calling $ source /Applications/FEniCS.app/Contents/Resources/share/fenics/TestFenicsPath.conf "
import numpy
import math
from pyopencl.array import vec



class DolfinSolverBucci:

	def __init__(self, solverParams):
		""" 
		Initialise the dolfin solver using a dictionary of params.
		"""
		# extract fixed params from params dictionary
		self.pickleSteps = solverParams['pickleSteps'] # Controls when fields are exported as .vtu files
		self.h = solverParams['h']			 # The FEM mesh element length
		self.origin = solverParams['origin'] # The coordinates of the mesh's origin
		self.N_x = int(solverParams['N_x'])  # The number of elements in the mesh along x
		self.L_x = solverParams['L_x']       # The length of the mesh along x in units of l [1um]
		self.D_N = solverParams['D_N']       # The nutrient damkohler number (a timescale ratio)
		self.D_T = solverParams['D_T']       # The toxin damkohler number (a timescale ratio)
		self.E_T = solverParams['E_T']       # The toxin effectiveness factor (a timescale rato)
		self.eta_N = solverParams['eta_N']   # The nutrient saturation ratio
		self.f_T = solverParams['f_T']       # The fractional investment into toxin production (0<=f<=1)
		self.delta = solverParams['delta']   # The boundary height in units of l [1um]
		
		# some attributes that we'll update on the fly: set them to None for now
		self.boundaryCondition = None
		self.mesh = None
		self.V = None
		self.solution = None
		
		
	def SolveBucciPDE(self, centers, centers_A, centers_B, areas_A, areas_B, filename, dir, stepNum=0):
		"""
		The solver wrapper called by the simulation.
		Note that this combines elements from DolfinPDESolver_DualFields (coupled PDEs) and " "_RectangleTwoTypes (split volume fractions)
		"""
		global L_y
		
		# get height of highest cell in domain using the combined list "centers"
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
		
		# define a 2-dimensional function space and functions on it
		self.V = VectorFunctionSpace(self.mesh, 'CG', 1, 2)
		phi = TestFunction(self.V)
		W = Function(self.V)
		p, q = split(phi)   # test functions
		u, v = split(W)     # solution fields: u is Nutrient (N), v is Toxin (T)
		
		# Use cell centres to evaluate volume occupancy fields for each cell type
		self.GetVolFracsCrossMesh_A(centers_A, areas_A)
		self.GetVolFracsCrossMesh_B(centers_B, areas_B)
		
	    # Define boundary conditions 
		tbc = TopDirichletBoundary()
		bc0 = DirichletBC(self.V, Constant((1.0, 0.0)), tbc) # N = Nbulk , T = 0 at top of domain
		bcs = [bc0]
		
		# Read off parameter values
		D_N = self.D_N
		D_T = self.D_T
		eta_N = self.eta_N
		
		# Weak statement of the equations
		F1 = dot(grad(u), grad(p))*dx + D_N*(u / (u+eta_N))*(VolumeFraction_A()+VolumeFraction_B())*p*dx 
		F2 = dot(grad(v), grad(q))*dx - D_T*(u / (u+eta_N))*(VolumeFraction_A())*q*dx
		F = F1 + F2	

	    # some additional parameters
		set_log_level(WARNING)
		parameters["form_compiler"]["quadrature_degree"]=1
				
		# call the in-built Newton solver
		solve(F==0, W, bcs, solver_parameters = {"newton_solver":
										        {"relative_tolerance": 1e-6,
										         "convergence_criterion": "incremental",
										         "linear_solver": "gmres",
									       	     "maximum_iterations": 50
										         }})	
		
		# unpack the solution fields from vector W, one at a time
		self.solution = W.split()[0]
		u_local_N = self.InterpolateToCenters2D(centers)
		if stepNum % self.pickleSteps == 0:
			self.WriteFieldToFile(dir+filename+'_Nutrient.pvd')
			
		self.solution = W.split()[1]
		u_local_T = self.InterpolateToCenters2D(centers)
		if stepNum % self.pickleSteps == 0:
			self.WriteFieldToFile(dir+filename+'_Toxin.pvd')
	
		return u_local_N, u_local_T
		
		plot(self.solution), interactive()
		
		
	def SetRectangularMesh(self, L_y):
		"""
		Given a mesh height L_y (and the self.N_x, self.L_x parameters), create an L_x by L_y cross mesh.
		Uses h as the mesh element size parameter. Center must be a Dolfin Point() instance.
		/!\ Exports N_y as a global variable
		/!\ Mesh MUST be crossed for volume fraction calculator to work
		"""
		global N_y
		
		mesh_type = 'crossed'
		N_y = int(round(L_y/self.h))
		self.mesh = RectangleMesh(0, 0, self.L_x, L_y, self.N_x, N_y, mesh_type)
		
		
	def GetVolFracsCrossMesh_A(self, centers_A, vols):
		"""
		Create a global list of the cell volume fractions in mesh elements - for type A cells.
		Assumes that self.mesh is up to date.
		'Volumes' are equivalent to areas in 2D.
		/!\ Exports the array vol_fracs as a global array, for use by VolumeFraction.
		/!\ Takes 
		"""
		global vol_fracs_A, L_y, N_y

		# assign elements of cells
		elements = self.AssignElementsToDataCrossMesh(centers_A)
		
		# need to define volume fraction for every element in the mesh
		# (not just the canonical elements for counting)
		num_elements = self.mesh.num_cells()

		# sum cell volumes over each element
		a = (self.L_x*L_y) / (4.0*float(self.N_x)*float(N_y))
		vol_fracs_A = numpy.bincount(elements,vols,num_elements) / a
		
		
	def GetVolFracsCrossMesh_B(self, centers_B, vols):
		"""
		Create a global list of the cell volume fractions in mesh elements - for type B cells.
		Assumes that self.mesh is up to date.
		'Volumes' are equivalent to areas in 2D.
		/!\ Exports the array vol_fracs as a global array, for use by VolumeFraction.
		/!\ Takes 
		"""
		global vol_fracs_B, L_y, N_y

		# assign elements of cells
		elements = self.AssignElementsToDataCrossMesh(centers_B)
		
		# need to define volume fraction for every element in the mesh
		# (not just the canonical elements for counting)
		num_elements = self.mesh.num_cells()

		# sum cell volumes over each element
		a = (self.L_x*L_y) / (4.0*float(self.N_x)*float(N_y))
		vol_fracs_B = numpy.bincount(elements,vols,num_elements) / a
		
		
	def InterpolateToCenters2D(self, centers):
		"""
		Interpolate a solution object u onto a list of cell coordinates
		"""
		u = self.solution
		data_t = tuple(map(tuple, centers)) 	   		     # Convert to tuple format
		u_local = numpy.zeros((len(data_t),),numpy.float64)  # preallocate solution array
		for i in range(0,len(data_t)):				  		 # loop over all cells
			u_local[i] = numpy.max(u(data_t[i][0:2]), 0.0)	 # extrapolate solution value at cell centre
		
		return u_local	
		
		
	def WriteFieldToFile(self, filename):
		"""
		Export the PDE solution as a pvd mesh.
		"""
		print "Writing fields..."
		File(filename) << self.solution
		print 'Done.'
		
		
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


	def TestProblem_B(self, dir, filename, height):
		"""
		Solves the homogenised reaction-diffusion problem on a 2D standard mesh. 
		Imaginary cells of type A are placed at the centroids of each element, so that vol_fracs_A should evaluate to 1 everywhere,
		and vol_fracs_B to 0 everywhere.
		You can check this by eye, since we export the volume fraction function too.
		This is meant to test the main SolvePDE method, so keep the two similar!
		"""
		global L_y
		
		L_y = height
		self.SetRectangularMesh(L_y)

		# create imaginary cells of type A, filling the first 50% of elements (assuming regular indexing from bottom)
		N = int(0.5 * self.mesh.num_cells())
		centers_A = numpy.zeros((N,), vec.float4)
		centers_B = numpy.zeros((0,), vec.float4) 
		areas_A = numpy.zeros((N,))  
		areas_B = numpy.zeros((0,))
		
		for cell_no in range(N):
			thisEl = Cell(self.mesh, cell_no)
			centers_A[cell_no][0] = thisEl.midpoint().x()
			centers_A[cell_no][1] = thisEl.midpoint().y()
			centers_A[cell_no][2] = thisEl.midpoint().z()
			areas_A[cell_no] = thisEl.volume()
			
		centers = centers_A	
		
		# define a 2-dimensional function space and functions on it
		self.V = VectorFunctionSpace(self.mesh, 'CG', 1, 2)
		phi = TestFunction(self.V)
		W = Function(self.V)
		p, q = split(phi)   # test functions
		u, v = split(W)     # solution fields: u is Nutrient (N), v is Toxin (T)
		
		# Use cell centres to evaluate volume occupancy fields for each cell type
		self.GetVolFracsCrossMesh_A(centers_A, areas_A)
		self.GetVolFracsCrossMesh_B(centers_B, areas_B)
		
	    # Define boundary conditions 
		tbc = TopDirichletBoundary()
		bc0 = DirichletBC(self.V, Constant((1.0, 0.0)), tbc) # N = Nbulk , T = 0 at top of domain
		bcs = [bc0]
		
		# Read off parameter values
		D_N = self.D_N
		D_T = self.D_T
		eta_N = self.eta_N
		
		# Weak statement of the equations
		F1 = dot(grad(u), grad(p))*dx + D_N*(u / (u+eta_N))*(VolumeFraction_A()+VolumeFraction_B())*p*dx 
		F2 = dot(grad(v), grad(q))*dx - D_T*(u / (u+eta_N))*(VolumeFraction_A())*q*dx
		F = F1 + F2	

	    # some additional parameters
		set_log_level(WARNING)
		parameters["form_compiler"]["quadrature_degree"]=1
				
		# call the in-built Newton solver
		solve(F==0, W, bcs, solver_parameters = {"newton_solver":
										        {"relative_tolerance": 1e-6,
										         "convergence_criterion": "incremental",
										         "linear_solver": "lu",
									       	     "maximum_iterations": 50
										         }})	
		
		# unpack the solution fields from vector W, one at a time
		self.solution = W.split()[0]
		u_local_N = self.InterpolateToCenters2D(centers)
		self.WriteFieldToFile(dir+filename+'TestBSolution_Nutrient.pvd')
			
		self.solution = W.split()[1]
		u_local_T = self.InterpolateToCenters2D(centers)
		self.WriteFieldToFile(dir+filename+'_TestBSolution_Toxin.pvd')
	
		# Export meshfiles showing element occupancies
		#PhiA = VolumeFraction_A(), PhiB = VolumeFraction_B()
		#gA = Function(self.V, name = "Area_fraction_A"), gB = Function(self.V, name = "Area_fraction_B")
		#gA.interpolate(PhiA), gB.interpolate(PhiB)
		
		#self.WriteFieldToFile(dir+filename+'TestBSolution_VolFracsA'+'.pvd', gA)
		#self.WriteFieldToFile(dir+filename+'TestBSolution_VolFracsB'+'.pvd', gB)	    




"""
Some supporting classes
"""
	
class VolumeFraction_A(Expression):
	"""
	Supporting class defining element-wise volume fraction function, for nutrient PDEs.
	"""

	def eval_cell(self, value, x, ufc_cell):
		"""
		Evaluate the cell volume fraction for this mesh element.
		/!\ Assumes vol_fracs is being supplied as a global variable.
		"""
		global vol_fracs_A
		value[0] = vol_fracs_A[ufc_cell.index]
		
		
class VolumeFraction_B(Expression):
	"""
	Supporting class defining element-wise volume fraction function, for nutrient PDEs.
	"""

	def eval_cell(self, value, x, ufc_cell):
		"""
		Evaluate the cell volume fraction for this mesh element.
		/!\ Assumes vol_fracs is being supplied as a global variable.
		"""
		global vol_fracs_B
		value[0] = vol_fracs_B[ufc_cell.index]

		
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
