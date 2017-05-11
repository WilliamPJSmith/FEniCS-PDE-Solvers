"""
DolfinumpyDESolver_Circle.py
==================

A python class structure written to interface CellModeller4 with the FEniCs/Dolfin finite element library.
This is a analogue of the original DolfinumpyDESolver class, this time designed for radial simulations.
At some point, it'd be useful to unite the various Solver class versions into one all-purpose system.

Created: W. P. J. Smith, 13.03.17
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

class CircleDolfinSolver:

	def __init__(self, solverParams):
		""" 
		Initialise the dolfin solver using a dictionary of params.
		"""
		# extract fixed params from params dictionary
		self.pickleSteps = solverParams['pickleSteps']
		self.h = solverParams['h']
		self.delta = solverParams['delta']
		self.eta = solverParams['eta']
		self.Da = solverParams['Da'] 
		self.rel_tol = solverParams['rel_tol']
		self.max_iter = solverParams['max_iter']

		# some attributes that we'll update on the fly: set them to None for now
		self.mesh = None
		self.V = None
		self.solution = None
		
		
	def SolvePDE(self, centers, areas, filename, dir, stepNum=0):
		"""
		Solve the nutrient consumption PDE and return the corresponding field
		"""
		global area_fracs
		
		self.mesh = self.BuildDiscMeshAroundConfig(centers)
		area_fracs = self.ComputeCellDensityFunction(centers, areas)
	
		# Define function space
		self.V = FunctionSpace(self.mesh, "Lagrange", 1)

		# Define boundary conditions on this mesh
		u0 = Constant(1.0)
		boundary = CircleBoundary()
		bc = DirichletBC(self.V, u0, boundary)

		# Define variational problem
		u = Function(self.V, name = "Nutrient")
		v = TestFunction(self.V)
		F = dot(grad(u), grad(v))*dx + self.MonodNutrientSink(u)*v*dx

		# Call built-in Newton solver
		set_log_level(PROGRESS) 			# very detailed info for testing
		#set_log_level(WARNING) 				# near-silent info for simulations
		solve(F == 0, u, bc, solver_parameters = {"newton_solver":
										        {"relative_tolerance": self.rel_tol,
										         "convergence_criterion": "incremental",
										         "linear_solver": "gmres",
									       	     "maximum_iterations": self.max_iter
										         }})
		self.solution = u

		# write field to file
		if stepNum % self.pickleSteps == 0:
			self.WriteFieldToFile(dir+filename+'_Nutrient.pvd', self.solution)
			
		# interpolate solution
		u_local_N = self.InterpolateToCenters2D(centers)
		return u_local_N
		
		
	def BuildDiscMeshAroundConfig(self, centers):
		"""
		Given a configuration of cells, create a circular mesh containing that configuration.
		"""
		centroid = numpy.mean(centers,axis=0)
		cdists = numpy.add(numpy.square(centers[:,0]-centroid[0]), numpy.square(centers[:,1]-centroid[1]))
		radius = numpy.sqrt(numpy.amax(cdists))
		mesh = CircleMesh(Point(centroid[0], centroid[1]), radius+self.delta, self.h)
	
		return mesh
		
		
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
			
		
	def WriteFieldToFile(self, filename, u):
		"""
		Export the PDE solution as a pvd mesh.
		"""
		print "Writing fields..."
		File(filename) << u
		print 'Done.'
		
			
	def MonodNutrientSink(self, u):
		""" 
		Monod function with which to build RHS of nutrient consumption PDE.
		"""
		a = Constant(self.Da)
		b = Constant(self.eta)
	
		return a * u * VolumeFraction() / (b + u)
			
		
	def ComputeCellDensityFunction(self, centers, areas):
		"""
		Given a mesh and a list of cell centroids, compute the cell density (area fraction) in each mesh element.
		"""
	
		Elements = self.AssignPointsToTriangles(centers)
		T = self.mesh.num_cells()
		element_areas = numpy.zeros((T,)) 
		for i in range(0,T):
			element_areas[i] = Cell(self.mesh,i).volume()
		binned_area_fracs = numpy.bincount(Elements,areas,T)	
		area_fracs = numpy.divide(binned_area_fracs, element_areas)
		
		return area_fracs


	def AssignPointsToTriangles (self, points):
		""" 
		Given an array of points (Points) and a mesh (POI, TRI, TRC), map points to mesh elements.
		"""
		P = len(points)
		TRI = self.mesh.cells()
		POI = self.mesh.coordinates()
		TRC = numpy.sum(POI[TRI],axis=1) / float(3)
		Elements = numpy.zeros((P,), numpy.int32)
		for p in range(0,P):
			px = points[p,0]
			py = points[p,1]
			Elements[p] = self.FindTriangle(POI, TRI, TRC, px, py)
		
		return Elements
		
		
	def FindTriangle(self, POI, TRI, TRC, px, py):
		""" 
		Given a focal point p=(px,py), and a mesh (P,TRI) with centroids TRC, determine the mesh cell t in which lies p.
		"""
	
		# Sort triangles by their separation from the focal point
		SQD = numpy.add(numpy.square(TRC[:,0]-px), numpy.square(TRC[:,1]-py))
		STR = numpy.argsort(SQD)

		# Test triangles in this order
		for i in range(0,len(STR)):
			x = self.CheckTriangle(POI, TRI, px, py, STR[i])
			if not any(x < 0):
				return STR[i]
	
		# If every triangle test fails, the point lies outside the mesh
		raise ValueError('Focal point lies outside the mesh - could not assign element.') 
	
	
	def CheckTriangle(self, POI, TRI, px, py, t):
		""" 
		Given a point p=(px,py), and a mesh (P,TRI), return barycentric coefficients for p in t.
		"""
		b = numpy.array([px,py,1.0])
		p1 = POI[TRI[t,0]]; p2 = POI[TRI[t,1]];	p3 = POI[TRI[t,2]]
		p1 = numpy.append(p1,1.0); p2 = numpy.append(p2,1.0); p3 = numpy.append(p3,1.0)
		A = numpy.column_stack((p1,p2,p3))
		x = numpy.linalg.solve(A, b)
	
		return x
		
		
	def GetCellCentersAndAreas(self, cellStates):
		"""
		Given cellStates, return arrays of cell centers and areas.
		/!\ NOTE: we're now working with configuration data in standard numpy arrays, not vec4s.
		"""
	
		N = len(cellStates)
		centers = numpy.zeros((N,3)) 
		areas = numpy.zeros((N,))
		for state in cellStates.values():
			i = state.idx
			L = state.length
			R = state.radius
			x = state.pos[0]
			y = state.pos[1]
			centers[i,0] = x
			centers[i,1] = y
			areas[i] = (math.pi*R**2) + (2*L*R)

		return centers, areas

"""
Supporting Classes
"""
		
class VolumeFraction(Expression):
	"""
	Supporting class defining element-wise area fraction function, for nutrient PDEs.
	"""
	def eval_cell(self, value, x, ufc_cell):
		"""
		Evaluate the cell volume fraction for this mesh element.
		/!\ Assumes vol_fracs is being supplied as a global variable.
		"""
		global area_fracs
		value[0] = area_fracs[ufc_cell.index]
		
		
class CircleBoundary(SubDomain):
	"""
	Supporting class defining outer boundary of circular mesh.
	"""
	def inside(self, x, on_boundary):
		"""
		Determine whether point x lies on the Dirchlet Boundary subdomain.
		"""
		return on_boundary
