# FEniCS-PDE-Solvers

Brief descriptions for each solver:

1. DolfinPDESolver.py
=====================
Python module for nutrient field modelling with CellModeller. 
Given a list of cell centroids <centers>,  a list of cell volumes <volumes>, an output filename <filename> and directory <dir>, use a finite element method to compute steady state solutions <u(x,y,z)> to the reaction-diffusion nutrient equation

u_t = A.u'' - B*phi*u / (u+C) 

within a cuboidal domain, subject to various boundary conditions at the domain’s edges.
Here, A, B and C are positive constants, and where phi(x,y,z) is a cell density function. Interpolate <u> to compute a list of nutrient concentrations <u_local> at each cell’s centroid.


2. DolfinPDESolver_Circle.py
=============================
Similar to DolfinPDESolver.py except now the (2-D) domain is circular.


3. DolfinPDESolver_Rectangle.py
=============================
Similar to DolfinPDESolver.py except now the (2-D) domain is rectangular.


4. DolfinPDESolver_DualFields.py
=====================
Extension to DolfinPDESolver.py: now cells consume two rate-limiting nutrients, represented by two coupled solute fields <u> and <v>.
