===================================
PHYS375 Coding Project, by John Ray
===================================
This is a single-file program that requires the following 4 Python packages to run:

* SciPy
* NumPy 
* Matplotlib
* Tkinter

Of these, SciPy, NumPy, and Matplotlib MUST be installed by the user.

SciPy can only run on an older version of NumPy (< version 2), so please ensure that this is the case 
before running. The other libraries should function regardless of version.

Upon loading, you will see a GUI appear. This GUI uses the Tkinter library which is installed as a 
default in Python. As a result, you should not need to manually install anything for this.

===============================
Instructions for using the GUI:
===============================

The Graphical User Interface (GUI) gives options to modify the chain size, hopping parameters 
of t1 and t2, and the initial boundary conditions of the system, following the default constraints 
of the variables. Additionally, there is the option to add new hopping parameters beyond 
nearest-neighbour hopping, with preset parameters from A -> A and B -> B being provided.

Three graphs are generated automatically each time the program is loaded or updated. These show a 
plot of the system's energy spectrum, a phase diagram of the energy, as a function of t1/t2, and a 
visualisation of the system's wavefunction. The graphs can be updated from the user inputs by pressing
the "update" button provided.