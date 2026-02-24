# Imports Numpy and Scipy for algebra
import numpy as np
import scipy.linalg as la
# Imports Matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
#from matplotlib import ticker
# Uses Tkinter for the GUI, as I've used it before
import tkinter as tk
# Ttk gives basic buttons, dropdown menus, etc. that I can use for convenience
from tkinter import ttk

"""

                    PHYS375 Project, by John Ray
             A simulation of a one-dimensional SSH Model.

"""

# Function to create the Hamiltonian
def Hamiltonian(N, t1, t2, BC, t3, t4, customHops):
    """
    Definition of each hopping parameter:

    T1 = A -> B, T2 = B -> A, T3 = A -> A, and T4 = B -> B.
    
    """
    # Initialises the size of the rows and columns as 2*N.
    rows = 2*N
    cols = rows
    # Creates a 2N by 2N matrix for H (initialised as full of float zeros)
    H = np.zeros((rows,cols),dtype=float)

    # Maps the sublattices to the appropriate cells (A -> 2*cell, B -> 2*cell + 1)
    def siteIndex(cell, sublattice):
        if sublattice == 'A':
            return 2*int(cell)
        else:
            return 2*int(cell) + 1

    # To add new hopping terms (e.g. t3 and t4), with real amplitudes (so conjugates = originals)
    def addTerm(fromCell, fromSublattice, toCell, toSublattice, Amplitude):
        # This function varies the behaviour between OBC and PBC
        if BC == "Periodic Boundary Conditions":
            """
            For PBC, the chain is in a loop, so the cell number is wrapped into a loop for all terms.
             e.g. the Nth + 1 cell = the 1st cell.

             This is done by letting the cell number = its remainder across the chain length, N.

            """
            fromCellcorrected = fromCell % N
            toCellcorrected = toCell % N
        else:
            # For OBC, the chain is not a loop, so terms greater or less than the chain length are ignored.
            if (fromCell < 0) or (fromCell >= N) or (toCell < 0) or (toCell >= N):
                return
            # Other OBC terms are set to stay as they are.
            fromCellcorrected = fromCell
            toCellcorrected = toCell
        """
        Here, the indices of the Hamiltonian are taken by putting in the recalculated cell number and sublattice number into the 
        "siteIndex" function, which is used to fill the Hamiltonian in the way described in the report's "Theoretical Approach" section.

        This is done by filling the i,jth and the j,ith terms with the amplitude of the hopping parameter that goes from the sublattice 
        that gives i when x 2 (if that sublattice is A), or when x 2 and + 1 (if that sublattice is B). This logic is done the same way 
        for j, and loops for all N of the matrix.
        
        For example, the t1 term is from A to B with range n, so i is 2 *N, and j is 2*N + 1. The first t1 term is therefore when N = 0, 
        so i = 0 and j = 1, the second t1 term is when N = 1, so i = 2, and j = 3, etc. From this, the terms H[0,1], H[1,0], H[1,2], 
        H[2,1], etc. are filled for t1; giving an intermittent diagonal of t1s in the matrix. This is then done for t2 and for any 
        other hopping parameters that the user inputs, using the same logic.
        """
        i = siteIndex(fromCellcorrected, fromSublattice)
        j = siteIndex(toCellcorrected, toSublattice)
        H[i, j] += float(Amplitude)
        H[j, i] += float(Amplitude)

    """ Code for the default nearest-neighbour SSH hopping parameters, t1 and t2, without any additional terms added. """

    #The function below generates a t1 intracell hopping parameter (from A_n <-> B_n).
    if t1 is None or t2 is None:
        # Failsafe error code - prevents the program from running without t1 and t2.
        raise ValueError("t1 and t2 must be provided")
    # Simply adds in t1 terms for the entire chain size, using the previous "addTerm" function.
    for n in range(N):
        addTerm(n, 'A', n, 'B', t1)

    # Generates a t2 intercell hopping parameter (from B_n <-> A_{n+1})
    for n in range(N):
        addTerm(n, 'B', n + 1, 'A', t2)

    """ Beyond nearest-neighbour hopping Hamiltonian code: """

    # Generates t3 and t4 cell hoppings, if specified to by the user (with a range between the cells of 1)
    if t3 != 0.0:
        for n in range(N):
            # Adds a t3 term.
            addTerm(n, 'A', n + 1, 'A', t3)
    if t4 != 0.0:
        for n in range(N):
            # Adds a t4 term.
            addTerm(n, 'B', n + 1, 'B', t4)
    """
    Adds custom longer-range hopping parameters, if requested. This is done by taking the "term" variable as a dictionary,
    with each dictionary of a custom term having the form "{'From':'A'/'B', 'To':'A'/'B', 'Range':int, 'Amplitude':float}".
    This allows arbitrary, longer-range, hopping parameters to be added. The choice of sublattices the parameter comes from / goes to, 
    the range of the parameter, and the amplitude (size) of the parameter, can all be specified by the user.
    """
    if customHops is not None:
        for term in customHops:
            fromTerm = term.get('From')
            toTerm = term.get('To')
            rangeTerm = int(term.get('Range', 0))
            amplitudeTerm = float(term.get('Amplitude', 0.0))
            for n in range(N):
                # Simply assigns each part of the dictionary "term" variable to a separate variable, and parses
                # this into the previous "addTerm" function.
                addTerm(n, fromTerm, n + rangeTerm, toTerm, amplitudeTerm)

    # Returns the Hamiltonian.
    return H

# Function to solve the eigenvectors and eigenvalues of the Hamiltonian matrix.
def EigenSolver(H):
        # Variables are made global, for future use in graphical plotting
        global evals
        global evecs
        # Uses Scipy's Linalg "eigh" function that solves for the values.
        evals,evecs = la.eigh(H)
        return evals, evecs

# Main master function for the computational code
def functionality(N,t1,t2,BC, t3, t4, customHops):
    # Variables are made global, for future use in graphical plotting
    global evals, evecs, H
    # Calls the Hamiltonian function
    H = Hamiltonian(N, t1, t2, BC, t3, t4, customHops)
    # This code reformats the matrix to round all values to 2d.p
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # Prints the Hamiltonian for debugging / to be viewed terminal-side by the user, if desired.
    print(f"Hamiltonian: \n {H}")
    # Calls the eigenvector / eigenvalue solver function.
    EigenSolver(H)
    # Prints the eigenvalues and eigenvectors, for debugging.
    print("    ")
    print(f"Eigenvalues: {evals}")
    print("    ")
    print(f"Eigenvectors: {evecs}")
    return
# This function simply creates the f-string used in the Tkinter "Spinbox" to display the details of added custom hopping parameters.
def description(term):
    # The amplitude term must be a single variable in the f-string for its float value to be reformatted.
    amp = term['Amplitude']
    return f"{term['From']} -> {term['To']}, range = {term['Range']} and amplitude = {amp:.3g}."

""" Graphical User Interface code (using a Tkinter GUI): """

# Defines the window as a class for the Tkinter UI
class window(tk.Tk):
    # Standard required __init__ function, for the "window" class.
    def __init__(self):
        super().__init__() # Keeps initial conditions set here in other methods like the UI.
        # Sets a title to the window and the window's size
        self.title("PHYS375 Coding Project, by John Ray")
        self.geometry("1920x768")
        # Defines the main variables:
        self.N = tk.IntVar(value=20)
        self.T1 = tk.DoubleVar(value=1)
        self.T2 = tk.DoubleVar(value=1.5)
        self.T3 = tk.DoubleVar(value=0)
        self.T4 = tk.DoubleVar(value=0)
        self.customHops = None # No custom hopping parameters, by default.
        self.BoundCond = tk.StringVar(value="OBC")
        # Creates frames for the GUI (left-centres the buttons and such).
        self.frame = ttk.Frame(self, padding = 1)
        self.frame.pack(side=tk.LEFT, fill=tk.Y,expand=True)
        self.UI() # Loads the UI function.
        # The functionality function is called, with the appropriate inputs in the form "self.VARIABLENAME.get()".
        functionality(self.N.get(),self.T1.get(),self.T2.get(),self.BoundCond.get(), self.T3.get(), self.T4.get(), self.customHops)
        # Debugging code, left for terminal-side verification.
        print("Loading UI...")
        # The plot function is called, using the functionality function's global "evals" and "evecs" as inputs, beside other variables.
        self.plot(evals,evecs,self.T1.get(),self.T2.get())

    """ The main UI function: """

    def UI(self):
        """ Code for the left-most user-inputs panel of the GUI, from the top-down. """
        # Separating line (just to look nicer), from the ttk sub-package.
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        # The title shown in the program's window. Text details have to be added using ".config", and packed in a later line.
        Title = tk.Label(self.frame,text="PHYS375 Coding Project")
        Title.config(font=("Helvetica", 12, "bold", "underline"))
        Title.pack()
        # The window's subtitle.
        Subtitle = tk.Label(self.frame,text="by John Ray")
        Subtitle.config(font=("Helvetica", 10))
        Subtitle.pack()
        # Another separator.
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        # A command function for the chain size scale (ensuring that only integer values can be inserted):
        def integer(val):
            # Takes any float inputs and converts them to integer format.
            val = int(float(val))
            # Sets this new value to be the value for the N Tkinter intVar.
            return self.N.set(val)
        # Buttons and menus (uses Tkinter's ttk sub-library):
        ttk.Label(self.frame,text="Chain size (from 10 to 100 cells):").pack()
        # Chain size can only be an integer value, so I have to use the command=integer parameter here to 
        # ensure this (otherwise the scale still shows float values, but in reality on takes integers).
        ttk.Scale(self.frame, variable=self.N,from_ = 10, to = 100, command=integer).pack()
        ttk.Label(self.frame, textvariable = self.N).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        # Titles with underlines or larger font have to have effects applied via ".config()", as mentioned earlier.
        HP = ttk.Label(self.frame,text="Standard hopping parameters:")
        HP.config(font=("Helvetica",10,"underline"))
        HP.pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        # Basic function to round t1 values to 2 d.p.
        def t1round(val):
            val = f'{float(val):.02f}'
            return self.T1.set(val)
        ttk.Label(self.frame,text="Value for t1 (0.0 to 2.0):").pack()
        # Creates a scale with the variable T1, from 0.0 to 2.0:
        ttk.Scale(self.frame, variable=self.T1,from_ = 0.0, to = 2.0, command=t1round).pack()
        ttk.Label(self.frame, textvariable=self.T1).pack()
        # Basic function to round t2 values to 2 d.p.
        def t2round(val):
            val = f'{float(val):.02f}'
            return self.T2.set(val)
        ttk.Label(self.frame,text="Value for t2 (0.0 to 2.0):").pack()
        # Creates a scale with the variable T2, from 0.0 to 2.0:
        ttk.Scale(self.frame, variable=self.T2, from_ = 0.0, to = 2.0,command=t2round).pack()
        ttk.Label(self.frame, textvariable=self.T2).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)

        # Creates a phase variable to show whether the system is in a trivial / topological phase, or at the phase transition.
        self.phase_var = tk.StringVar(self.frame,value="")
        # Main function for GUI display:
        def phaseAnalysis(t1,t2):
            t1 = float(t1.get())
            t2 = float(t2.get())
            # Phase variable to be used in later calculations
            phase = f'{float(t1/t2):.02f}'
            if t1 > t2:
                self.phase_var.set(f"The system is in a trivial phase: {phase}")
            elif t2 > t1:
                self.phase_var.set(f"The system is in a topological phase: {phase}")
            else:
                self.phase_var.set(f"The system is at a phase transition: {phase}")
            return phase
        # Calls the phaseAnalysis function.
        phaseAnalysis(self.T1,self.T2)
        ttk.Label(self.frame, textvariable=self.phase_var).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        ttk.Label(self.frame,text="Please select boundary conditions:").pack()
        # This option menu works better for fewer options in my opinion. The OBC option is put twice
        # (the first time sets it as the default and the second time adds it to the displayed list).
        ttk.OptionMenu(self.frame, self.BoundCond, "Open Boundary Conditions","Open Boundary Conditions","Periodic Boundary Conditions").pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        Subtitle2 = ttk.Label(self.frame, text="Extending the SSH Model beyond nearest-neighbours hopping:")
        Subtitle2.config(font=("Helvetica", 10, "underline"))
        Subtitle2.pack()
        # Basic function to round t3 values to 2 d.p.
        def t3round(val):
            val = f'{float(val):.02f}'
            return self.T3.set(val)
        ttk.Label(self.frame,text="Value for a hopping parameter from A -> A (0.0 to 2.0):").pack()
        # Creates a scale with the variable t3, from 0.0 to 2.0:
        ttk.Scale(self.frame, variable=self.T3,from_ = 0.0, to = 2.0, command=t3round).pack()
        ttk.Label(self.frame, textvariable=self.T3).pack()
        # Basic function to round t4 values to 2 d.p.
        def t4round(val):
            val = f'{float(val):.02f}'
            return self.T4.set(val)
        ttk.Label(self.frame,text="Value for a hopping parameter from B -> B (0.0 to 2.0):").pack()
        # Creates a scale with the variable t4, from 0.0 to 2.0:
        ttk.Scale(self.frame, variable=self.T4,from_ = 0.0, to = 2.0, command=t4round).pack()
        ttk.Label(self.frame, textvariable=self.T4).pack()
        # To correctly format the box for custom parameters, a new Tkinter frame is needed.
        frm = ttk.Frame(self.frame)
        frm.pack(fill='none', pady=0)
        # The buttons in this frame are packed on a grid, to save space (so more can be entries are visible simultaneously).
        ttk.Label(frm, text="From sublattice:").grid(row=0, column=0, sticky='w')
        self.fromSublattice = tk.StringVar(value='A')
        ttk.OptionMenu(frm, self.fromSublattice, 'A', 'A', 'B').grid(row=0, column=1, sticky='w')
        ttk.Label(frm, text="To sublattice:").grid(row=0, column=2, sticky='w')
        self.toSublattice = tk.StringVar(value='B')
        ttk.OptionMenu(frm, self.toSublattice, 'B', 'A', 'B').grid(row=0, column=3, sticky='w')
        ttk.Label(frm, text="Range of parameter:").grid(row=1, column=0, sticky='w')
        self.parameterRange = tk.IntVar(value=0)
        # This time I use a Tkinter "Spinbox" to save more space, and to enforce integer inputs.
        ttk.Spinbox(frm, from_=-20, to=20, textvariable=self.parameterRange, width=6).grid(row=1, column=1)
        ttk.Label(frm, text="Amplitude:").grid(row=1, column=2, sticky='w')
        self.amp_entry = tk.Entry(frm, width=10)
        self.amp_entry.grid(row=1, column=3)
        self.amp_entry.insert(0, "1.0")

        # A Tkinter "listbox" is used to show these terms:
        ttk.Label(self.frame, text="Current hopping terms:").pack(anchor='w', pady=(8,0))
        self.terms_listbox = tk.Listbox(self.frame, height=3, width=40)
        self.terms_listbox.pack(fill='x', pady=0)

        # Another Tkinter frame is used to place the lower buttons in-line:
        btnfrm = ttk.Frame(self.frame)
        btnfrm.pack(fill='none', pady=0)
        # The command functions called here are defined later.
        ttk.Button(btnfrm, text="Add term", command=self.addTerm).pack(side='left', padx=2)
        ttk.Button(btnfrm, text="Remove selected", command=self.removeSelected).pack(side='left', padx=2)
        ttk.Button(btnfrm, text="Clear terms", command=self.clearTerms).pack(side='left', padx=2)
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        """
        An update button calls both the functionality() and self.plot() functions by appending them both to a temporary lambda function
        (this is because of how ttk variables' command parameters can only take one input function, unfortunately).

        The functionality function is initialised with the values from self.N, self.T1, and self.T2.
        
        """
        ttk.Button(self.frame, text="Update", command=lambda:(functionality(self.N.get(),self.T1.get(),self.T2.get(),self.BoundCond.get(),self.T3.get(),self.T4.get(),self.customHops),self.plot(evals,evecs,self.T1.get(),self.T2.get()),phaseAnalysis(self.T1,self.T2))).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        # Decreases the font size of the axis, and redefines the font of Matplotlib graphs to be Times New Roman.
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rc('font',family='Times New Roman')
        # Creates a second right-centred frame for the graphs to be added to.
        plotframe = ttk.Frame(self, padding=8)
        plotframe.pack(side=tk.RIGHT, padx=24,pady=9)
        # Creates figures for each graph (I had to fiddle with the figsize a bit to get a good size).
        self.fig = plt.Figure(figsize=(5,4))
        self.fig2 = plt.Figure(figsize=(5,4))
        self.fig3 = plt.Figure(figsize=(5,4)) 
        # Creates axis to each graph.
        self.ax = self.fig.add_subplot()
        self.ax2 = self.fig2.add_subplot()
        self.ax3 = self.fig3.add_subplot()
        # Enables the figures to be attached to a Tkinter "canvas" that enables Tkinter variables to be used.
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotframe)
        # Fills in the canvas' information:
        self.canvas.draw()
        # Unpacks (creates) the widget associated with Tkinter:
        self.canvas.get_tk_widget().pack()
        # Does the same for the canvas widget.
        self.canvas._tkcanvas.pack()
        # Repeat process for the second graph.
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plotframe)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack()
        self.canvas2._tkcanvas.pack()
        # Adds a nice separating line to the bottom of the right-centred frame
        ttk.Separator(plotframe).pack(fill=tk.X, pady=10)
        # Creates another right-centred frame for the third graph to be added to.
        plotframe2 = ttk.Frame(self, padding=0)
        plotframe2.pack(side=tk.RIGHT, padx=30,pady=9)
        # Same logic again.
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plotframe2)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack()
        self.canvas3._tkcanvas.pack()
     
    # Defines the plot function:
    def plot(self,evalues,evectors,t1,t2):
        # Refreshes the data of both graphs
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        # Deletes the boxes of the phase diagram (which are not cleared fully by self.ax2.clear())
        if getattr(self, "cbar", None) is not None:
            self.fig2.delaxes(self.cax)
            self.cax = None
        # Creates an enumerating variable for the energy spectrum graph:
        x = np.arange(len(evalues))
        # Plots the energy spectrum using the eigenvalues from the "eigensolver" function:
        self.ax.scatter(x,evalues,s=10, color="purple")
        self.ax.grid(True, which="major",linestyle=":") # Adds a grid with dashed major lines.
        self.ax.axline((0, 0), slope=0, label="Zero-energy line",color="red") # Plots a horizontal zero-energy line.
        # Adds a legend, title, and axes labels, respectively:
        self.ax.legend(loc='upper left', prop={'size': 8})
        self.ax.set_title("Energy Spectrum Plot",fontname="Times New Roman")
        self.ax.set_xlabel("State Index",fontname="Times New Roman")
        self.ax.set_ylabel("Energy",fontname="Times New Roman")

        # Placeholder Phase Diagram code
        x = np.linspace(0, 2, 50) # t1 values
        y = np.linspace(0, 2, 50) # t2 values
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        # Function for the phase diagram:
        def energyGap(N, t1, t2, BC,phase):
            gaplist = []
            for r in phase:
                realT1 = float(r) * float(t2)  # Gets t1 from each phase entry (as the phase is t1/t2).
                H = Hamiltonian(int(N), realT1, float(t2), BC, self.T3.get(), self.T4.get(), self.customHops)
                eigenvals = np.linalg.eigvalsh(H)
                eigenvals = np.sort(eigenvals)
                gap = (eigenvals[N] - eigenvals[N - 1]) # Mid-gap for the 2N states, which is appended to a list.
                gaplist.append(gap)

            # Fills the topological / trivial regions of the phase diagram with an array of ratios:
            rarr = np.array(phase)
            # The axes for this diagram must be cleared again in this function, to prevent an overlay.
            self.ax2.clear()
            # Creates semi-opaque boxes for each region of the phase of the graph:
            self.ax2.fill_between(rarr, 0, max(gaplist), where=rarr < 1.0, color="green", alpha=0.12, label="Topological state")
            self.ax2.fill_between(rarr, 0, max(gaplist), where=rarr >= 1.0, color="yellow", alpha=0.12, label="Trivial state")
            # Plots the energy gap at each t1/t2 ratio:
            self.ax2.scatter(rarr, gaplist, s=10,color="purple")

            # Computes the current (t1/t2) point's actual gap, to accurately plot an "X" marker:
            global currentRatio
            currentRatio = float(self.T1.get()) / float(self.T2.get())
            # Current Hamiltonian specific to this point:
            Hcurr = Hamiltonian(int(N), float(self.T1.get()), float(self.T2.get()), BC,self.T3.get(), self.T4.get(), self.customHops)
            # Sort the eigenvalues of this point using np.linalg.eigvalsh.
            ev = np.sort(np.linalg.eigvalsh(Hcurr))
            # Determines the gap at the current t1/t2 ratio.
            currentGap = ev[N] - ev[N - 1]
            # Plots the current point, grid, phase transition line, title, labels and legend, respectively.
            self.ax2.scatter([currentRatio], [currentGap], marker='x', s=75, label='t1/t2 ratio',color="red")
            self.ax2.grid(True, which="major",linestyle=":")
            self.ax2.axvline(1, label="Phase transition",color="red")
            self.ax2.set_title("Phase diagram",fontname="Times New Roman")
            self.ax2.set_xlabel("t1/t2",fontname="Times New Roman")
            if currentRatio > 3:
                self.ax2.set_xlim(0,currentRatio + 1)
                self.ax2.legend(loc='lower right', prop={'size': 8})
            else:
                self.ax2.legend(loc='upper right', prop={'size': 8})
            self.ax2.set_ylabel("Energy Gap",fontname="Times New Roman")
            return

        # Varies the input grid on the size of the ratio (improves how the graph looks at larger t1/t2 ratios when t2 nears 0)
        if float(self.T1.get()) / float(self.T2.get()) <= 3:
            energyGap(self.N.get(),self.T1.get(),self.T2.get(), self.BoundCond.get(),phase=np.linspace(0.1,3, 50))
        else:
            energyGap(self.N.get(),self.T1.get(),self.T2.get(), self.BoundCond.get(),phase=np.linspace(0.1,float(self.T1.get()) / float(self.T2.get()) + 1, 50))

        # Code for the wavefunction visualisation graph:
        # Takes integer, positive values of the smallest eigenvalues (as instructed in the word document).
        normalise = int(np.argmin(abs(evalues)))
        # Creates a wavefunction of these points (wavefunction = |normalised eigenvalues|^2)
        wavefunction = np.abs(evectors[:, normalise])**2
        self.ax3.plot(range(2*self.N.get()), wavefunction, color="purple")
        self.ax3.grid(True, which="major",linestyle=":")
        self.ax3.set_title(f"Wavefunction Visualisation",fontname="Times New Roman")
        self.ax3.set_xlabel("Site index (i)",fontname="Times New Roman")
        self.ax3.set_ylabel(r'$|\phi|^2$',fontname="Times New Roman",fontsize="8")
        # Draws (updates) the canvases of all 3 graphs: s=10
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        return

    # Code to add new hopping parameter terms:
    def addTerm(self):
        # Allows amplitudes of 0.
        try:
            amp = float(self.amp_entry.get())
        except Exception:
            amp = 0.0
        # Fills the term dictionary variable, using the appropriate sub-variables:
        term = {'From': self.fromSublattice.get(),
            'To': self.toSublattice.get(),
            'Range': int(self.parameterRange.get()),
            'Amplitude': amp}
        # Creates a list of custom parameters if none exists:
        if self.customHops is None:
            self.customHops = []
            self.customHops.append(term)
            self.terms_listbox.insert('end', description(term))
        # Otherwise, the new parameter is appended onto a prior list:
        else:
            self.customHops.append(term)
            self.terms_listbox.insert('end', description(term))

    # Function for removing a variable that is selected in the "listbox":
    def removeSelected(self):
        # Defines the selected variable using the ".curselection()" function.
        sel = self.terms_listbox.curselection()
        if not sel:
            return
        deleteTerm = sel[0]
        # Deletes the selected variable(s):
        self.terms_listbox.delete(deleteTerm)
        if self.customHops is not None and 0 <= deleteTerm < len(self.customHops):
            del self.customHops[deleteTerm]
        if len(self.customHops) == 0:
            self.customHops = None

    # Function to clear all custom parameter terms:
    def clearTerms(self):
        self.terms_listbox.delete(0, 'end')
        self.customHops = None

# Default code for closing Tkinter GUI
window().mainloop()
