# Comprehensive Python Packages for High-Energy Physics and QGP Visualization

**Author:** Deiríkr Jaiusadastra Afrauthïhinngreygaärd
**Date:** December 16, 2025
**Project:** QGP Light-Ion

This document provides a comprehensive overview of Python packages relevant for high-energy physics (HEP) and Quark-Gluon Plasma (QGP) visualization and computation, focusing on packages that can generate precomputed data for TikZ/pgfplots figures.

---

## Table of Contents

1. [Scientific Visualization](#1-scientific-visualization)
2. [HEP-Specific Packages](#2-hep-specific-packages)
3. [Physics Simulation & Performance](#3-physics-simulation--performance)
4. [Multi-dimensional Analysis](#4-multi-dimensional-analysis)
5. [Statistical Analysis](#5-statistical-analysis)
6. [Data Formats & Management](#6-data-formats--management)
7. [Heavy-Ion Collision Specific](#7-heavy-ion-collision-specific)
8. [Installation Summary](#8-installation-summary)

---

## 1. Scientific Visualization

### 1.1 Matplotlib (Foundation)

**Purpose:** Python's de facto plotting library for static, animated, and interactive 2D plots. The mplot3d toolkit provides basic 3D visualization capabilities.

**QGP Visualization Use:**
- Creating publication-quality 2D plots for R_AA vs pT, flow coefficients v_n
- Basic 3D surface plots for energy density profiles
- Generating static plots that can be exported as data for TikZ/pgfplots

**Example:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Generate R_AA data
pt = np.logspace(0, 2, 50)  # 1-100 GeV
raa = 0.5 + 0.3 * np.exp(-pt/20)  # Simple model

# Save data for TikZ
np.savetxt('data/jet_quenching/raa_data.dat',
           np.column_stack([pt, raa]),
           header='pT[GeV] RAA')
```

**Installation:**
```bash
pip install matplotlib
```

### 1.2 Plotly

**Purpose:** Interactive, web-based visualization library with extensive 2D/3D capabilities including surface plots, volume rendering, and isosurfaces.

**QGP Visualization Use:**
- Interactive 3D visualizations of nuclear geometry (Woods-Saxon profiles)
- Dynamic exploration of energy density evolution in spacetime
- Real-time parameter adjustment for physics models
- Exporting static images from interactive plots

**Example:**
```python
import plotly.graph_objects as go

# Create 3D surface for energy density
fig = go.Figure(data=[go.Surface(z=energy_density_grid, x=x_coords, y=y_coords)])
fig.update_layout(title='Energy Density Profile (O-O Collision)')

# Export to static image for inclusion in paper
fig.write_image('figures/energy_density_3d.png')

# Or export data for TikZ/pgfplots
np.savetxt('data/spacetime/energy_density_2d.dat',
           energy_density_grid)
```

**Installation:**
```bash
pip install plotly
# For static image export:
pip install kaleido
```

### 1.3 PyVista

**Purpose:** Pythonic interface to VTK (Visualization Toolkit) for 3D scientific data visualization. Handles meshes, grids, and scalar fields with high performance.

**QGP Visualization Use:**
- Advanced 3D visualization of fireball evolution
- Volume rendering of QGP energy density distributions
- Particle trajectory visualization in heavy-ion collisions
- Generating high-quality 3D renderings for publication

**Example:**
```python
import pyvista as pv
import numpy as np

# Create 3D grid for QGP fireball
grid = pv.UniformGrid()
grid.dimensions = (100, 100, 100)
grid.spacing = (0.1, 0.1, 0.1)  # fm spacing

# Add energy density scalar field
grid['energy_density'] = energy_density_function(grid.points)

# Render and export
plotter = pv.Plotter(off_screen=True)
plotter.add_volume(grid, opacity='sigmoid', cmap='hot')
plotter.screenshot('figures/fireball_3d.png')

# Extract 2D slice for TikZ
slice_2d = grid.slice(normal='z')
slice_2d.save('data/spacetime/fireball_slice.vtk')
```

**Installation:**
```bash
pip install pyvista
```

### 1.4 Mayavi

**Purpose:** General-purpose 3D scientific data visualization based on VTK. Part of the Enthought suite with focus on interactive scene creation.

**QGP Visualization Use:**
- Legacy code compatibility for existing QGP visualizations
- Interactive 3D scene creation for nuclear collision geometry
- Integration with other Enthought tools

**Note:** PyVista is generally recommended over Mayavi for new projects due to better documentation, easier API, and wider adoption.

**Installation:**
```bash
pip install mayavi
# May require Qt backend:
pip install PyQt5
```

### 1.5 VisPy

**Purpose:** High-performance GPU-accelerated 2D/3D visualization using OpenGL, designed for very large datasets (1M+ points).

**QGP Visualization Use:**
- Real-time visualization of millions of particle trajectories
- GPU-accelerated rendering of large-scale hydrodynamic simulations
- Interactive exploration of high-resolution spacetime evolution

**Example:**
```python
from vispy import scene
import numpy as np

# Visualize 1M+ hadrons from hydrodynamic output
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Plot particle positions
scatter = scene.visuals.Markers()
scatter.set_data(hadron_positions, edge_color=None,
                 face_color='red', size=2)
view.add(scatter)

# Export screenshot
img = canvas.render()
# Convert to image format and save
```

**Installation:**
```bash
pip install vispy
```

### 1.6 Bokeh

**Purpose:** Interactive visualization library for web browsers with focus on large datasets and streaming data.

**QGP Visualization Use:**
- Dashboard-style presentations of multiple QGP observables
- Real-time data streaming from simulation outputs
- Interactive parameter exploration tools

**Installation:**
```bash
pip install bokeh
```

---

## 2. HEP-Specific Packages

### 2.1 Uproot

**Purpose:** Pure-Python ROOT file reader/writer with simple, Pythonic interface. Produces NumPy arrays directly without requiring ROOT installation.

**QGP Visualization Use:**
- Reading experimental data from ALICE, ATLAS, CMS ROOT files
- Extracting histograms and ntuples for analysis
- Converting ROOT data to formats suitable for Python analysis

**Example:**
```python
import uproot
import numpy as np

# Read ALICE data (example)
with uproot.open('alice_data.root') as file:
    # Extract flow coefficients
    tree = file['FlowTree']
    v2 = tree['v2'].array(library='np')
    centrality = tree['centrality'].array(library='np')

    # Save for TikZ visualization
    np.savetxt('data/flow/alice_v2_data.dat',
               np.column_stack([centrality, v2]),
               header='centrality v2')
```

**Installation:**
```bash
pip install uproot
```

### 2.2 Awkward Array

**Purpose:** NumPy-like arrays with irregular/nested structure, essential for HEP event data with variable-length lists.

**QGP Visualization Use:**
- Handling particle-level data with variable numbers of particles per event
- Efficient manipulation of jet constituents
- Processing complex event structures from simulations

**Example:**
```python
import awkward as ak
import numpy as np

# Example: Jets with variable numbers of constituents
events = ak.Array([
    {'jets': [{'pt': 50, 'eta': 0.5}, {'pt': 30, 'eta': -0.3}]},
    {'jets': [{'pt': 80, 'eta': 0.0}]},
    {'jets': [{'pt': 45, 'eta': 0.8}, {'pt': 35, 'eta': -0.5}, {'pt': 20, 'eta': 1.0}]}
])

# Calculate average jet pT per event
avg_pt = ak.mean(events.jets.pt, axis=1)

# Convert to regular numpy for saving
np.savetxt('data/jet_quenching/avg_jet_pt.dat',
           ak.to_numpy(avg_pt))
```

**Installation:**
```bash
pip install awkward
```

### 2.3 Hist

**Purpose:** Analyst-friendly histogramming library built on boost-histogram, designed for HEP workflows with plotting capabilities.

**QGP Visualization Use:**
- Creating sophisticated histograms of particle distributions
- Analyzing differential cross-sections
- Generating histogram data for publication plots

**Example:**
```python
from hist import Hist
import hist
import numpy as np

# Create histogram of charged particle multiplicity
h = Hist.new.Reg(100, 0, 500, name='Nch', label='$N_{ch}$').Double()

# Fill with simulated data
multiplicity_data = np.random.poisson(200, 10000)
h.fill(multiplicity_data)

# Export binned data for TikZ
centers = h.axes[0].centers
values = h.values()
errors = np.sqrt(values)

np.savetxt('data/multiplicity/nch_dist.dat',
           np.column_stack([centers, values, errors]),
           header='Nch counts error')
```

**Installation:**
```bash
pip install hist
```

### 2.4 boost-histogram

**Purpose:** Python bindings for C++14 Boost::Histogram library, providing high-performance histogram filling.

**QGP Visualization Use:**
- Core histogramming engine (Hist is built on top of this)
- Maximum performance for large datasets
- Multi-dimensional histogram creation

**Installation:**
```bash
pip install boost-histogram
```

### 2.5 mplhep

**Purpose:** Matplotlib wrapper for HEP-specific plotting styles (ATLAS, CMS, ALICE, LHCb) with prebinned histograms.

**QGP Visualization Use:**
- Creating plots in official LHC experiment styles
- Consistent formatting across all figures
- Quick histogram plotting with HEP conventions

**Example:**
```python
## Common plotting style (Recommended)

```python
from cern_analysis_common.plotting import set_hep_style, add_experiment_label
import matplotlib.pyplot as plt

set_hep_style("ALICE")
plt.plot([1, 2, 3], [1, 4, 9])
add_experiment_label("ALICE", data=True)
plt.show()
```

## Manual Style
import mplhep as hep
import matplotlib.pyplot as plt

plt.style.use(hep.style.ALICE)


# Plot R_AA with ALICE style
fig, ax = plt.subplots()
ax.errorbar(pt, raa, yerr=raa_err, fmt='o', label='O-O')
ax.set_xlabel(r'$p_T$ [GeV]')
ax.set_ylabel(r'$R_{AA}$')
hep.alice.label(ax=ax, data=True, lumi='Pb-Pb $\\sqrt{s_{NN}}$ = 5.02 TeV')
plt.savefig('figures/raa_alice_style.pdf')

# Export data
np.savetxt('data/jet_quenching/raa_alice.dat',
           np.column_stack([pt, raa, raa_err]))
```

**Installation:**
```bash
pip install mplhep
```

### 2.6 pyhepmc

**Purpose:** Easy-to-use Python bindings for HepMC3, the standard Monte Carlo event record format in HEP.

**QGP Visualization Use:**
- Reading/writing Monte Carlo generator outputs (PYTHIA, HERWIG)
- Interfacing with event generators for baseline pp comparisons
- Processing particle-level predictions

**Example:**
```python
import pyhepmc

# Read event generator output
with pyhepmc.open('pythia_output.hepmc') as f:
    for event in f:
        # Extract final-state charged particles
        charged_particles = [p for p in event.particles
                           if p.status == 1 and abs(p.pid) in [211, 321, 2212]]

        # Calculate observables
        pt_spectrum = [p.momentum.pt() for p in charged_particles]
```

**Installation:**
```bash
pip install pyhepmc
```

### 2.7 Particle

**Purpose:** Provides access to particle physics data from the Particle Data Group (PDG), including masses, widths, charges, and quantum numbers.

**QGP Visualization Use:**
- Looking up particle properties for calculations
- Identifying particles by PDG ID codes
- Ensuring correct particle masses in simulations

**Example:**
```python
from particle import Particle

# Get particle properties
proton = Particle.from_pdgid(2212)
print(f"Proton mass: {proton.mass} MeV")

# Find all kaons
kaons = Particle.findall(lambda p: 'K' in p.name)

# Use in calculations
m_kaon = Particle.from_pdgid(321).mass / 1000  # Convert to GeV
```

**Installation:**
```bash
pip install particle
```

---

## 3. Physics Simulation & Performance

### 3.1 NumPy

**Purpose:** Fundamental package for numerical computing in Python, providing N-dimensional array objects and mathematical functions.

**QGP Visualization Use:**
- Core array operations for all physics calculations
- Linear algebra for hydrodynamic simulations
- Random number generation for Monte Carlo methods

**Example:**
```python
import numpy as np

# Bjorken energy density calculation
def bjorken_energy_density(dET_deta, tau0, area):
    """Calculate Bjorken energy density [GeV/fm^3]"""
    return dET_deta / (tau0 * area)

# Generate energy density evolution
tau = np.linspace(0.6, 10, 100)  # fm/c
epsilon = 14.0 * (0.6/tau)**(4/3)  # Simple cooling model

np.savetxt('data/spacetime/energy_density_evolution.dat',
           np.column_stack([tau, epsilon]),
           header='tau[fm/c] epsilon[GeV/fm3]')
```

**Installation:**
```bash
pip install numpy
```

### 3.2 SciPy

**Purpose:** Scientific computing library with optimization, integration, interpolation, FFT, signal processing, and special functions.

**QGP Visualization Use:**
- Numerical integration for Woods-Saxon profiles
- Bessel functions for flow calculations
- Interpolation of simulation data
- Solving differential equations

**Example:**
```python
import numpy as np
from scipy.special import iv as bessel_i
from scipy.integrate import quad

# Flow coefficient calculation
def azimuthal_distribution(phi, v2, v3, psi2=0, psi3=0):
    """Particle azimuthal distribution with flow"""
    return 1 + 2*v2*np.cos(2*(phi - psi2)) + 2*v3*np.cos(3*(phi - psi3))

# Woods-Saxon nuclear density
def woods_saxon(r, R0, a):
    """Nuclear density profile"""
    return 1 / (1 + np.exp((r - R0) / a))

# Normalization integral
norm, _ = quad(lambda r: r**2 * woods_saxon(r, 2.5, 0.5), 0, np.inf)
```

**Installation:**
```bash
pip install scipy
```

### 3.3 Numba

**Purpose:** JIT (just-in-time) compiler that translates Python to optimized machine code, providing 10-150x speedups for numerical code.

**QGP Visualization Use:**
- Accelerating Monte Carlo simulations (Glauber model)
- Speeding up particle trajectory calculations
- GPU acceleration with CUDA for hydrodynamics
- Fast loop-heavy calculations

**Example:**
```python
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def glauber_monte_carlo(n_events, n_nucleons_A, n_nucleons_B):
    """Fast Monte Carlo Glauber simulation"""
    npart = np.zeros(n_events)
    ncoll = np.zeros(n_events)

    for i in prange(n_events):
        # Sample nucleon positions
        positions_A = sample_nucleons(n_nucleons_A)
        positions_B = sample_nucleons(n_nucleons_B)

        # Calculate participants and collisions
        npart[i], ncoll[i] = count_collisions(positions_A, positions_B)

    return npart, ncoll

# Run simulation
npart, ncoll = glauber_monte_carlo(100000, 16, 16)  # O-O collisions

# Save results
np.savetxt('data/nuclear_geometry/glauber_results.dat',
           np.column_stack([npart, ncoll]),
           header='Npart Ncoll')
```

**Performance:** 10-150x faster than pure Python, 8-12x with GPU for large ensembles.

**Installation:**
```bash
pip install numba
```

### 3.4 JAX

**Purpose:** Google's library combining automatic differentiation, JIT compilation, and GPU/TPU acceleration with a NumPy-like API.

**QGP Visualization Use:**
- Differentiable physics simulations
- Gradient-based optimization of model parameters
- GPU-accelerated hydrodynamics
- Machine learning approaches to QGP analysis

**Example:**
```python
import jax.numpy as jnp
from jax import grad, jit, vmap

@jit
def energy_loss_model(pt, qhat, L):
    """BDMPS-Z energy loss model"""
    return pt - jnp.sqrt(qhat * L * pt)

# Gradient for optimization
grad_qhat = grad(lambda q: jnp.sum((raa_model(pt, q) - raa_data)**2))

# Vectorize over multiple events
energy_loss_batch = vmap(energy_loss_model, in_axes=(0, None, None))
```

**Note:** JAX vs Numba comparison presented at PyCon UK 2025 - JAX offers auto-differentiation and easier CPU/GPU/TPU portability, while Numba has simpler syntax for numerical acceleration.

**Installation:**
```bash
pip install jax jaxlib
# For GPU support:
pip install jax[cuda12]
```

---

## 4. Multi-dimensional Analysis

### 4.1 scikit-learn

**Purpose:** Machine learning library with comprehensive tools for classification, regression, clustering, and dimensionality reduction.

**QGP Visualization Use:**
- PCA (Principal Component Analysis) for reducing high-dimensional data
- t-SNE for visualizing event topology in 2D/3D
- Clustering for identifying different collision geometries
- Feature selection for identifying key QGP signatures

**Example:**
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Example: Reduce event-by-event observables to 2D
# Each event has many features: v2, v3, v4, eccentricities, etc.
event_features = np.random.randn(10000, 20)  # 10k events, 20 features

# PCA for dimensionality reduction
pca = PCA(n_components=2)
events_2d = pca.fit_transform(event_features)

# t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30)
events_tsne = tsne.fit_transform(event_features)

# Save for visualization
np.savetxt('data/analysis/events_pca.dat', events_2d,
           header='PC1 PC2')
np.savetxt('data/analysis/events_tsne.dat', events_tsne,
           header='dim1 dim2')
```

**Installation:**
```bash
pip install scikit-learn
```

### 4.2 umap-learn

**Purpose:** UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction, often faster and better at preserving global structure than t-SNE.

**QGP Visualization Use:**
- Event clustering and classification
- Visualizing high-dimensional parameter spaces
- Identifying phase transitions in phase diagrams
- Faster than t-SNE for large datasets

**Example:**
```python
import umap
import numpy as np

# Reduce high-dimensional event features
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(event_features)

# Visualize different collision systems
np.savetxt('data/analysis/events_umap.dat',
           np.column_stack([embedding[:, 0], embedding[:, 1], system_type]),
           header='dim1 dim2 system')
```

**Installation:**
```bash
pip install umap-learn
```

---

## 5. Statistical Analysis

### 5.1 iminuit

**Purpose:** Jupyter-friendly Python interface for CERN's Minuit2 C++ library, designed for maximum-likelihood and least-squares fitting.

**QGP Visualization Use:**
- Fitting physics models to experimental data
- Parameter extraction (qhat, eta/s, etc.)
- Error estimation with HESSE and MINOS
- Chi-square minimization

**Example:**
```python
from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np

# Define R_AA model
def raa_model(pt, qhat, L):
    """Energy loss model for R_AA"""
    delta_E = np.sqrt(qhat * L * pt)
    return (pt - delta_E) / pt

# Experimental data
pt_data = np.array([5, 10, 20, 40, 80])
raa_data = np.array([0.2, 0.25, 0.35, 0.45, 0.55])
raa_err = np.array([0.02, 0.03, 0.04, 0.05, 0.06])

# Create cost function
least_squares = LeastSquares(pt_data, raa_data, raa_err, raa_model)

# Minimize
m = Minuit(least_squares, qhat=2.0, L=5.0)
m.migrad()  # Find minimum
m.hesse()   # Compute uncertainties

print(f"Best fit: qhat = {m.values['qhat']:.2f} ± {m.errors['qhat']:.2f} GeV²/fm")

# Generate fitted curve
pt_fit = np.linspace(1, 100, 100)
raa_fit = raa_model(pt_fit, *m.values)

np.savetxt('data/jet_quenching/raa_fit.dat',
           np.column_stack([pt_fit, raa_fit]),
           header='pT[GeV] RAA_fit')
```

**Performance:** 10x faster than RooFit when combined with Numba JIT compilation.

**Installation:**
```bash
pip install iminuit
```

### 5.2 zfit

**Purpose:** Scalable Pythonic fitting framework built on TensorFlow, designed as a modern alternative to RooFit with GPU support.

**QGP Visualization Use:**
- Complex likelihood fits with multiple parameters
- GPU-accelerated fitting for large datasets
- Building sophisticated statistical models
- Extensible framework for custom PDFs

**Example:**
```python
import zfit
import numpy as np

# Define observable
pt = zfit.Space('pt', limits=(0, 100))

# Define model (e.g., Tsallis distribution)
n = zfit.Parameter('n', 6.0, 1.0, 20.0)
T = zfit.Parameter('T', 0.15, 0.1, 0.5)

# Custom PDF for particle pT spectrum
class TsallisPDF(zfit.pdf.ZPDF):
    _PARAMS = ['n', 'T']

    def _unnormalized_pdf(self, x):
        pt = x.unstack_x()
        return pt * (1 + pt / (self.params['n'] * self.params['T']))**(-self.params['n'])

model = TsallisPDF(obs=pt, n=n, T=T)

# Fit to data
data = zfit.Data.from_numpy(obs=pt, array=pt_spectrum_data)
nll = zfit.loss.UnbinnedNLL(model=model, data=data)
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

print(result)
```

**Installation:**
```bash
pip install zfit
```

### 5.3 uncertainties

**Purpose:** Transparent calculations with uncertainties using automatic error propagation.

**QGP Visualization Use:**
- Propagating experimental uncertainties through calculations
- Calculating derived quantities with errors
- Systematic uncertainty combination

**Example:**
```python
from uncertainties import ufloat
import uncertainties.umath as umath
import numpy as np

# Experimental inputs with uncertainties
dET_deta = ufloat(450, 50)  # GeV
tau0 = ufloat(0.6, 0.1)  # fm/c
area = ufloat(15, 2)  # fm²

# Automatic error propagation
epsilon_Bj = dET_deta / (tau0 * area)
print(f"Bjorken energy density: {epsilon_Bj:.2f} GeV/fm³")

# Temperature from energy density (Stefan-Boltzmann)
# ε = (π²/30) * g * T⁴
g = 37  # Effective degrees of freedom
T = (30 * epsilon_Bj / (np.pi**2 * g))**(1/4)
print(f"Initial temperature: {T:.3f} GeV")

# Save results
with open('data/analysis/thermodynamic_quantities.txt', 'w') as f:
    f.write(f"epsilon_Bj = {epsilon_Bj}\n")
    f.write(f"T_initial = {T}\n")
```

**Installation:**
```bash
pip install uncertainties
```

---

## 6. Data Formats & Management

### 6.1 h5py

**Purpose:** Pythonic interface to HDF5 binary data format for storing huge numerical datasets with efficient slicing.

**QGP Visualization Use:**
- Storing large hydrodynamic simulation outputs
- Multi-terabyte datasets for event-by-event analysis
- Efficient random access to subset of data
- Hierarchical organization of simulation results

**Example:**
```python
import h5py
import numpy as np

# Create HDF5 file for simulation data
with h5py.File('simulation_data.h5', 'w') as f:
    # Create groups for organization
    geom = f.create_group('geometry')
    flow = f.create_group('flow')
    jets = f.create_group('jets')

    # Store Woods-Saxon profiles
    r = np.linspace(0, 10, 1000)
    rho_O = woods_saxon_profile(r, nucleus='O16')
    rho_Ne = woods_saxon_profile(r, nucleus='Ne20')

    geom.create_dataset('r', data=r)
    geom.create_dataset('O16_profile', data=rho_O)
    geom.create_dataset('Ne20_profile', data=rho_Ne)

    # Store event-by-event flow
    flow.create_dataset('v2', data=v2_array)
    flow.create_dataset('v3', data=v3_array)
    flow.attrs['system'] = 'O-O'
    flow.attrs['sqrt_sNN'] = 5020  # GeV

# Read back specific data
with h5py.File('simulation_data.h5', 'r') as f:
    v2 = f['flow/v2'][:]

    # Export for TikZ
    centrality = np.arange(len(v2))
    np.savetxt('data/flow/v2_vs_cent.dat',
               np.column_stack([centrality, v2]),
               header='centrality v2')
```

**Installation:**
```bash
pip install h5py
```

### 6.2 pandas

**Purpose:** High-performance data structures (DataFrame) for tabular data with powerful indexing and manipulation.

**QGP Visualization Use:**
- Organizing experimental results in tables
- Merging data from different sources
- Statistical analysis and grouping
- CSV/Excel I/O for collaboration

**Example:**
```python
import pandas as pd
import numpy as np

# Create DataFrame for R_AA measurements
df = pd.DataFrame({
    'system': ['Pb-Pb', 'Pb-Pb', 'Pb-Pb', 'O-O', 'O-O', 'O-O'],
    'centrality': ['0-10%', '10-20%', '20-40%', '0-10%', '10-20%', '20-40%'],
    'pt_min': [5, 5, 5, 5, 5, 5],
    'pt_max': [10, 10, 10, 10, 10, 10],
    'raa': [0.18, 0.25, 0.35, 0.45, 0.55, 0.65],
    'stat_err': [0.02, 0.03, 0.03, 0.05, 0.06, 0.07],
    'sys_err': [0.04, 0.05, 0.06, 0.08, 0.09, 0.10]
})

# Group by system
grouped = df.groupby('system')
for name, group in grouped:
    # Save each system separately
    group.to_csv(f'data/jet_quenching/raa_{name.replace("-", "")}.csv',
                 index=False)

    # Also save as .dat for TikZ
    np.savetxt(f'data/jet_quenching/raa_{name.replace("-", "")}.dat',
               group[['pt_min', 'raa', 'stat_err']].values,
               header='pT RAA stat_err')
```

**Installation:**
```bash
pip install pandas
```

### 6.3 xarray

**Purpose:** N-dimensional labeled arrays and datasets, extending pandas to multiple dimensions. Ideal for scientific data (climate, geospatial, physics).

**QGP Visualization Use:**
- Multi-dimensional hydrodynamic grids (x, y, z, t, event)
- Coordinate-aware operations
- NetCDF/HDF5 integration
- Efficient handling with Dask for parallelization

**Example:**
```python
import xarray as xr
import numpy as np

# Create multi-dimensional energy density dataset
# Dimensions: (event, time, x, y, z)
n_events = 1000
n_t = 50
nx = ny = nz = 50

energy_density = np.random.exponential(5, size=(n_events, n_t, nx, ny, nz))

# Create DataArray with labeled coordinates
ds = xr.Dataset(
    {
        'energy_density': (['event', 'time', 'x', 'y', 'z'], energy_density),
    },
    coords={
        'event': np.arange(n_events),
        'time': np.linspace(0.6, 10, n_t),  # fm/c
        'x': np.linspace(-5, 5, nx),  # fm
        'y': np.linspace(-5, 5, ny),  # fm
        'z': np.linspace(-5, 5, nz),  # fm
    }
)

# Add metadata
ds.attrs['system'] = 'O-O'
ds.attrs['sqrt_sNN'] = 5020
ds['energy_density'].attrs['units'] = 'GeV/fm³'

# Save to NetCDF
ds.to_netcdf('data/spacetime/energy_density_4d.nc')

# Extract 2D slice at z=0, averaged over events
slice_2d = ds.sel(z=0, method='nearest').mean(dim='event')

# Export for TikZ plotting
for i, t in enumerate([0.6, 1.0, 2.0, 5.0]):  # Select specific times
    data_t = slice_2d.sel(time=t, method='nearest')['energy_density']

    # Save as 2D grid
    np.savetxt(f'data/spacetime/energy_2d_t{t:.1f}.dat',
               data_t.values)
```

**When to use:** Multi-dimensional scientific data (3D+time), NetCDF/GRIB/HDF5 files, need for labeled dimensions.

**Installation:**
```bash
pip install xarray
# For NetCDF support:
pip install netcdf4
# For parallel processing:
pip install dask
```

---

## 7. Heavy-Ion Collision Specific

### 7.1 hic (Duke-QCD)

**Purpose:** Collection of Python modules for analyzing heavy-ion collision simulation data, calculating flow coefficients and eccentricities.

**QGP Visualization Use:**
- Computing v_n flow harmonics
- Calculating initial condition eccentricities ε_n
- Event-plane angle determination
- Participant plane calculations

**Example:**
```python
# Note: Actual implementation depends on hic package structure
import numpy as np

# Typically would import from hic package
# Example based on typical flow analysis

def calculate_flow_coefficients(particle_data):
    """Calculate v2, v3 from particle azimuthal distribution"""
    phi = particle_data['phi']
    weights = particle_data['pt']  # pT weighting

    v2 = np.average(np.cos(2*phi), weights=weights)
    v3 = np.average(np.cos(3*phi), weights=weights)

    return v2, v3

# Process events
v2_array = []
v3_array = []
for event in event_generator:
    v2, v3 = calculate_flow_coefficients(event)
    v2_array.append(v2)
    v3_array.append(v3)

# Save results
np.savetxt('data/flow/flow_coefficients.dat',
           np.column_stack([v2_array, v3_array]),
           header='v2 v3')
```

**Installation:**
```bash
pip install hic
# May require:
pip install numpy scipy
```

### 7.2 SPARKX

**Purpose:** Open-source package for analyzing simulation data from heavy-ion collision experiments, supporting multiple formats (OSCAR2013, SMASH, JETSCAPE).

**QGP Visualization Use:**
- Reading outputs from transport models
- Kinematic analysis of particle distributions
- Format conversion between different simulation codes
- Statistical analysis of collision data

**Installation:**
```bash
pip install sparkx
```

### 7.3 NuclearConfectionery (2025)

**Purpose:** Modular framework for simulating full dynamical evolution of relativistic heavy-ion collisions with advanced hydrodynamics (ccake 2.0).

**QGP Visualization Use:**
- Running complete event-by-event simulations
- Hydrodynamic evolution with conserved charges (B, S, Q)
- Multi-stage collision dynamics (RHIC to LHC energies)
- Generating realistic QGP evolution data

**Note:** This is a multi-language framework (C++, Python, Bash, SQLite).

**Installation:**
```bash
# Check GitHub repository for installation instructions
# Likely requires compilation of C++ components
```

### 7.4 MCGlauber

**Purpose:** Monte-Carlo Glauber model for geometrically simulating nuclear collisions.

**QGP Visualization Use:**
- Initial state geometry calculations
- Npart (participants) and Ncoll (collisions) distributions
- Impact parameter sampling
- Eccentricity calculations for flow

**Example:**
```python
# Pseudo-code based on typical Glauber implementations
import mcglauber  # Hypothetical import

# Setup collision
collision = mcglauber.Collision(
    projectile='O16',
    target='O16',
    sqrt_s=5020  # GeV
)

# Run Monte Carlo
results = collision.run(n_events=100000)

# Extract distributions
npart = results['npart']
ncoll = results['ncoll']
ecc2 = results['ecc2']  # Eccentricity

# Save for analysis
np.savetxt('data/nuclear_geometry/glauber_distributions.dat',
           np.column_stack([npart, ncoll, ecc2]),
           header='Npart Ncoll epsilon2')
```

**Installation:**
```bash
# Check repository for specific installation
pip install mcglauber  # If available on PyPI
```

---

## 8. Installation Summary

### Essential Core Stack

```bash
# Scientific computing foundation
pip install numpy scipy matplotlib

# HEP data analysis
pip install uproot awkward hist mplhep particle

# Performance
pip install numba

# Data management
pip install pandas h5py

# Statistical fitting
pip install iminuit
```

### Extended Visualization Stack

```bash
# Interactive and 3D
pip install plotly pyvista

# For plotly static export
pip install kaleido

# GPU-accelerated (optional)
pip install vispy
```

### Advanced Analysis Stack

```bash
# Machine learning and dimensionality reduction
pip install scikit-learn umap-learn

# Multi-dimensional data
pip install xarray netcdf4

# Modern fitting framework
pip install zfit

# Uncertainty propagation
pip install uncertainties
```

### GPU Acceleration (Optional)

```bash
# JAX for GPU/TPU
pip install jax jaxlib
# For CUDA 12:
pip install jax[cuda12]
```

### Heavy-Ion Specific

```bash
# Duke-QCD tools
pip install hic

# Analysis framework
pip install sparkx

# PDG data
pip install particle
```

### Complete Environment Setup

For a complete environment, consider using a `requirements.txt` file:

```txt
# requirements.txt for QGP Light-Ion Project

# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0

# HEP-specific
uproot>=5.0.0
awkward>=2.0.0
hist>=2.7.0
boost-histogram>=1.4.0
mplhep>=0.3.0
pyhepmc>=2.13.0
particle>=0.23.0

# Visualization
plotly>=5.17.0
kaleido>=0.2.0
pyvista>=0.42.0

# Performance
numba>=0.58.0

# Data management
pandas>=2.0.0
h5py>=3.9.0
xarray>=2023.1.0
netcdf4>=1.6.0

# Statistical analysis
iminuit>=2.24.0
zfit>=0.16.0
uncertainties>=3.1.0

# Machine learning
scikit-learn>=1.3.0
umap-learn>=0.5.0

# Heavy-ion specific
# hic  # Install manually if needed
# sparkx
```

Install all at once:
```bash
pip install -r requirements.txt
```

### Using Virtual Environments (Recommended)

```bash
# Create virtual environment
python3 -m venv qgp-env

# Activate (macOS/Linux)
source qgp-env/bin/activate

# Activate (Windows)
# qgp-env\Scripts\activate

# Install packages
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Using Conda (Alternative)

```bash
# Create conda environment
conda create -n qgp-analysis python=3.11

# Activate
conda activate qgp-analysis

# Install from conda-forge where possible
conda install -c conda-forge numpy scipy matplotlib pandas h5py xarray

# Install HEP packages with pip
pip install uproot awkward hist mplhep particle iminuit

# Deactivate
conda deactivate
```

---

## Usage for This Project

Based on the existing code in `src/`, the current project uses:

- **NumPy** - Core calculations
- **SciPy** - Special functions (Bessel), integration
- **Matplotlib** (implied) - For any quick visualization checks

### Recommended Additions for Enhanced Capabilities

1. **Numba** - Speed up Monte Carlo Glauber simulations and particle sampling
2. **h5py** - Store large event-by-event datasets
3. **pandas** - Organize multi-system comparisons (Pb-Pb vs O-O vs Ne-Ne)
4. **iminuit** - Fit models to experimental data
5. **mplhep** - Create figures in ALICE style
6. **uncertainties** - Proper error propagation in calculations

### Example Integration into Existing Code

```python
# In qgp_physics.py, add Numba acceleration:
from numba import jit

@jit(nopython=True)
def sample_nucleon_positions_fast(n_nucleons, R0, a):
    """Faster nucleon sampling with Numba"""
    positions = np.zeros((n_nucleons, 3))
    for i in range(n_nucleons):
        # Rejection sampling for Woods-Saxon
        # ... implementation
    return positions
```

### Data Export for TikZ/pgfplots

All packages can export data using NumPy's `savetxt`:

```python
import numpy as np

# Standard format for pgfplots
np.savetxt('data/output.dat',
           np.column_stack([x, y, y_err]),
           header='x y y_err',
           fmt='%.6e')
```

This creates space-separated columns that pgfplots can read directly:

```latex
\begin{tikzpicture}
\begin{axis}[xlabel=$x$, ylabel=$y$]
\addplot[error bars/.cd, y dir=both, y explicit]
    table[x=x, y=y, y error=y_err] {data/output.dat};
\end{axis}
\end{tikzpicture}
```

---

## Conclusion

This comprehensive guide covers the essential Python packages for high-energy physics and QGP visualization, organized by category. The packages range from core scientific computing (NumPy, SciPy) to HEP-specific tools (Uproot, Awkward) to advanced visualization (PyVista, Plotly) and statistical analysis (iminuit, zfit).

**Key Takeaways:**

1. **For data generation:** NumPy, SciPy, Numba provide the computational foundation
2. **For HEP workflows:** Uproot, Awkward, Hist, mplhep integrate with experiment data
3. **For visualization:** Matplotlib (2D), Plotly/PyVista (3D interactive), VisPy (GPU)
4. **For fitting:** iminuit (fast, Jupyter-friendly), zfit (GPU, TensorFlow-based)
5. **For data management:** pandas (tabular), h5py (large files), xarray (multi-D)
6. **For heavy-ion physics:** hic, SPARKX, MCGlauber for collision-specific analysis

All packages support exporting data in formats compatible with TikZ/pgfplots for publication-quality LaTeX figures.

---

## Sources

- [Scikit-HEP Project](https://scikit-hep.org/)
- [IRIS-HEP Institute](https://iris-hep.org/)
- [Uproot Documentation](https://iris-hep.org/projects/uproot.html)
- [Awkward Array Project](https://iris-hep.org/projects/awkward.html)
- [Duke-QCD hic Package](http://qcd.phy.duke.edu/hic/index.html)
- [SPARKX Package](https://www.researchgate.net/publication/389786549_SPARKX_A_Software_Package_for_Analyzing_Relativistic_Kinematics_in_Collision_Experiments)
- [NuclearConfectionery Framework](https://arxiv.org/html/2511.22852)
- [PyVista Tutorial SciPy 2025](https://cfp.scipy.org/scipy2025/talk/MHNTAD/)
- [Best 3D Scientific Visualization](https://www.epsilonforge.com/post/best-3d-scientific-visualization/)
- [iminuit Documentation](https://scikit-hep.org/iminuit/)
- [zfit: Scalable Pythonic Fitting](https://www.sciencedirect.com/science/article/pii/S2352711019303851)
- [Xarray Documentation](https://xarray.dev/)
- [Numba Performance Guide](https://numba.pydata.org/)
- [Top 10 Python Visualization Libraries 2025](https://reflex.dev/blog/2025-01-27-top-10-data-visualization-libraries/)
