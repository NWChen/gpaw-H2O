import matplotlib.pyplot as plt
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase.units import fs
from gpaw import GPAW, PW

# Set up the initial structure of a water molecule
# shape (n, 3): [(x1,y1,z1), (x2,y2,z2), …]
water = Atoms('H2O', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

# Define a unit cell with enough vacuum around the water molecule
cell = 10.0
water.set_cell((cell, cell, cell))
water.center()

# Set up the calculator (GPAW) with desired parameters
# mode: Use plane-wave (PW) basis mode with cutoff of 300eV
# kpts: a 1x1x1 grid of k-points (to represent electron wavefunctions in k-space)
# txt: output file for calc data
calc = GPAW(mode=PW(300), xc='PBE', kpts=(1, 1, 1), txt='gpaw_water_md.txt', symmetry='off')
water.set_calculator(calc)

# Set initial momenta corresponding to T=300K
MaxwellBoltzmannDistribution(water, temperature_K=300)

# Set up the MD simulation with a 1fs timestep
dyn = VelocityVerlet(water, dt=1.0 * fs)
traj = Trajectory('water_md.traj', 'w', water)
dyn.attach(traj.write, interval=10)

# Run the MD simulation for N_STEPS steps
N_STEPS = 20
energies = []
for _ in range(N_STEPS):
    dyn.run(1)
    epot = water.get_potential_energy() / len(water)
    ekin = water.get_kinetic_energy() / len(water)
    etot = epot + ekin
    energies.append((epot, ekin, etot))

# Plot the energies
energies = list(zip(*energies))
plt.plot(energies[0], label='E_pot')
plt.plot(energies[1], label='E_kin')
plt.plot(energies[2], label='E_tot')

plt.xlabel('Timestep')
plt.ylabel('Energy per atom (eV)')
plt.legend()
plt.show()
