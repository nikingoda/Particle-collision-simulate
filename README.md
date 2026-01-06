# Particle collision simulation

## 1.Introduction:
- This project is a high-performance simulation of molecule collision in a box.
- It is used to simulate collision of about 1000-10000 particles with density of 0.7-0.9 throughout 1000 steps.
- Parallel computing techniques are applied (using OpenMP in particular) to optimize the runtime about 10 times faster than sequential implement.

## 2.How to run:
### Compile:
Run the following command to compile the code:

```bash
make
```
### Run the Simulation:
After compiling, run:
```bash
./sim <test> <number_of_threads>
```
- ``<test>``: Input file (initialize the molecule) (e.g: ``test/100k_density_0.9_fixed.in``)
- ``<number_of_threads> : Number of threads used to run simulation.

About the detailed implements and result, see **report.pdf**
