# **Coded_WFS_SIM**

## **Description**
`OptiVolume` is a Python library for synthesizing differentiable refractive index volumes with arbitrary configurations of simple shapes, such as ellipsoids, cuboids, and planes. Light propagation through 3D structures is simulated using the Beam Propagation Method (BPM). 
---

## **Features**
- **Customizable 3D Structures**: Define refractive index distributions for cubes, spheres, and other geometries.
- **Beam Propagation Method (BPM)**: Accurately model light propagation through inhomogeneous media. (Uses third party package.)
---

## **Installation**

1. Clone the repository using:
```bash
git clone https://github.com/Muhammad-Kazim/OptiVolume.git
```
2. Update the environment.yml conda env location, torch+cu if required.
```bash
cd coded_wfs_sim
conda env create -f environment.yml
```
