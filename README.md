# Kinetic Monte Carlo Simulation of Charge Transport

<!-- This project simulates charge transport in disordered organic semiconductors using kinetic Monte Carlo methods. -->

As of now, this repository contains code that models organic molecules using meshes, alongside a blocking algorithm which determine where edges may form between molecules.

## 🐍 Conda Environment Setup

Python dependencies for this project are listed in the `environment.yml` file. To quickly create the necessary environment for this project:

- Within the directory with `environment.yml` file, enter the command `conda env create --file environment.yml`.
- Check whether the new environment is created by the command `conda env list`.
- Activate the environment by `conda activate kmc`.

## ❗ Important Folders & Files

The only folders that are relevant to the final program are:
- `completed` — Holds the main file (`main_pipeline.ipynb`)
- `utils` — All packages for this project

Every other folders are not used in the final program, and can be considered _prototypes_ which catalog this project's development process. 

However, if something should go wrong in running `\completed\main_pipeline.ipynb`, then refer to `\notebooks\08_\` folder. It contains all necessary CSV files (generated from our program) in order for `main_pipeline.ipynb` to run smoothly. A potential source of error might be the _relative PATH_ to CSV files within `main_pipeline.ipynb`.

## 👾 Running the Program
This section details the steps needed to run the program in `\completed\main_pipeline.ipynb`.

For the first run:

1. Execute `Preprocessing` header.
2. Execute `Generate Molecule Meshes` header.
3. Execute `Generate neighbors pairs (Blocking Algo)` header.
4. Execute `Results` header.
5. Execute `Verify blocking molecules` header.

For the second run onwards, most data are already stored as CSV(s) within the same directory:

1. Execute `Preprocessing` header.
2. Execute `Import meshes` header
3. Execute `Import neighbor pairs` header.
4. Execute `Results` header.
5. Execute `Verify blocking molecules` header.
