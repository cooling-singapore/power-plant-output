# power-plant-output
# genDispatch
Power plant dispatch model, including the energy product-only model of NEMS

## 1 Setup
Copy all the files and directories (as they are) in your desired project location (just make sure you have write access to). genDispatch was built with Python 3.7 and
- NumPy 1.16.5
- SciPy 1.3.1
- Pandas 0.25.1
- Matplotlib 3.1.1
- Pint 0.9 (expicit units)
- FuzzyWuzzy 0.17.0 (string matching)
- xlrd 1.2.0

## 2 Project directory
The project directory is structured as follows:
- top level contains all source codes
- `Inputs and resources` contains the input files and low-level resources, such as:
  - `config.ini` is the configuration file, which defines project-level parameters
  - `Pint energy defn.txt` is a custom Pint 0.9 units definitions file
  - `metadata.ini` contains metadata and instructions to read the inputs data (parsed via DataHandler.py)
  - `genDispatch.log` is the project log
  - `SG power plant database v3.xlsx` is the power plant database file. This, as with many other inputs, can be configured via the metadata file.
- `Results` contains all the dispatch-related results
- `To WRF` contains all the WRF input files and resources
- `Scripts` contains demos and auxiliary scripts (Jupyter notebooks)
