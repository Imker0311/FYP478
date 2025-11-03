# FYP478
Comparison of Autoencoders and UMAP for Fault Detection and Diagnosis

To use code:
(In all cases, the h5 file names should be adjusted as necessary, they are all currently connected, but the file corresponding to the particular dimensionality reduction technique should changed in the Classifiers.py and Classification_Metrics.py files. Also, the file name should be adjusted from in InputVectors.py and CSTR_Simulation.py from Test to Train)
1. Simulate training and testing input vectors with InputVectors.py file. (Files names should be adjusted accordingly)
2. Simulate the CSTR system for both training and testing data. The run and num_steps parameters should be adjusted accordingly.
3. Run the Data_Preprocessing.py file.
4. Run any of the DimRed.py files
5. Run the Classifiers.py file. (Files names should be adjusted accordingly)
6. Run the Classification_Metrics.py file. (Files names should be adjusted accordingly)

The InputVectors_Plot.py and CSTR_Simulation_Plot.py files can be used to visualise input vector and simulation results.
