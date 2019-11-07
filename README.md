# Master-Thesis
In this repository you can find all the code used in the simulations and modelling behind the master thesis "Explaining Grid Cell Activity in One Spatial Dimension by Hippocampal Activity Correlations" (click [here](https://www.researchgate.net/publication/330738714_Master_Thesis_Explaining_Grid_Cell_Activity_in_One_Spatial_Dimension_by_Hippocampal_Activity_Correlations) to access the thesis).

Four different parts have been included:
1. [Plot of the Correlation between Place Cells and comparison with numerical results (simulations).](https://github.com/LeviGuerra/Master-Thesis/blob/master/Code/1-Correlation.py)
2. [Calculation of the Convolutional Kernel.](https://github.com/LeviGuerra/Master-Thesis/blob/master/Code/2-Kernel.py)
3. [Fourier Transform Analysis: Finding the Dominant Fourier Mode.](https://github.com/LeviGuerra/Master-Thesis/blob/master/Code/3-Fourier-Transforms.py)
4. [Visualizing the Pattern Formation.](https://github.com/LeviGuerra/Master-Thesis/blob/master/Code/4-Pattern.py)

## Abstract

Grid cells of the entorhinal cortex of mammals fire when the animal traverses specific locations of the spatial environment. When the animal is able to explore an open field the firing fields of a grid cell form a hexagonal lattice (Hafting et al. 2005). However, in one dimensional environments, grid cells exhibit an irregular pattern with no periodicity and highly variable firing rate between its fields. It has been suggested that such patterns are the result of 1D slices through the hexagonal lattice formed during open field exploration (Yoon et al. 2016, Pröll et al. 2018). 

In the present work, we extend a model that successfully describes the activity of grid cells in the open field to the one dimensional case. This model uses a learning rule for grid cells based on place cell activity using spike-timing dependent plasticity (Monsalve-Mercado et al. 2017). Place cells, present in the hippocampus, have spatial selectivity like grid cells, but fire only at a single location (O'Keefe et al. 1971). 

Different linear track geometries, types of learning windows and place cell parameters have been explored. A regime of parameters has been identified for which the patterns formed have similarities with the experimental findings: varying maximum field firing along the track, together with the loss of symmetry. Nevertheless, the model fails to reproduce other essential features, since the maximum firing rate simply decays along the tracks. 

Anatomically inspired ideas are proposed to extend our model, such as considering nonhomogeneous place cell distributions over linear tracks, together with the possible addition of border cells or inhibition between grid cells to our network.

## About me:

Levi Guerra García (26 y/o). Physicist (M.Sc.) passionate about Machine Learning, Data Analysis and Neuroscience. Further information about myself [here](https://www.linkedin.com/in/leviguerra/).

For any suggestion regarding my master thesis, or for a collaboration proposal, please do not hesistate to send me an email: leviguerra.g@gmail.com

------

*Note: I am working to present the content as clear as possible. However, due to the background needed and the highly specific content, the code might be quite tough to fully understand without being familiar with the thesis. If the abstract is not clear enough, please, check the thesis itself for further details (link above) or contact me; I will be glad to help you!*
