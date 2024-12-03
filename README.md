# Transition Data and Validation Code

This repository stores all the python code used to generate the figures for the manuscript titled "*Fake it till you make it: Using synthetic turbulence to achieve swift converged turbulence statistics in a pressure-driven channel flow* - Patil & García-Sánchez (In Review)"

A preliminary version of the manuscript can be accessed through the pre-print service arXiv at https://doi.org/10.48550/arXiv.2411.11416 

**Please note that this version is a pre-print and not published yet!**


## How to use this repository

Each figure requires a `data` directory to be populated with the requisite data. The original data is stored on `surfDrive` located at https://surfdrive.surf.nl/files/index.php/s/I3OVfon5uUaUXfL

Once you download the data directory, place individual directories into their respective locations corresponding to the figure name. For example: Contents of `data_figure1_and_2/` should be placed within the `figure1_and_2/data/` directory, and so on.

## Plotting

It is important to note that running the python code requires some dependency libraries to be installed via pip. This can be simply done by installing the following libraries.

```
pip install numpy pandas matplotlib cblind scipy
```

All plots are saved as `*.png` format, for high-resolution versions, please consider saving in the `*.eps` format and convert to `*.pdf` using `epstopdf filename.eps`.

--- 


Regards,  
Akshay (https://3d.bk.tudelft.nl/apatil/)  
Clara (https://3d.bk.tudelft.nl/gsclara/)
