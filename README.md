# SigmaZ_funcs

This package contains a set of functions that allows you to extract as well asection of transport, T, S and rho along any arbitrary cross-section in GFDL ocean models, and then 
calculate the meridional overturning along that cross section and a SigmaZ diagram of the transports (See Zhang and Thomas, 2021, Nat Comm Earth Env. for a description), as well
as some other high level calculations like the MOC and gyre heat and freshwater transport components along the cross section. 

The SigmaZ_funcs.py script contains more detailed descriptions of its functionality and usability. It requires Raf Dussin's sectionate tool (https://github.com/MDTocean/sectionate), 
John Krasting's momlevel (https://github.com/jkrasting/momlevel), as well as some other packages that can be installed directly with pip or conda.  

An example jupyter notebook, SigmaZ_diag_exmaple.ipynb, gives an example of the usage of the package using various gfdl model output. 

