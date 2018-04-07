Montager
========

Small program to create a montage from video stacks taken in acq4. 

Usage::

    python montage/montager.py
    
    Brings up a window to select a directory to pull the video images from. This usually will be a "cell" directory.
    
Requirements
------------

pylibrary: https://github.com/pbmanis/pylibrary

matplotlib conda install matplotlib

pyqtgraph: https://github.com/pyqtgraph or conda install pyqtgraph

tifffile: https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html or: pip install tifffile 

imreg (included)

mahotas: conda install -c https://conda.anaconda.org/conda-forge mahotas


Todo
----
Stitch images together using openCV (not a trivial project)

4/6/2018 pbmanis
 