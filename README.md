Unzip the archive, go into the directory that is created, execute python script.
For the script you need numpy, scipy and matplotlib. It was written with Python 3.8
but other versions may work too.

For fitting your own data: make a 'mystarname.dat' file with the same
format as abaur.dat: frequency in Hz, observed flux in erg/s/cm^2/Hz
and the error bars (down and up) in erg/s/cm^2/Hz. If both error bars
are taken 0.0 then the code connects all the points with a line. Otherwise
they are plotted as points with the error bars given. If they are both
-1.0 they will be data points without an error bar. 

