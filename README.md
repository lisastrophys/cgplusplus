========================================================================

                         >>> WELCOME TO CG++ <<<

   A Python version of CG+, a modeling program in IDL for dusty circumstellar disks,
 based on the models of Chiang & Goldreich (1997) and Dullemond, Dominik & Natta (2001).

                    Original written by C.P. Dullemond
                          in collaboration with
                         C. Dominik and A. Natta
                                (C) 2002

                       A port to Python by L. Zwicky
                                (C) 2025

       ----->>>> NOTE: Original software is _NOT_ public domain. <<<<-----
        ----->>>>  Any use of this software requires written  <<<<-----
        ----->>>>         permission from the authors         <<<<-----

                        Last CG+ update: 14 Feb 2003
                        Last CG++ update: 19 Jun 2025

    ------------------------------------------------------------------------
    Original CG+ intro:
     This program solves the structure of a protostellar/protoplanetary
     dusty disk surrounding a T Tauri or Herbig Ae/Be star. The disk is
     assumed to be passive and having a flaring geometry. The structure
     of the disk is purely determined by the irradiation of it's surface
     by the central star. The SED of such a disk can then be determined
     and fitted to observations.

     This program comes with a user-interface written by S. Walch and
     C.P. Dullemond (FITCGPLUS). So the easiest way to use this code is
     to start FITCGPLUS in the following way: start IDL, type .r fitcgplus
     and type fitcgplus. Read the header of fitcgplus.pro for more details.

         The equations used in this program are described in:

            Dullemond, Dominik & Natta (2001) ApJ 560, 957

         The model is an improvement of the model proposed by:

            Chiang & Goldreich (1997) ApJ 490, 368

    Intro to CG++:
     A port to Python was inspired by the fact that IDL is a legacy
     paywalled software and CG+ does not run on GDL. The core of
     the program (cgplus.pro) was rewritten with little change (including
     comments) and the results of both versions (IDL and Python) must be
     identical. This script includes a user-interface similar to the
     one provided by fitcgplus.pro but it was written independently.
     It has a few additional features such as enhanced figure control,
     help button and save SED button.
     Currently there is no plan to do anything more with it but
     I am open to suggestions.
------------------------------------------------------------------------

How to use:
Unzip the archive, go into the directory that is created, execute python script.
For the script you need numpy, scipy and matplotlib. It was written with Python 3.8
but other versions may work too.

For fitting your own data: make a 'mystarname.dat' file with the same
format as abaur.dat: frequency in Hz, observed flux in erg/s/cm^2/Hz
and the error bars (down and up) in erg/s/cm^2/Hz. If both error bars
are taken 0.0 then the code connects all the points with a line. Otherwise
they are plotted as points with the error bars given. If they are both
-1.0 they will be data points without an error bar. 

