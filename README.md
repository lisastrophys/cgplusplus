# Welcome to CG++

A Python version of CG+, a modeling program in IDL for dusty circumstellar disks, based on the models of Chiang & Goldreich (1997) and Dullemond, Dominik & Natta (2001).

Originally written by C.P. Dullemond in collaboration with C. Dominik and A. Natta (C) 2002

A port to Python by L. Zwicky (C) 2025

## Original CG+ intro:
This program solves the structure of a protostellar/protoplanetary dusty disk surrounding a T Tauri or Herbig Ae/Be star. The disk is assumed to be passive and having a flaring geometry. The structure of the disk is purely determined by the irradiation of it's surface by the central star. The SED of such a disk can then be determined and fitted to observations.
 
This program comes with a user-interface written by S. Walch and C.P. Dullemond (FITCGPLUS). So the easiest way to use this code is to start FITCGPLUS in the following way: start IDL, type .r fitcgplus and type fitcgplus. Read the header of fitcgplus.pro for more details.
 
The equations used in this program are described in: Dullemond, Dominik & Natta (2001) ApJ 560, 957
 
The model is an improvement of the model proposed by: Chiang & Goldreich (1997) ApJ 490, 368

## Intro to CG++:
A port to Python was inspired by the fact that IDL is a legacy paywalled software and CG+ does not run on GDL. The core of the program (cgplus.pro) was rewritten with little change (including comments) and the results of both versions (IDL and Python) must be identical. This script includes a user-interface similar to the one provided by fitcgplus.pro but it was written independently. It has a few additional features such as enhanced figure control, help button and save SED button. Currently there is no plan to do anything more with it but I am open to suggestions.

## How to use:
For fitting your own data: make a 'mystarname.dat' file with the same format as abaur.dat: frequency in Hz, observed flux in erg/s/cm^2/Hz and the error bars (down and up) in erg/s/cm^2/Hz. If both error bars are taken 0.0 then the code connects all the points with a line. Otherwise they are plotted as points with the error bars given. If they are both -1.0 they will be data points without an error bar. 

Opacity files are the same as in the original (source unknown). They can be replaced by your own but they must have the same names and style as original.

## Update 2026.10.27:

Computah, make the application prettier. Now there is a new GUI using Qt instead of Tk and available as executable too. See instructions below.

## Download and run
Ready-made apps for Linux and Windows are attached to each release on the [Releases page](https://github.com/lisastrophys/cgplusplus/releases). They need no Python or other installation.

- **Linux:** download `cgplusplus-linux-x86_64.tar.gz`, unpack it with `tar xzf cgplusplus-linux-x86_64.tar.gz` and start it with `./cgplusplus`. It runs on Ubuntu 22.04, Debian 12, Fedora 36 or newer, and similar systems.
- **Windows:** download `cgplusplus-windows-x86_64.exe` and double-click it. Windows may warn that the app is unrecognised because it is not signed: click *More info*, then *Run anyway*.
- **macOS** (Apple Silicon, macOS 14 or newer): download `cgplusplus-macos-arm64.zip`, double-click it to unpack, and move `CG++.app` to your Applications folder if you like. The app is not signed by Apple, so macOS refuses to open it the first time. Open *System Settings → Privacy & Security*, scroll down and click *Open Anyway*; after that it opens normally. If macOS instead says the app "is damaged", run this once in Terminal and try again: `xattr -dr com.apple.quarantine /Applications/CG++.app` (adjust the path if the app is elsewhere).

The first start takes a few seconds while the app unpacks itself. Model files, observations and saved SEDs open and save in the folder you start the app from.

## Running from source
CG++ needs Python 3.10 or older and NumPy below 2.4, plus SciPy, Matplotlib and a Qt binding:

    pip install "numpy<2" scipy matplotlib PySide6     # on Python 3.8 use PyQt5 instead of PySide6
    python cgplusplus.py

Use `--scale 2` to enlarge the interface on high-resolution screens (also adjustable under Settings).

## License
CG++ is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or (at your option) any later version. See [LICENSE](LICENSE) for the full text. CG++ comes with no warranty.

The packaged apps also contain third-party libraries under their own licenses, among them Qt for Python (PySide6, LGPL v3), NumPy, SciPy and Matplotlib (BSD-style). The source code of each release is available on its release page.
