# CG++: a Python port of CG+, a model for passive flared protoplanetary disks.
# Copyright (C) 2025 L. Zwicky
# Based on CG+ by C.P. Dullemond, C. Dominik and A. Natta (C) 2002.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html
import numpy as np
import os
from dataclasses import dataclass
from typing import List
from scipy.special import erf
import warnings
import matplotlib
import sys
LAUNCH_DIR = os.getcwd()   # where CG++ was started; the file dialogs open here
if getattr(sys, 'frozen', False):
    # Packaged app (PyInstaller): opac/ and the icon live inside the bundle, so read them from there.
    os.chdir(sys._MEIPASS)
    for _stream in ('stdout', 'stderr'):   # a windowed Windows app has no console to print to
        if getattr(sys, _stream) is None:
            setattr(sys, _stream, open(os.devnull, 'w'))
elif not os.path.isdir('opac') and os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opac')):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # started from another folder
# (the Qt and matplotlib-backend imports live in the interface section at the bottom)

'''
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
'''
# Constants
ss = 5.6703e-5
AU = 1.496e13
GG    = 6.672e-8
Msun  = 1.99e33
Lsun  = 3.86e33
mugas = 2.3e0
mp    = 1.6726e-24
kk    = 1.3807e-16
hh = 6.6262e-27
cc = 2.9979e10
yr = 3.1536e7
pc = 3.08572e18

'''
#------------------------------------------------------------------------
#                       FUNCTION: PLANCK
#------------------------------------------------------------------------
'''
def bplanck(nu, T):
    if T==0:
        return np.zeros(len(nu))
    x = hh*nu/kk/T
    plank = np.zeros(len(x))
    mask = (x<100) & (x>0.01)
    plank[mask] = (2*hh*nu[mask]**3/cc**2)/(np.exp(x[mask])-1)
    mask = x<=0.01
    plank[mask] = 2*nu[mask]**2*kk*T/cc**2


    return plank

'''
------------------------------------------------------------------------
          COMPUTE THE dlg(H/R)/dlgR FROM GIVEN STRUCTURE

 This function is important when one wishes to iterate the flaring disk
 structure until energy conservation is guaranteed.

------------------------------------------------------------------------
'''
def find_dlgh(R, H):
    nr = len(R)
    r2 = R[1:]
    r1 = R[:-1]
    hr2 = H[1:]/r2 +1e-5
    hr1 = H[:-1]/r1 +1e-5

    dum = np.zeros(nr)
    dum[1:] = (r1+r2)*(hr2-hr1)/((r2-r1)*(hr2+hr1))
    dum[0] = dum[1]

    return dum

'''
------------------------------------------------------------------------
             COMPUTE THE ERROR IN ENERGY CONSERVATION

 Find the difference between the real flaring index and the computed one,
 and integrate this over d(H/R), since this is the measure of the covering
 fraction, and hence of the luminosity of the disk. The function returns
 the computed disk luminosity divided by the luminosity what it should have
 been. If this number is 1.d0, then there is no error.

------------------------------------------------------------------------
'''
def find_encons_error(R, H, flareindex=None):
    dlgh = find_dlgh(R, H)
    nr = len(R)
    if flareindex is None:
        flareindex = np.zeros(nr) + 2/7

    ### Original version
    # dum1=0
    # dum2=0
    # for ir in range(1,nr-1):
    #     dum1 += (flareindex[ir]/dlgh[ir])*0.5*((H[ir+1]/R[ir+1])-(H[ir-1]/R[ir-1]))
    #     dum2 += 0.5*((H[ir+1]/R[ir+1])-(H[ir-1]/R[ir-1]))

    ### Avoiding cycles version
    dum1 = np.sum((flareindex[1:-1]/dlgh[1:-1])*0.5*((H[2:]/R[2:])-(H[:-2]/R[:-2])))
    dum2 = np.sum(0.5*((H[2:]/R[2:])-(H[:-2]/R[:-2])))

    return dum1/dum2

'''
------------------------------------------------------------------------
         FUNCTION: FIND RADII OF TAU=1 AT ALL FREQUENCIES
------------------------------------------------------------------------
'''
def findtauradii(opac, sigma, r, Td, tauone=1):
    nr = len(r)
    nf = len(opac.freq)
    tau = np.zeros((nr, nf))
    rtau = np.zeros(nf)
    z0 = sigma[:-1]
    z1 = sigma[1:]
    zz = z1-z0
    if (min(zz)*max(zz) < 0):
        raise ValueError("Error in findtauradii: Sigma is not monotonic!")
    if (min(zz)*max(zz) == 0):
        return rtau

    for ir in range(nr):
        kappa = findkappa(opac, Td[ir])#, sca=True)
        tau[ir, :] = sigma[ir]*kappa

    if sigma[nr-1] > sigma[0]:
        rmin=r[0]
        rmax=r[nr-1]
    else:
        rmin = r[nr-1]
        rmax=r[0]

    for inu in range(nf):
        if np.min(tau[:, inu]) > 1:
            rtau[inu] = rmin
        else:
            if np.max(tau[:, inu]) < 1:
                rtau[inu] = rmax
            else:
                rtau[inu] = np.interp(tauone, np.flip(tau[:, inu]), np.flip(r))

    return rtau

'''
------------------------------------------------------------------------
              F_R FUNCTION: HOW MUCH OF STAR IS VISIBLE?
------------------------------------------------------------------------
'''
def visibfrac(Rstar, Hi, ri, h, r):
    if h==0:
        return 0.5
    if r-ri == 0:
        return 1
    eta = ri*(h-Hi)/(r-ri) - Hi
    if eta > Rstar:
        return 1
    if eta < -Rstar:
        return 0

    y = eta/Rstar
    return 1/np.pi * (y*np.sqrt(1-y**2)+np.arcsin(y)) + 0.5


'''
------------------------------------------------------------------------
         FUNCTION: FIND THE DUST KAPPA TABLE AT GIVEN TEMPERATURE

 Sicne some dust species can evaporate, we must find the opacity
 table temperature-dependently. Here's the function for doing so.
------------------------------------------------------------------------
'''
def findkappa(opac, T, abs: bool=True, sca: bool=False):
    kappa = np.zeros(len(opac.freq))
    ndust = len(opac.dnames)
    for i in range(ndust):
        name = opac.dnames[i]
        if T < opac.evaptemp[i]:
            kappa += opac.abun[i]*(abs*opac.abs[name] + sca*opac.sca[name])

    return kappa
'''
------------------------------------------------------------------------
              FUNCTION: FIND THE PLANCK MEAN KAPPA
------------------------------------------------------------------------
'''
def kappaplanck(opac, T):
    nu = opac.freq
    nf = len(nu)
    bnu = bplanck(nu, T)

    kappa = findkappa(opac, T)

    ### Original version of getting planck averaged kappa
    # aa=0
    # bb=0
    # for i in range(1,nf-1):
    #     dnu = 0.5*nu[i] * np.log(nu[i+1]/nu[i-1])
    #     aa += bnu[i]*kappa[i]*dnu
    #     bb += bnu[i]*dnu

    ### Avoiding cycles version
    dnu = 0.5*nu[1:-1] * np.log(nu[2:]/nu[:-2])
    aa = np.sum(bnu[1:-1]*kappa[1:-1]*dnu)
    bb = np.sum(bnu[1:-1]*dnu)

    return aa/bb

'''
------------------------------------------------------------------------
            FUNCTION: FIND THE PSI_S/PSI_I VALUE

 The psi=psi_s/psi_i value is the dimensionless constant in front
 of the Ti determination. It represents the effects of finite optical
 depth for the case of a frequency-dependent opacity.
------------------------------------------------------------------------
'''
def findpsi(sigma, Ti, Ts, opac):
    nu = opac.freq
    nf = len(nu)
    # psis = 0
    # psii = 0
    # btots = 0
    # btoti = 0
    bpls = bplanck(nu, Ts)
    bpli = bplanck(nu, Ti)

    ### A factor sqrt(3) in the opacity because we are doing mu-averaged stuff (fluxes)
    kappai = np.sqrt(3) * findkappa(opac, Ti)
    kappas = np.sqrt(3) * findkappa(opac, Ts)

    ### Original psis and psii
    # for inu in range(1, nf-1):
    #     tau = sigma*kappai[inu]
    #     xpp = 1 - np.exp(-tau)
    #     dnu = 0.5 * nu[inu] * (np.log(nu[inu+1]) - np.log(nu[inu-1]))
    #     psis += bpls[inu]*kappas[inu]*xpp*dnu
    #     psii += bpli[inu]*xpp*dnu
    #     btots += bpls[inu]*kappas[inu]*dnu
    #     btoti += bpli[inu]*dnu

    ### Avoiding cycles
    tau = sigma*kappai[1:-1]
    xpp = 1 - np.exp(-tau)
    dnu = 0.5 * nu[1:-1] * np.log(nu[2:] / nu[:-2])
    psis = np.sum(bpls[1:-1]*kappas[1:-1]*xpp*dnu)
    btots = np.sum(bpls[1:-1]*kappas[1:-1]*dnu)
    psii = np.sum(bpli[1:-1]*xpp*dnu)
    btoti = np.sum(bpli[1:-1]*dnu)

    psis /= btots
    psii /= btoti
    return [psis/psii, psis, psii]

'''
------------------------------------------------------------------------
              FUNCTION: FIND THE SURFACE HEIGHT

 Solve the equation:

                            2 * alpha
   1 - erf(chi/sqrt(2)) = -------------
                          Sigma * kappa

 where Sigma is the total surface density from z=-infty..infty.
 The solver goes down to chi=1.0, below which it will simply
 return 0.d0.
------------------------------------------------------------------------
'''
def findchi(kappa, alpha, sigma):
    rhs = 2 * alpha/(kappa*sigma)
    chi = 4
    iloop = 1
    if 1/rhs < 3:
        return 0
    while True:
        chiold = chi
        dum = -2*np.log(rhs*np.exp(-chi**2/2)/(1-erf(chi*np.sqrt(0.5))))
        chi = np.sqrt(abs(dum))
        iloop+=1
        if iloop > 20:
            print("No convergence in Hs solver")
            break
        if (abs(chi-chiold)/abs(chi+chiold)) < 1e-4:
            break

    return chi

# A function to find all .dat files in the current directory (for plotting observations)
def find_dat():
    filelist = []
    for file in os.listdir():
        if file.endswith(".dat"):
            filelist.append(file)

    return filelist

# A function to find all .model files in the current directory (for using saved models)
def find_model():
    filelist = []
    for file in os.listdir():
        if file.endswith(".model"):
            filelist.append(file)

    return filelist

'''
------------------------------------------------------------------------
               SOLVE RIN INCLUDING THE SELF-IRRADIATION

 This is a way to include the self-irradiation of the inner wall into
 the determination of the radius of the inner wall for a given wall
 temperature (use solvehpinselfirr() instead if you want to give Rin
 instead of Tin). So we give the Hp0 which is calculated without these
 self irradiation corrections, and then make the corrections. This
 involves an iteration, because as Rin moves outwards, the Sigma(Rin)
 changes (simply because Sigma(R)=Sigma0*(R/Rsig)^plsig, where Rsig
 is NOT the Rin, but a fixed radius). NOTE: The argument 'kap'
 contains already the factor of 8 due to the structure of the inner
 rim (see paper).
------------------------------------------------------------------------
'''
def solverinselfirr(rin0, Sigma0, rsig, plsigma, kap, Hp0, fixhs=None, chi=None):
    rin=rin0
    for i in range(10):
        if chi is None:
            # Modify Sigma, according to Sigma(R) powerlaw
            Sig = Sigma0*(rin/rsig)**plsigma
            chi = findchi(kap, 1, Sig)
        if fixhs is None:
            # Since Tin is fixed, we can use the equation of vertical
            # pressure balance to compute Hpin.
            Hpin = Hp0 * (rin/rin0)**1.5
            Hsin = chi * Hpin
        else:
            Hsin = fixhs
            Hpin = Hsin/chi

        # Now modify Rin by the effect of self-irradiation
        rinold = rin
        rin = rin0*np.sqrt(1 + Hsin/rin)
        if abs(rin-rinold)/(rin+rinold) < 1e-3:
            return [rin, Hsin, chi]

    warnings.warn("MAJOR PROBLEM: solve Rin with self-irradiation!")

'''
------------------------------------------------------------------------
               SOLVE TIN INCLUDING THE SELF-IRRADIATION

 This is a way to include the self-irradiation of the inner wall into
 the determination of the temperature of the inner wall, if the Rin
 is given (use solverinselfirr() instead if you want to give Tin
 instead of Rin).
------------------------------------------------------------------------
'''
def solvehpinselfirr(rin,chi,Hpin0):
    Hpin = Hpin0
    for i in range(10):
        ohp = Hpin
        Hpin = Hpin0*(1+chi*Hpin/rin)**0.125
        if abs(Hpin-ohp)/(Hpin+ohp) < 1e-3:
            return Hpin

    warnings.warn("MAJOR PROBLEM: solve Tin with self-irradiation!")

'''
------------------------------------------------------------------------
               SOLVE THE TI USING THE PSI FUNCTION

 This is a way to include the effects of psi into the deterination of
 the interior temperature Ti.
------------------------------------------------------------------------
'''
def solvetiwithpsi(Tiorig, Ts, sigma, opac):
    psi = 1
    for i in range(30):
        oldpsi = psi
        Ti = Tiorig * psi**0.25
        psi, psis, psii = findpsi(sigma, Ti, Ts, opac)
        if abs(psi-oldpsi)/(psi+oldpsi) < 1e-3:
            return [Ti, psi, psis, psii]

    raise ValueError("MAJOR PROBLEM: solve Ti with Psi did not converge!")

'''
------------------------------------------------------------------------
           FUNCTION: SOLVE OPTICALLY THIN DUST TEMPERATURE
    
     Solve the equation:
    
       1   /                       /
      ---- | F_nu kappa_nu dnu  =  | B_nu(Ts) kappa_nu dnu
      4 pi /                       /
    
     where F_nu is the sum of the stellar flux and the flux from
     the rim.
    
     NOTE: BUGFIX (12.02.02). The real occultation of the central
           star is more complicated than the simple thing used before.
           So now you must specify the starvis fraction yourself.

------------------------------------------------------------------------
'''
def solvetthin(opac, Tstar, Rstar, Trim, Rrim, Hrim, r, z, starvis: float = None, fredux: float = 1.0):
    if starvis == None:
        raise ValueError("Please specify star visibility fraction")
    nu = opac.freq
    nf = len(nu)

    # Compute the delta and the projected surface of the inner rim
    if ((Trim!=0) & (Rrim!=0) & (Hrim!=0) & (z!=0)):
        delta = (r/Rrim - 1)/(z/Hrim - 1)
        if delta > 0:
            cosi = np.cos(np.arctan((r-Rrim)/(z-Hrim)))
            if delta < 1:
                S = 2 * Rrim**2 * cosi * (delta * np.sqrt(1-delta**2) + np.arcsin(delta))
            else:
                S = np.pi * Rrim**2 * cosi
        else:
            S = 0
    else:
        S = 0
    # Check
    if S==0 and starvis==0:
        return 0

    flux = starvis * np.pi * bplanck(nu, Tstar) * Rstar**2 / r**2 + S * bplanck(nu, Trim) / r**2
    # Extinct by exp(-1), or not, dependent on the value of fredux, currently fredux=1 by default
    flux *= fredux

    if max(flux) == 0:
        raise ValueError('PROBLEM in solvetthin: zero flux')

    # Now the total flux, and the initial guess for the temperature
    kappa = findkappa(opac,0)

    # ftot = 0
    # for inu in range(1,nf-1):
    #     ftot += 0.5*flux[inu]*nu[inu]*(np.log(nu[inu+1]) - np.log(nu[inu-1]))

    ftot = np.sum(0.5*flux[1:-1]*nu[1:-1]*np.log(nu[2:]/nu[:-2]))

    if ftot == 0:
        return 0

    Tdold = (ftot/(4*ss))**0.25

    # fkap = 0
    # for inu in range(1,nf-1):
    #     fkap += 0.5*flux[inu]*nu[inu]*kappa[inu]*(np.log(nu[inu+1]) - np.log(nu[inu-1]))

    # Now compute the flux * kappa integral
    fkap = np.sum(0.5*flux[1:-1]*nu[1:-1]*kappa[1:-1]*np.log(nu[2:]/nu[:-2]))

    Td = (fkap/(4*ss*kappaplanck(opac,Tdold)))**0.25

    # Now do the iteration to get the real temperature
    iloop = 0
    while (np.abs(Tdold-Td)/np.abs(Tdold+Td) > 1e-4):
        Tdold = Td
        Td = (fkap / (4 * ss * kappaplanck(opac, Tdold))) ** 0.25
        iloop += 1
        if iloop > 100:
            print("No convergence in solvetthin")
            break

    return Td

@dataclass
class OpacityTable:
    '''
    This class is a replacement of a readgrainopac function
    It reads the "dustopac_name.inp" and "frequency.inp" files in opac folder
    '''
    dnames: List[str] = None
    abun: List[float] = None
    evaptemp: List[float] = None

    def __post_init__(self):
        if self.dnames is None:
            raise TypeError('No names for dust types were given')
        if len(self.dnames) == 1:
            self.abun = [1]
        if len(self.dnames) > 1:
            norm = sum(self.abun)
            self.abun = [x/norm for x in self.abun]
        if self.abun is None:
            raise TypeError('No abundances for dust types were given')
        if self.evaptemp is None:
            # For some reason fitcgplus never gave an option to set evaporation temperatures and maybe for the best...
            # (If it is set to something realistic around 1500 K everything crashes with current implementation of findkappa)
            # print('No evaporation temperatures for dust types were given, assuming your dust doesnt evaporate')
            self.evaptemp = np.zeros(len(self.dnames)) + 100000
        if len(self.dnames) != len(self.abun):
            raise ValueError('Number of dust species and their abundance must be the same')
        self.freq = np.loadtxt('opac/frequency.inp', skiprows=2)
        self.wav = 2.9979e14 / self.freq
        nf = len(self.freq)
        self.abs = {name: np.zeros(nf) for name in self.dnames}
        self.sca = {name: np.zeros(nf) for name in self.dnames}
        for name in self.dnames:
            self.abs[name] = np.loadtxt(f'opac/dustopac_{name}.inp', max_rows=nf)
            self.sca[name] = np.loadtxt(f'opac/dustopac_{name}.inp', skiprows=nf+1)

@dataclass
class Disk:
    '''
    This is a Disk class that hosts functions that calculate CGdisk, diffedge and accretion disk
    (accretion disk is currently not an option in the application while CGdisk+diffedge (combi) are the default)

    Meaning of the parameters:
    Mdisk --- *gas* disk mass in Msun
    Mdot --- accretion rate in Msun/yr, used only by accretion disk
    Sigma0 --- Sigma0 = Sigma(Rc)
    Rin --- inner radius of the disk in AU, start of the radial grid
    Rout --- outer radius of the disk in AU, end of the radial grid
    Rc --- characteristic radius of the disk in AU
    Lstar --- luminosity of the star in Lsun
    Mstar --- stellar mass in Msun
    Tstar --- effective temperature of the star in K
    nr --- number of radial cells
    plsigma --- sigma power law exponent, Sigma(R) = Sigma0 * (R/Rc)**plsigma
    opac --- opacity table with kappa and frequency
    Tin --- temperature at the inner radius in K, can be used to set the Rin alternatively
    grazang --- grazing angle or a way to set constant alpha that is used for temperature calculation
    starvis --- a way to set a constant star visibility fraction
    epsfix --- a constant to alternatively calculate surface temperature Ts = sqrt(0.5*Rstar/R) * Tstar / epsfix**0.25
    noselfirr --- parameter to turn off self-irradiation
    diffedge --- dictionary with parameters of diffuse inner edge, calculated in diffedgefunc or can be set manualy (not a part of app)
    fredux --- flux reduction constant
    usepsi --- a parameter whether to calculate psi for midplane temperature
    chifix --- fixed chi=Hs/Hp for the whole disk
    chiwall --- fixed chi for the inner rim
    fixhs --- fixed Hs for the inner rim
    nonear --- if set, skip the 0.4*Rstar/R term in alpha
    nopuff --- turn off puffing in diffedge
    initflare --- starting value of flaring angle
    flidx --- constant flaring angle
    disk_type --- parameter to know what type of disk we have
    '''
    Mdisk: float = 0.01
    Mdot: float = None
    Sigma0: float = None
    Rin: float = None
    Rout: float = 100
    Rc: float = 20
    Lstar: float = 1
    Mstar: float = 1
    Tstar: float = 4000
    nr: int = 100
    plsigma: float = -1
    opac: OpacityTable = OpacityTable(dnames=['silicate'])
    Tin: float = None
    grazang: List[float] = None
    starvis: float = None
    alpha: float = None
    epsfix: float = None
    noselfirr: bool = False
    diffedge: dict = None
    fredux: float = 1
    usepsi: bool = False
    chifix: float = None
    chiwall: float = None
    fixhs: float = None
    nonear: bool = False
    nopuff: bool = False
    initflare: float = 1/7
    flidx: List[float] = None
    disk_type: str = 'cgdisk'

    def ini_Sigma(self):
        r = self.r
        sigma_unnorm = (r/self.Rc/AU)**(self.plsigma)
        if self.Sigma0 is None:
            if self.plsigma != -2:
                self.Sigma0 = self.Mdisk / 100 * Msun * (self.plsigma + 2) * (self.Rc * AU) ** self.plsigma / (
                            2 * np.pi * (self.Rout * AU) ** (self.plsigma + 2) - (self.Rin * AU) ** (self.plsigma + 2))
            else:
                self.Sigma0 = self.Mdisk / 100 * Msun * (self.Rc * AU) ** self.plsigma / (
                            2 * np.pi * np.log(self.Rout / self.Rin))
        self.Sigma = self.Sigma0*sigma_unnorm


    def sigma0_from_mass(self):
        # This routine calculates the sigma0 from the given mass. This routine
        # takes into account the dependency of the position of the inner rim
        # on the Sigma0 itself.
        if (self.Rin is None) and (self.Tin is None):
            raise ValueError("Must specify either Tin or Rin")
        if (self.Rin is not None) and (self.Tin is not None):
            raise ValueError("Must specify either Tin or Rin but not both!!")


        if self.Tin is not None:
            rin = np.sqrt(self.Lstar*Lsun/(4*np.pi*ss*self.Tin**4))

            rinold = rin
            conv = 0
            for ii in range(10):
                if conv == 0:
                    if self.plsigma != -2:
                        sigma0 = self.Mdisk/100*Msun*(self.plsigma+2)*(self.Rc*AU)**self.plsigma/(2*np.pi*(self.Rout*AU)**(self.plsigma+2) - rin**(self.plsigma+2))
                    else:
                        sigma0 = self.Mdisk / 100 * Msun * (self.Rc*AU)**self.plsigma/(2*np.pi*np.log(self.Rout*AU/rin))

                    self.Sigma0 = sigma0
                    self.diffedgefunc(nodiff=True)
                    rin = self.r[0]

                    if abs(rin/rinold - 1) < 1e-2:
                        conv = 1
                    rinold = rin

            if conv==0:
                raise ValueError("Finding Sigma0 from Mass: no convergence")
        elif self.Rin is not None:
            rin = self.Rin * AU
            if self.plsigma != -2:
                sigma0 = self.Mdisk / 100 * Msun * (self.plsigma + 2) * (self.Rc * AU) ** self.plsigma / (
                            2 * np.pi * (self.Rout * AU) ** (self.plsigma + 2) - rin ** (self.plsigma + 2))
            else:
                sigma0 = self.Mdisk / 100 * Msun * (self.Rc * AU) ** self.plsigma / (
                            2 * np.pi * np.log(self.Rout * AU / rin))

            self.Sigma0 = sigma0


    def rstar(self):
        return np.sqrt((self.Lstar*Lsun)/(4*np.pi*ss*self.Tstar**4))


    '''
    ------------------------------------------------------------------------
               FUNCTION: MAKE A CHIANG & GOLDREICH DISK
    This function makes a flared disk according to the formulae of CG97,
    with some small modifications. In particular, we self-consistently
    compute certain dimensionless tuning parameters such as the value of
    chi=Hs/Hp and the psi-values that modify the equation for Ti in case
    of non-thick vertical optical depths. All these self-consistent
    determinations are done with the full opacity table.
    If diffedge is given, this function is automatically going to add
    the emission from the inner rim to the irradiative flux.
    
    NOTE:
    To get the original CG97 model from this function, call it as:
        cgdiskfunc(ms,ts,ls,rsig,sig0,pls,rin,rout,nr, flareindex=(2.d0/7.d0),chifix=4.d0) 
    but note that the original CG97 model is in fact wrong in the sense 
    that it does not conserve energy. Chiang et al 2001 fixed this (we use
    their method to fix it here too), but they didn't mention the seriousness 
    of the problem of their original model. 
    ------------------------------------------------------------------------
    '''
    def CGDisk(self):
        nr = self.nr

        if self.flidx is None:
            self.flareindex = np.zeros(nr) + self.initflare
            self.globflare = False
        else:
            if len(self.flidx) == 1:
                self.flareindex = np.zeros(nr) + self.flidx[0]
            else:
                self.flareindex = self.flidx
            self.globflare = True


        self.irfullshadow = -1
        self.irhalfshadow = -1

        alpha = np.zeros(nr)
        Hs = np.zeros(nr)
        fr = np.zeros(nr)
        Ts = np.zeros(nr)
        Ti = np.zeros(nr)
        Hp = np.zeros(nr)
        chi = np.zeros(nr)
        psitot = np.zeros(nr)
        psis = np.zeros(nr)
        psii = np.zeros(nr)
        tausurf = np.zeros(nr)
        sigsurf = np.zeros(nr)
        alb = np.zeros(nr)
        fscat = np.zeros(nr)
        flareindex = self.flareindex
        rstar = self.rstar()
        for ir in range(nr):
            '''
            ##
            ## Find the flareindex, but update only every two steps (see appendix
            ## of paper by Chiang et al. ApJ 547:1077-1089 (2001).
            ##
            '''
            if self.globflare==False:
                if (ir>0) & (ir % 2 == 0):
                    hr2 = Hs[ir-2]/self.r[ir-2]
                    hr1 = Hs[ir-1]/self.r[ir-1]
                    if (hr1 > 0) & (hr2 > 0) & (self.irhalfshadow < ir-2):
                        flareindex[ir] = (self.r[ir-2]+self.r[ir-1])*(hr2-hr1)/((self.r[ir-2]-self.r[ir-1])*(hr2+hr1))
                    else:
                        flareindex[ir]=flareindex[ir-1]
                    if flareindex[ir] < 0.05:
                        print(f"WARNING: Flaring index < 0.05 at ir={ir}")
                        flareindex[ir]=0.05
                elif (ir>0):
                    flareindex[ir]=flareindex[ir-1]
            # initial guess
            if self.grazang is not None:
                alpha[ir] = self.grazang[ir]
            else:
                alpha[ir] = 0.2
            if ir>0:
                Hs[ir] = Hs[ir-1] * self.r[ir]/self.r[ir-1]
            '''
            ##
            ## Now work on the CG97 disk strucuture, where we first do things
            ## without considering the shadowing effect by the inner rim, but
            ## we do add the emission from the inner rim. 
            ##
            ## So iterate on the CG97 equations (incl inner rim emission) 
            ## until convergence.
            ##
            '''
            iloop=1
            while True:
                '''
                
                 The fraction of the starlight visible (if R_in=R_*, this is 0.5
                 if R_in>>R_* this is 1.0).
                
                '''
                if self.starvis is not None:
                    if self.starvis == 1:
                        fr[ir] = 0.5
                    else:
                        fr[ir] = 1
                else:
                    fr[ir] = visibfrac(rstar, 0, self.r[0], Hs[ir], self.r[ir])
                '''
                
                 Then determine the temperature of the surface layer, which is not
                 dependent on the disk structure. Compute the dust temperature
                 either according to a given epsilon efficiency factor (kappa_vis
                 / kappa_ir) or self-consistently.
                
                '''
                if self.epsfix is not None:
                    Ts[ir] = np.sqrt(0.5*rstar/self.r[ir]) * self.Tstar / self.epsfix**0.25
                else:
                    if (self.noselfirr is False) & (self.diffedge is not None):
                        Ts[ir] = solvetthin(self.opac, self.Tstar, rstar, self.diffedge['tin'], self.r[0],
                                            self.diffedge['Hsin'], self.r[ir], Hs[ir], starvis=fr[ir], fredux=self.fredux)
                        if (Ts[ir] == 0):
                            break
                    else:
                        if fr[ir] > 0:
                            Ts[ir] = solvetthin(self.opac, self.Tstar, rstar, 0, 0,
                                                0, self.r[ir], Hs[ir], starvis=fr[ir],
                                                fredux=self.fredux)
                            if (Ts[ir] == 0):
                                break
                        else:
                            Ts[ir] = 1
                '''
                
                 The disk interior temperature according to Eq.(12a) of CG97. Note
                 that their factor of 1/4 comes from two factors of 1/2. One comes
                 from the fact that they assume the disk to go all the way to the 
                 star (i.e. fr=0.5). We use the fr (above) for that. The other 
                 comes from the fact that half of the emission of the surface layer
                 is radiated away and only half reaches the interior. Tistar is,
                 by definition, the Ti due to only stellar irradiation flux. First
                 we compute Tistar0, which is the value for zero albedo.
                
                '''
                Tistar0 = (0.5 * alpha[ir] * fr[ir])**0.25 * np.sqrt(rstar/self.r[ir]) * self.Tstar
                '''
                
                 We postpone the scattering effects on Tistar to a later point.
                 
                 Now, if there is an inner rim of the disk, this may add some more
                 flux to the irradiation (we included this effect in Ts already but
                 not yet in Ti, so we do this here). We assume here that at 2--3 mum
                 the albedo is zero, and we only include the thermal emission from 
                 the inner rim. Later we can include also the scattered light from
                 the inner rim.##############
                
                '''
                if (self.noselfirr is False) & (self.diffedge is not None):
                    if (self.r[ir] != self.r[0]) and (Hs[ir] != self.diffedge['Hsin']):
                        delta = (self.r[ir]/self.r[0] - 1)/(Hs[ir]/self.diffedge['Hsin'] - 1)
                    else:
                        delta=0
                    if delta > 0:
                        costheta = np.cos(np.arctan((self.r[ir] - self.r[0])/(Hs[ir] - self.diffedge['Hsin'])))
                        if delta < 1:
                            S = 2 * self.r[0]**2 * costheta * (delta*np.sqrt(1-delta**2) + np.arcsin(delta))
                        else:
                            S = np.pi * self.r[0]**2 * costheta
                    else:
                        S=0
                    Tiself = (0.5*alpha[ir]*S/(np.pi*self.r[ir]**2))**0.25 * self.diffedge['tin']
                else:
                    Tiself = 0
                '''
                ## 
      ##  <<added for scattering>>
      ##
      ## Now include all the effects of scattering into the computation
      ##
      ##  !!!! SCATTERING SWITCHED OFF !!!!
      ##  No good recipe for scattering is found, so for the moment 
      ##  switch off scattering. See Dullemond & Natta for a discussion 
      ##  of the effects of scattering on the SED of a disk.
      ##
#      if n_elements(noscat) eq 0 then begin
#         ##
#         ## Modify the Tistar due to scattering away of part of the secondary
#         ## radiation from the surface layer. This radiation consists of 
#         ## (1-albedo) thermal and (albedo) scattered radiation. Of the 
#         ## scattered radiation only 1-reflect(albedo) will eventually
#         ## thermalize. So of the total flux downwards, only a fraction 
#         ## 1-albedo*reflect(albedo) can be used to thermalize the interior.
#         ## So modify the Tistar by this factor to the power 0.25. 
#         ##
#         ## NOTE: in using the function "reflect(albedo)", we assume that the
#         ## disk is vertically optically thick to scattering and also
#         ## effectively optically thick. This may be a wrong assumption, and
#         ## may need to be fixed in the future.
#         ##
#         if n_elements(albedo) ne 0 then alb[ir]=albedo else begin
#            alb[ir] = albedoplanck(opac,tstar,Ts[ir])
#         endelse
#         refl   = reflect(alb[ir])
#         Tistar = Tistar0 * (1.d0-alb[ir]*refl)^0.25
#         ##
#         ## The Tistar0 can be used to remind what the actual irradiation
#         ## flux from the star was: Firrstar=2*ss*Tistar0^4. This holds even
#         ## when scattering is included. Note that one must use Tistar0, and
#         ## not Tistar, because the latter (see below) is corrected for the
#         ## fact that with scattering part of the flux from the disk
#         ## interior is in fact the 'albedo*reflect(albedo)' fraction of the
#         ## input flux that is not thermalized but instead reflected.
#         ##
#         ## Using this Firrstar, and knowing the albedo, we know the total
#         ## flux of the scattered radiation.
#         ##
#         Firrstar  = 1.13406d-4 * Tistar0^4
#         Fscat[ir] = 0.5 * alb[ir] * (1.d0+refl) * Firrstar
#         ##
#      endif else begin
                '''
                Tistar = Tistar0
                alb[ir] = 0
                fscat[ir] = 0

                '''
                
                 Then add the contributions of the stellar irradiation and the
                 self irradiation to obtain the actual internal temperature.
                
                '''
                Ti[ir] = (Tistar**4 + Tiself**4)**0.25
                '''
                
                 Now, if the psi-factor should be consistently included, then we 
                 call a separate subroutine to modify the Ti according to the value
                 of psi. It then becomes an implicit equation for Ti. The psi factor
                 is the factor taking into account the fact that the disk interio may
                 not be optically thick to its own radiation and/or the radiation 
                 the surface layer. The psi factor depends on the Ti and Ts, and
                 on the surface density of the interior and of course on the opacity.
                 
                 Note on the inclusion of scattering: we assume here that the disk
                 is optically thick to scattering!
                
                '''
                if self.usepsi:
                    Ti[ir], psitot[ir], psis[ir], psii[ir] = solvetiwithpsi(Ti[ir], Ts[ir], self.Sigma[ir], self.opac)
                '''
                
                 The pressure scale height according to Eq.(7) of CG97, see also
                 for rewritten version Eq.(4) Dullemond 2000 A&A 361, L17-20
                
                '''
                Hp[ir] = self.r[ir] * np.sqrt((Ti[ir]*self.r[ir])/(self.Tc * rstar))
                '''
                
                 Find the chi==Hs/Hp self-consistently by using the grazing angle
                 alpha and the Planck mean opacity at stellar temperatures. If 
                 self irradiation by the rim is also present, another chi is
                 computed at the temperature of the rim.
                
                 [bugfix 12-12-00: Ti[ir]-->Tstar in call to kappaplanck]
                '''
                if self.chifix is None:
                    chi[ir] = findchi(kappaplanck(self.opac, self.Tstar), alpha[ir], self.Sigma[ir])
                    if chi[ir]<1:
                        print(f"Optically thin disk at {ir}")
                        chi[ir] = 1
                else:
                    chi[ir] = self.chifix

                Hs[ir] = Hp[ir] * chi[ir]
                '''
                
                 Find the grazing angle alpha according to Eq.(5) of CG97
                
                '''
                alphaold = alpha[ir]
                if self.grazang is None:
                    alpha[ir] = 0.4 * (1-int(self.nonear)) * rstar / self.r[ir] + flareindex[ir]*Hs[ir]/self.r[ir]
                else:
                    alpha[ir] = self.grazang[ir]
                iloop += 1
                if iloop>100:
                    print(f"No convergence at ir={ir}")
                    break
                if abs(alphaold-alpha[ir])/abs(alpha[ir]+alphaold) < 1e-4:
                    break
            '''
            
             Now that we found the CG97 value of the Hs (incl emission from the
             inner rim, but still without the self-shadowing), we can check 
             whether it lies in the shadow.
            
            '''
            if self.diffedge is not None:
                if (Hs[ir]/self.r[ir] < self.diffedge['Hsin']/self.r[0]):
                    '''
                    
                     CG97 lies in shadow. So CG cannot exist here. We either simply
                     put the temperature, and scale height to zero (if noselfirr is
                     explicitly set to prevent this), or we do the self-irradiated
                     disk: irradiation by the disk's inner rim
                    
                    '''
                    if self.noselfirr is False:
                        '''
                         
                         Switch to the half-shadowed disk: the disk with irradiation
                         by the inner rim. The procedure to find the disk in this
                         region is roughly the same as
                        
                        '''
                        iloop = 1
                        while True:
                            '''
                            
                             Determine the temperature of the surface layer. 
                             We don't test for epsfix here.
                             
                             BUGFIX 12-02-02
                            '''
                            Ts[ir] = solvetthin(self.opac, 0, 0, self.diffedge['tin'], self.r[0], self.diffedge['Hsin'],
                                                self.r[ir], Hs[ir], starvis=0)
                            '''
                            
                             Now compute the self-irradiation flux. Assume that
                             the albedo at 2--3 mum is zero.
                            
                            '''
                            delta = (self.r[ir]/self.r[0] - 1)/(Hs[ir]/self.diffedge['Hsin'] - 1)
                            if delta > 0:
                                costheta = np.cos(np.arctan((self.r[ir]-self.r[0])/(Hs[ir]-self.diffedge['Hsin'])))
                                if delta < 1:
                                    S = 2 * self.r[0]**2 * costheta * (delta*np.sqrt(1-delta**2) + np.arcsin(delta))
                                else:
                                    S = np.pi * self.r[0]**2 * costheta
                            else:
                                S = 0
                            Ti[ir] = (0.5*alpha[ir]*S/(np.pi*self.r[ir]**2))**0.25 * self.diffedge['tin']

                            if (Ti[ir] > 0) and (Ts[ir] > 0):
                                if self.usepsi:
                                    Ti[ir], psitot[ir], psis[ir], psii[ir] = solvetiwithpsi(Ti[ir], Ts[ir], self.Sigma[ir], self.opac)
                            else:
                                '''
                                
                                 Apparently the disk cannot be sustained
                                 777 is a magic cookie which will be recognized
                                 below
                                
                                '''
                                iloop = 777

                            Hp[ir] = self.r[ir] * np.sqrt((Ti[ir] * self.r[ir]) / (self.Tc * rstar))
                            if self.chifix is None:
                                chi[ir] = findchi(kappaplanck(self.opac, self.diffedge['tin']), alpha[ir], self.Sigma[ir])
                                if chi[ir] < 1:
                                    print(f"Optically thin disk at {ir}")
                                    chi[ir] = 1
                            else:
                                chi[ir] = self.chifix

                            Hs[ir] = Hp[ir] * chi[ir]

                            alphaold = alpha[ir]
                            if self.grazang is None:
                                alpha[ir] = flareindex[ir] * Hs[ir] / self.r[ir]
                            else:
                                alpha[ir] = self.grazang[ir]
                            iloop += 1
                            if (abs(alphaold - alpha[ir]) / abs(alpha[ir] + alphaold) < 1e-4) or (iloop == 777) or (Hs[ir] == 0):
                                break

                        if (iloop==777) or (Hs[ir]==0):
                            '''
                            
                             Apparently no solution was found here, so the disk
                             must be in the full shadow here.
                            
                            '''
                            Hs[ir]=0
                            Hp[ir]=0
                            Ti[ir]=0
                            Ts[ir]=0
                            self.irfullshadow = ir
                            self.irhalfshadow = ir
                        else:
                            '''
                            
                             Apparently the half-shadowed disk can already survive
                             here, so we update the halfshadow radius, but leave
                             full shadow un-updated.
                            
                            '''
                            self.irhalfshadow = ir
                    else:
                        '''
                         
                         No half-shadowed disk allowed: just switch off CG97
                        
                        '''
                        Hs[ir] = 0
                        Hp[ir] = 0
                        Ti[ir] = 0
                        Ts[ir] = 0
                        self.irfullshadow = ir
                        self.irhalfshadow = ir
                    '''
                    
                     Some of the previously used internal variables for the non
                     self-shadowed part of the disk will be used below. So we 
                     set them to the appropriate values now that this
                     non-self-shadowed part does not exist at this point.
                     
                    '''
                    Tistar0=0
                    Tiself=Ti[ir]

            '''
            
             The vertical optical depth of the surface layer is fixed according
             to energy conservation: half of the radiation is emitted by the
             surface layer, and half by the interior. With scattering this is
             not that easy anymore, but after a modification we can still use
             the simple equation of the temperature ratios as used in DDN2001.
            
              <<modified for scattering>>
            
            '''
            if Ts[ir] > 0:
                tausurf[ir] = (1 - alb[ir])*(Tistar0**4 + Tiself**4) / Ts[ir]**4 / 2.
            else:
                tausurf[ir] = 0
            if (Ts[ir] > 0) and (Ts[ir] < max(self.opac.evaptemp)):
                sigsurf[ir] = tausurf[ir] / kappaplanck(self.opac, Ts[ir])
            else:
                sigsurf[ir] = 0

            if (Hs[ir]/self.r[ir] > 1):
                warnings.warn(f'Problem at ir={ir}, R={self.r[ir]}: Hs/R > 1')

        '''
        
         Check whether at some radius the Hs/r goes down, in order to be
         able to check energy conservation: which fraction of 4pi is covered
         by the CG disk? NOTE: this is really a hack....
        
        '''
        irmaxcg=0
        for ir in range(1, nr):
            if Hs[ir]/self.r[ir] > Hs[ir-1]/self.r[ir-1]:
                irmaxcg=ir

        self.alpha = alpha
        self.flareindex = flareindex
        self.Hs = Hs
        self.Hp = Hp
        self.Ti = Ti
        self.Ts = Ts
        self.Fscat = fscat
        self.alb = alb
        self.psitot = psitot
        self.psis = psis
        self.psii = psii
        self.sigsurf = sigsurf
        self.tausurf = tausurf
        self.irmaxcg = irmaxcg

    '''
    ------------------------------------------------------------------------
           FUNCTION: COMPUTE DIFFUSIVE INNER EDGE STRUCTURE
    ------------------------------------------------------------------------
    '''
    def diffedgefunc(self, noselfirr: bool = False, nodiff: bool = False):
        # Tinsf = 0
        # tauinsf=0
        if (self.Tin is None) and (self.Rin is None):
            warnings.warn("Setting T_in=1500K by default!!")
            self.Tin = 1500

        '''
        
         Find the effective kappa for radial outward rays. The factor of 8
         comes in because of the expected structure of the diffusive disk
         behind the inner rim. See paper.
        
        '''
        kap8 = 8 * kappaplanck(self.opac, self.Tstar)
        '''
        
         From here there are two possible ways to determine the inner edge
        
        '''
        if self.Tin is not None:
            '''
            
             Determine the inner radius from the temperature tin. Include the
             self-irradiation in this computation, unless /noselfirr. 
             First guess of Rin, based on non-self-irradiating inner rim
            
            '''
            self.Rin = np.sqrt(self.Lstar*Lsun/(4*np.pi*ss*self.Tin**4))/AU
            rin = self.Rin * AU
            Hpin = np.sqrt(kk*self.Tin*rin**3/(mugas*mp*GG*self.Mstar*Msun))
            if noselfirr is False:
                rin0 = rin
                Hp0 = Hpin
                rin, Hsin, chi = solverinselfirr(rin0, self.Sigma0, self.Rc*AU, self.plsigma, kap8, Hp0, fixhs=self.fixhs, chi=self.chiwall)
                self.Rin = rin/AU
                Hpin = np.sqrt(kk*self.Tin*rin**3/(mugas*mp*GG*self.Mstar*Msun))
            else:
                Sig = self.Sigma0*(rin/self.Rc/AU)**self.plsigma
                chi = findchi(kap8, 1, Sig)
                if chi < 1:
                    raise ValueError("Optically thin disk at inner radius")
                if self.fixhs is None:
                    Hsin = Hpin * chi
                else:
                    Hsin = self.fixhs
                warnings.warn('Self-irradiation of inner wall not included!')
        elif self.Rin is not None:
            rin = self.Rc * AU
            Sig = self.Sigma0*(rin/self.Rc/AU)**self.plsigma

            if self.chiwall is not None:
                chi = self.chiwall
            else:
                chi = findchi(kap8, 1, Sig)
            '''
            
             Determine temperature of the inner rim and the vertical height 
             together. If the wall-self-irradiation iss included, then do this
             by an iterative procedure.
            
            '''
            Tin = (self.Lstar*Lsun/(4*np.pi*rin**2*ss))**0.25
            if self.fixhs is None:
                Hpin = np.sqrt(kk * Tin * rin ** 3 / (mugas * mp * GG * self.Mstar * Msun))
                Hsin = chi * Hpin
            else:
                Hsin = self.fixhs
                Hpin = Hsin / chi
            if noselfirr is False:
                if self.fixhs is None:
                    Hpin0 = Hpin
                    Hpin = solvehpinselfirr(rin, chi, Hpin0)
                    Hsin = chi * Hpin

                Tin = Tin*(1+Hsin/rin)**0.25
            else:
                warnings.warn('Self-irradiation of inner wall not included!')

            self.Tin = Tin

        '''
        #
        # The scattering stuff
        #
        #  !!!! SCATTERING SWITCHED OFF !!!!
        #  No good recipe for scattering is found, so for the moment 
        #  switch off scattering. See Dullemond & Natta for a discussion 
        #  of the effects of scattering on the SED of a disk.
        #
        # if n_elements(noscat) ne 0 then albedowall=0.d0 else begin
        #    if n_elements(albedo) ne 0 then albedowall=albedo else begin
        #       albedowall = albedoplanck(opac,tstar,tin)
        #    endelse
        # endelse
        albedowall = 0.d0
        #
        #  !!!! SCATTERING SWITCHED OFF !!!!
        #  No good recipe for scattering is found, so for the moment 
        #  switch off scattering. See Dullemond & Natta for a discussion 
        #  of the effects of scattering on the SED of a disk.
        #
        #reflwall   = reflect(albedowall)
        reflwall=0.d0
        #
        # Stuff...
        #
        '''

        if self.Rin < self.rstar()/AU:
            raise ValueError('Inner radius of disk within stellar radius... Aborting...')

        self.r = self.Rin * (self.Rout / self.Rin) ** (np.array(range(self.nr)) / (self.nr - 1)) * AU
        self.ini_Sigma()
        Hsdiff = Hsin-(self.r-rin)/8
        Hsdiff = 0.5*(abs(Hsdiff)+Hsdiff)
        Hpdiff = Hsdiff/chi
        Tidiff = self.Tin * (Hsdiff/Hsin)**2
        Hshadow = Hsin * (self.r/self.r[0])

        if not nodiff:
            self.diffedge = {
                'Hsin': Hsin, 'Hpin': Hpin, 'tin': self.Tin,
                'Hsdiff': Hsdiff, 'Hpdiff': Hpdiff, 'Tidiff': Tidiff,
                'Hshadow': Hshadow, 'chirim': chi
                             }

    '''
    ------------------------------------------------------------------------
                    FUNCTION: MAKE AN ACCRETION DISK
    
     This function makes a simple accretion disk model with grey opacity.
    ------------------------------------------------------------------------
    '''
    def accrdiskfunc(self):
        if self.Rin is None:
            self.Rin = self.rstar()/AU
        if self.alpha is None:
            self.alpha = 0.01

        self.r = self.Rin * (self.Rout / self.Rin) ** (np.array(range(self.nr)) / (self.nr - 1)) * AU

        Omk = np.sqrt(GG*self.Mstar*Msun/self.r**3)
        Qplus = 3/(4*np.pi)*Omk**2 * self.Mdot * Msun/yr
        Teff = (Qplus*(1-np.sqrt(self.Rin*AU/self.r))/(2*ss))**0.25

        Tc = np.zeros(self.nr)
        Sigma = np.zeros(self.nr)
        for ir in range(self.nr):
            kappa = kappaplanck(self.opac, Teff[ir])
            Tc[ir] = (Qplus[ir]*mugas*mp*kappa*(self.Mdot*Msun/yr)*Omk[ir]/(8*np.pi*ss*kk*self.alpha))**0.2
            if Tc[ir] < Teff[ir]:
                raise ValueError(f'Optically thin accretion disk.... ir={ir}')

            Sigma[ir] = mugas*mp*(self.Mdot*Msun/yr)*Omk[ir]/(2*np.pi*kk*self.alpha*Tc[ir])

        self.Teff = Teff
        self.Tc = Tc
        self.Sigma = Sigma

    '''
    ------------------------------------------------------------------------
          GLUE THE DIFFUSIVE INNER RIM AND OUTER FLARING DISK TOGETHER
    
     This routine is meant to create the proper diffusive shadowed region
     in between the flaring part and the inner rim. This includes the
     diffusion from the inner rim outwards (which is already done in the
     diffedgefunc()), the diffusion from the flaring disk inwards, and a
     lower limit to the temperature. In later versions we may add effects
     of irradiation from the top of the innr rim (the 'ring source'), or
     other effects. For now we do it in this simple way.
    ------------------------------------------------------------------------
    '''
    def gluediffcg(self):
        hscg = np.zeros(self.nr)
        if self.irfullshadow < self.nr - 1:
            hscg[self.irfullshadow + 1:] = self.Hs[self.irfullshadow + 1:]
        if self.irhalfshadow < self.nr - 1:
            for ir in range(self.irhalfshadow, -1, -1):
                dum = (hscg[ir + 1] / self.r[ir + 1] - np.log(self.r[ir + 1] / self.r[ir]) / 8) * self.r[ir]
                if dum < 0:
                    dum = 0
                hscg[ir] = (dum ** 8 + hscg[ir] ** 8) ** (1 / 8.)

        hs = (hscg**8 + self.diffedge['Hsdiff']**8)**(1/8.)
        hp = np.zeros(self.nr)
        Ti = np.zeros(self.nr)
        Ts = np.zeros(self.nr)
        if self.irhalfshadow < self.nr - 1:
            hp[self.irhalfshadow+1:] = self.Hp[self.irhalfshadow+1:]
            Ti[self.irhalfshadow+1:] = self.Ti[self.irhalfshadow+1:]
            Ts[self.irhalfshadow+1:] = self.Ts[self.irhalfshadow+1:]
            alpha = self.alpha[self.irhalfshadow+1]
        else:
            alpha = self.alpha[self.nr-1]
        for ir in range(min([self.irhalfshadow, self.nr-1]), -1, -1):
            chi = findchi(kappaplanck(self.opac, self.Tstar), alpha, self.Sigma[ir])
            hp[ir] = max([hs[ir]/chi, self.diffedge['Hpdiff'][ir]])
            Ti[ir] = hp[ir]**2 * self.Tc * self.rstar() / self.r[ir]**3
            Ts[ir] = max([self.Ts[ir], Ti[ir]])

        self.Hs = hs
        self.Hp = hp
        self.Ti = Ti
        self.Ts = Ts

    '''
    ------------------------------------------------------------------------
                       FUNCTION: MAKE AN SED
    
     incl             Make the SED as seen at inclination incl.
     notrans          If set: do not take the transparancy of the disk
                      into account (which is wrong, but which could be
                      interesting to see what the effect is)
    ------------------------------------------------------------------------
    '''
    def makesed(self, incl: float=0, notrans: bool=False, nostar: bool=False, noscat: bool=False):

        if self.disk_type != 'accrdisk':
            nu = self.opac.freq
            wav = cc/nu
            nwav = len(wav)
        else:
            nwav = 200
            wav = 1e-4*0.1*10**((4*range(nwav))/(nwav-1))
            nu = cc/wav

        incrad = 0.01745329252 * incl
        cosi = np.cos(incrad)

        lnu_star = np.zeros(nwav)
        lnu_mid = np.zeros(nwav)
        lnu_surf = np.zeros(nwav)
        lnu_scat = np.zeros(nwav)
        lnu_wall = np.zeros(nwav)
        lnu_accr = np.zeros(nwav)
        lnu = np.zeros(nwav)

        lnu_star = np.pi * bplanck(nu, self.Tstar) * 4 * np.pi * self.rstar()**2

        irmaxcg = 0
        irmincg = 0
        if self.disk_type == 'cgdisk' or self.disk_type=='combi':
            irmaxcg = self.irmaxcg
        if self.disk_type == 'combi':
            irmincg = self.irhalfshadow+1

        scatspec = lnu_star / abs(np.trapz(lnu_star, x=nu))
        if (self.disk_type == 'cgdisk' or self.disk_type == 'combi') and (irmincg < irmaxcg):
            image_mid = np.zeros((nwav, self.nr))
            image_surf = np.zeros((nwav, self.nr))
            image_scat = np.zeros((nwav, self.nr))
            for ir in range(irmincg, irmaxcg+1):
                kappai=findkappa(self.opac, self.Ti[ir])
                if not notrans:
                    dil = 1 - np.exp(-self.Sigma[ir]*kappai/cosi)
                    transpf = 1 + np.exp(-self.Sigma[ir]*kappai/cosi)
                else:
                    dil = 1
                    transpf = 1
                kappas = findkappa(self.opac, self.Ts[ir])
                if (self.Ts[ir] > 0) and (self.Ts[ir] < max(self.opac.evaptemp)):
                    sigsurf = self.tausurf[ir] / kappaplanck(self.opac, self.Ts[ir])
                else:
                    sigsurf = 0
                taulay = transpf * sigsurf * kappas/cosi
                srfdil = 1 - np.exp(-taulay)
                image_mid[:, ir] = dil * bplanck(nu, self.Ti[ir])
                image_surf[:, ir] = srfdil * bplanck(nu, self.Ts[ir])
                image_scat[:, ir] = self.Fscat[ir] * scatspec / np.pi

            dum=0
            for ir in range(irmincg+1, irmaxcg+1):
                dum += 4*np.pi**2 * 0.5 * (image_mid[:,ir]+image_mid[:, ir-1]) * (self.r[ir]**2 - self.r[ir-1]**2) * cosi
            lnu_mid += dum

            dum=0
            for ir in range(irmincg+1, irmaxcg+1):
                dum += 4*np.pi**2 * 0.5 * (image_surf[:, ir]+image_surf[:, ir-1]) * (self.r[ir]**2 - self.r[ir-1]**2) * cosi
            lnu_surf += dum

            dum = 0
            for ir in range(irmincg + 1, irmaxcg + 1):
                dum += 4 * np.pi ** 2 * 0.5 * (image_scat[:, ir] + image_scat[:, ir - 1]) * (
                            self.r[ir] ** 2 - self.r[ir - 1] ** 2) * cosi
            lnu_scat += dum

        if self.diffedge is not None:
            lnu_wall = 16 * np.pi * self.r[0] * self.diffedge['Hsin'] * np.sin(incrad) * bplanck(nu, self.Tin)
            # I skip the no incl version since theres always incl

        if self.disk_type == 'accrdisk':
            for ir in range(1, self.nr-1):
                lnu_accr += bplanck(nu, self.Teff[ir]) * 4 * np.pi**2 * (self.r[ir]*self.r[ir+1] - self.r[ir]*self.r[ir-1]) * cosi
            ir = 0
            lnu_accr += bplanck(nu, self.Teff[ir]) * 4 * np.pi ** 2 * (
                        self.r[ir] * self.r[ir + 1] - self.r[ir] * self.r[ir]) * cosi
            ir=self.nr-1
            lnu_accr += bplanck(nu, self.Teff[ir]) * 4 * np.pi ** 2 * (
                        self.r[ir] * self.r[ir] - self.r[ir] * self.r[ir - 1]) * cosi

        if nostar:
            inclstar = 0
        else:
            inclstar = 1

        if noscat:
            inclscat = 0
        else:
            inclscat = 1

        lnu = inclstar*lnu_star + lnu_mid + lnu_surf + inclscat*lnu_scat + lnu_wall + lnu_accr

        l_star = abs(np.trapz(lnu_star, x=nu))
        l_mid = abs(np.trapz(lnu_mid, x=nu))
        l_surf = abs(np.trapz(lnu_surf, x=nu))
        l_scat = abs(np.trapz(lnu_scat, x=nu))
        l_wall = abs(np.trapz(lnu_wall, x=nu))
        l_accr = abs(np.trapz(lnu_accr, x=nu))
        #l_tot = l_star + l_mid + l_surf + l_scat + l_wall + l_accr #Only for no-incl case ?
        l_tot = l_star

        return {'nu': nu, 'wav': wav, 'lnu': lnu,
                'lnu_star': lnu_star, 'lnu_mid': lnu_mid, 'lnu_surf': lnu_surf,
                'lnu_scat': lnu_scat, 'lnu_wall': lnu_wall, 'lnu_accr': lnu_accr,
                'l_star': l_star, 'l_mid': l_mid, 'l_surf': l_surf,
                'l_scat': l_scat, 'l_wall': l_wall, 'l_accr': l_accr,
                'l_tot': l_tot}


    def __post_init__(self):
        rstar = self.rstar()
        nr = self.nr

        if self.disk_type == 'cgdisk':
            if self.Rin is None:
                raise ValueError("Please set R_in for CGdisk!! If you don't want to set it, please first run diffedge/choose combi")
            if self.Rin < rstar / AU:
                self.Rin = rstar / AU
            self.r = self.Rin * (self.Rout / self.Rin) ** (np.array(range(nr)) / (nr - 1)) * AU
            self.Tc = GG * self.Mstar * Msun * mugas * mp / (kk * rstar)
            self.ini_Sigma()
            self.CGDisk()
        if self.disk_type == 'diffedge':
            if self.Rin is not None:
                if self.Rin < rstar / AU:
                    warnings.warn("R_in is smaller than R_star, changing to R_star")
                    self.Rin = rstar / AU

            if self.Sigma0 is None:
                raise ValueError("Please set up Sigma0 for diffedge calculation")
            self.diffedgefunc()
        if self.disk_type == 'accrdisk':
            self.accrdiskfunc()
        if self.disk_type == 'combi':
            if self.Rin is not None:
                if self.Rin < rstar / AU:
                    warnings.warn("R_in is smaller than R_star, changing to R_star")
                    self.Rin = rstar / AU
            if self.Sigma0 is None:
                self.sigma0_from_mass()
            if not self.nopuff:
                self.diffedgefunc()
                self.Tc = GG * self.Mstar * Msun * mugas * mp / (kk * rstar)
                self.CGDisk()
            else:
                self.diffedgefunc(noselfirr=True, nodiff=True)
                self.Tc = GG * self.Mstar * Msun * mugas * mp / (kk * rstar)
                self.CGDisk()
                self.fixhs = self.Hs[0]
                self.diffedgefunc(noselfirr=self.noselfirr)
                self.CGDisk()
            self.gluediffcg()

def run_disk(nr, incl, Tin, Mstar, Lstar, Tstar, Rout, Mdisk, plsig, opac, chiwall):
    if chiwall<=0:
        chiwall=None
    a = Disk(disk_type='combi', Tin=Tin,
             Mstar=Mstar, Lstar=Lstar, Tstar=Tstar,
             Rout=Rout, Rc=1, Mdisk=Mdisk,
             plsigma=plsig, nr=nr, usepsi=True, opac=opac, chiwall=chiwall)
    dustmass = np.trapz(2*np.pi*a.Sigma*a.r, x=a.r)
    print(f"Mass of the disk (dust) = {dustmass/Msun} Msun")
    print(f"Mass of the disk (dust+gas) = {dustmass*100/Msun} Msun")
    print(f"Radius of the inner rim = {a.Rin} AU")
    print(f"Covering fraction by the wall = {a.diffedge['Hsin']/a.r[0]} *4pi")
    print(f"Inner wall Hs/Hp = {a.diffedge['chirim']}")
    print(f"Covering fraction by CG disk = {a.Hs[a.irmaxcg]/a.r[a.irmaxcg]}")
    print(f"Full shadow at ir={a.irfullshadow}, R={a.r[a.irfullshadow]/AU} AU")
    print(f"Half shadow at ir={a.irhalfshadow}, R={a.r[a.irhalfshadow]/AU} AU")

    psii_mask = a.psii <= 0.95
    psis_mask = a.psis <= 0.95
    if True in psii_mask:
        print(f'Disk optthin to own rad at  R = {a.r[psii_mask][0]/AU} AU')
    if True in psis_mask:
        print(f'Disk optthin to surf rad at  R = {a.r[psis_mask][0]/AU} AU')
    print(f'Energy conservation error = {find_encons_error(a.r[a.irhalfshadow+1:], a.Hs[a.irhalfshadow+1:], a.flareindex[a.irhalfshadow+1:])-1}')
    sed = a.makesed(incl=incl)
    print(f"Full luminosity = {sed['l_tot']/Lsun}")
    print(f"Relative luminosities:")
    print(f"L_wall/L_tot = {sed['l_wall']/sed['l_tot']}")
    print(f"L_mid/L_tot = {sed['l_mid']/sed['l_tot']}")
    print(f"L_surf/L_tot = {sed['l_surf']/sed['l_tot']}")
    print(f"L_scat/L_tot = {sed['l_scat']/sed['l_tot']}")
    print(f"L_accr/L_tot = {sed['l_accr']/sed['l_tot']}")

    return [a, sed]

'''
========================================================================
                       GRAPHICAL INTERFACE (Qt)
  Everything above this line is the computational core and is unchanged.
  The interface below only calls OpacityTable, run_disk, findkappa and
  findtauradii, and plots their results.

  Needs one Qt binding: PySide6 (Python 3.9+), or PyQt5 / PySide2,
  which also work on Python 3.8.   pip install PyQt5
  Run it from the folder that contains opac/.
========================================================================
'''
import io
import re
import sys
import json
import time
import queue
import threading
import contextlib
import traceback
import importlib

# ---------------------------------------------------------------- palette
INK = '#1E2235'          # sidebar, primary text, total-flux curve
INK_2 = '#2A2F47'        # sidebar inputs
INK_3 = '#3B415E'        # sidebar borders
SIDE_TEXT = '#EEF0F7'
SIDE_LABEL = '#CDD1E0'
SIDE_MUTED = '#9097B2'
BLUE = '#5BCEFA'         # trans flag blue: primary actions, focus
BLUE_HOVER = '#80DAFB'
BLUE_PRESS = '#3FBEEE'
BLUE_PALE = '#DDF3FD'
BLUE_PALE_2 = '#C8EBFB'
PINK = '#F5A9B8'         # trans flag pink: headings, secondary actions
PINK_DEEP = '#E58FA2'
PINK_PALE = '#FCE6EB'
PINK_PALE_2 = '#F9D3DC'
WHITE = '#FFFFFF'
CANVAS_BG = '#F3F6FA'
CARD = '#FFFFFF'
CARD_LINE = '#DFE4EC'
TEXT = '#1E2235'
TEXT_2 = '#5D6479'
CHIP = '#EBEFF5'
CHIP_HOVER = '#DFE5EE'
ERROR = '#D0533F'
OK = '#3AAE8C'
STALE = PINK_DEEP

# curve colours: pastel family, deepened enough to read on white
C_STAR = '#E8A33D'
C_WALL = '#E07A93'
C_MID = '#3A9FD8'
C_SURF = '#8E7CC3'
C_OBS = '#8A8F99'

# ---------------------------------------------------------------- parameters
# (key, label, unit, default, tooltip)
SECTIONS = [
    ('Star', [
        ('ms', 'Mass', 'Msun', '2.4', None),
        ('ls', 'Luminosity', 'Lsun', '47.0', None),
        ('ts', 'Temperature', 'K', '9520.0', None),
    ]),
    ('Disk', [
        ('mdisk', 'Mass', 'Msun', '0.05', None),
        ('rout', 'Outer radius', 'AU', '400', 'Outer radius of the disk.'),
        ('plsig', 'Sigma power law', '', '-2.0',
         'Power-law exponent of the surface density:\nSigma(R) = Sigma0 * (R/Rc)^plsig'),
        ('tin', 'Rim temperature', 'K', '1500', 'Temperature of the inner rim (T_in).'),
        ('chi', 'Rim Hs/Hp', '', '0',
         'Ratio of surface to pressure scale height, inner rim only.\n'
         'Set to 0 or below to let the model compute it.'),
    ]),
    ('Observer', [
        ('dist', 'Distance', 'pc', '144.0', None),
        ('incl', 'Inclination', 'deg', '60.0', None),
    ]),
    ('Dust composition', [
        ('silicate', 'Silicate', '', '1.00', None),
        ('forsterite', 'Forsterite', '', '0.00', None),
        ('silbeta1', 'Silbeta1', '', '0.00', None),
        ('carbon', 'Carbon', '', '0.00', None),
        ('ice', 'Ice', '', '0.00', None),
        ('albedo', 'Albedo', '', '0',
         'Kept for model files only: scattering is turned off,\nso this value has no effect.'),
    ]),
    ('Grid', [
        ('nr', 'Radial cells', '', '30', 'Number of radial cells in the grid (N_r).'),
    ]),
]
DUST_KEYS = ['silicate', 'forsterite', 'silbeta1', 'carbon', 'ice']
DUST_TIP = 'Relative abundance. 0 leaves the species out;\nabundances are normalised to their sum.'
INT_KEYS = {'nr'}
LABELS = {k: lab for _, rows in SECTIONS for k, lab, *_ in rows}
DEFAULTS = {k: d for _, rows in SECTIONS for k, _, _, d, _ in rows}

# The .model file layout, identical to the original interface
MODEL_FILE_LINES = [
    ('nr', 'Nr:      '), ('dist', 'Distance[pc]:       '),
    ('incl', 'Inclination[degrees]:       '), ('ms', 'Starmass[MS]:       '),
    ('ls', 'Starluminosity[LS]:       '), ('ts', 'Startemperature[K]:       '),
    ('rout', 'Outer_disk-radius[AU]:       '), ('mdisk', 'Mdisk:    '),
    ('plsig', 'Plsigma:      '), ('silicate', 'silicate:       '),
    ('carbon', 'carbon:       '), ('forsterite', 'forsterite:       '),
    ('ice', 'ice:       '), ('silbeta1', 'silbeta1:       '),
    ('tin', 'Inner Disk temp.:       '), ('chi', 'Chi:       '),
    ('albedo', 'Albedo:       '), ('ymin', 'Yrange[0]:   '), ('ymax', 'Yrange[1]:   '),
]

SED_YLABEL = r'$\nu F_\nu$ (erg/s/cm$^2$)'
WAV_XLABEL = r'$\lambda$ ($\mu$m)'

SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
PREFS_PATH = os.path.join(os.path.expanduser('~'), '.cgplusplus.json')
UI = {'scale': 1.0}


def px(value):
    '''Scale a size given in pixels at 100 % to the current interface size.'''
    return max(1, int(round(value * UI['scale'])))


def load_prefs():
    try:
        with open(PREFS_PATH) as f:
            prefs = json.load(f)
        return prefs if isinstance(prefs, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(prefs):
    try:
        with open(PREFS_PATH, 'w') as f:
            json.dump(prefs, f, indent=1)
    except OSError:
        pass


def auto_scale(app):
    '''Qt already scales for the OS where the OS tells it to; this covers the rest
    (typically X11 reporting 96 dpi on a high-resolution panel).'''
    screen = app.primaryScreen()
    width = screen.geometry().width() if screen else 1440
    guess = max(1.0, width / 1440)
    return min(SCALES, key=lambda s: abs(s - guess))


def format_scale(s):
    return f'{int(round(s * 100))}%'


def stylesheet(family):
    f = f'"{family}"'
    return f'''
    * {{ font-family: {f}; font-size: {px(13)}px; color: {TEXT}; }}
    QMainWindow, QWidget#main {{ background: {CANVAS_BG}; }}
    QToolTip {{ background: {WHITE}; color: {TEXT}; border: {px(2)}px solid {PINK};
                padding: {px(6)}px {px(8)}px; font-size: {px(12)}px; }}

    QWidget#sidebar, QWidget#sidebarBody, QScrollArea#sideScroll {{ background: {INK}; border: none; }}
    QWidget#sidebar QLabel {{ color: {SIDE_LABEL}; background: transparent; }}
    QWidget#sidebar QLabel#brand {{ color: {SIDE_TEXT}; font-size: {px(29)}px; font-weight: bold; }}
    QWidget#sidebar QLabel#tagline, QWidget#sidebar QLabel#unit, QWidget#sidebar QLabel#hint {{ color: {SIDE_MUTED}; font-size: {px(12)}px; }}
    QWidget#sidebar QLabel#section {{ color: {PINK}; font-weight: bold; padding-top: {px(10)}px; }}
    QWidget#sidebar QLabel#badge {{ color: {SIDE_TEXT}; background: {INK_3}; border-radius: {px(8)}px;
                    font-size: {px(11)}px; font-weight: bold;
                    min-width: {px(16)}px; max-width: {px(16)}px; min-height: {px(16)}px; max-height: {px(16)}px; }}
    QLineEdit#param {{ background: {INK_2}; color: {SIDE_TEXT}; border: {px(1)}px solid {INK_3};
                       border-radius: {px(4)}px; padding: {px(4)}px {px(7)}px;
                       selection-background-color: {BLUE}; selection-color: {INK}; }}
    QLineEdit#param:focus {{ border-color: {BLUE}; }}
    QLineEdit#param[error="true"] {{ border-color: {ERROR}; }}
    QFrame#rule {{ background: {INK_3}; max-height: {px(1)}px; min-height: {px(1)}px; }}

    QScrollBar:vertical {{ background: transparent; width: {px(8)}px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {INK_3}; border-radius: {px(4)}px; min-height: {px(30)}px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QPlainTextEdit QScrollBar::handle:vertical {{ background: {CHIP_HOVER}; }}

    QPushButton {{ border: none; border-radius: {px(6)}px; padding: {px(6)}px {px(12)}px; }}
    QPushButton:focus {{ outline: none; }}
    QPushButton#primary {{ background: {BLUE}; color: {INK}; font-size: {px(15)}px; font-weight: bold;
                           padding: {px(10)}px; }}
    QPushButton#primary:hover {{ background: {BLUE_HOVER}; }}
    QPushButton#primary:pressed {{ background: {BLUE_PRESS}; }}
    QPushButton#primary:disabled {{ background: {INK_3}; color: {SIDE_MUTED}; }}
    QPushButton#dark {{ background: {INK_2}; color: {SIDE_TEXT}; padding: {px(7)}px {px(6)}px; }}
    QPushButton#dark:hover {{ background: {INK_3}; }}
    QPushButton#pink {{ background: {PINK_PALE}; }}
    QPushButton#pink:hover {{ background: {PINK_PALE_2}; }}
    QPushButton#pink:pressed {{ background: {PINK}; }}
    QPushButton#ghost {{ background: transparent; color: {TEXT_2}; padding: {px(4)}px {px(10)}px; }}
    QPushButton#ghost:hover {{ background: {CHIP}; }}
    QPushButton#apply {{ background: {BLUE}; color: {INK}; font-weight: bold; }}
    QPushButton#apply:hover {{ background: {BLUE_HOVER}; }}

    QFrame#segbox {{ background: {CHIP}; border-radius: {px(6)}px; }}
    QPushButton#seg {{ background: transparent; color: {TEXT_2}; border-radius: {px(4)}px;
                       padding: {px(4)}px {px(11)}px; }}
    QPushButton#seg:hover {{ background: {CHIP_HOVER}; }}
    QPushButton#seg:checked {{ background: {BLUE_PALE}; color: {TEXT}; }}

    QFrame#card {{ background: {CARD}; border: {px(1)}px solid {CARD_LINE}; border-radius: {px(10)}px; }}
    QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
    QLabel#title {{ font-size: {px(16)}px; font-weight: bold; }}
    QLabel#muted {{ color: {TEXT_2}; }}
    QLabel#key {{ color: {TEXT_2}; font-size: {px(12)}px; }}
    QLabel#value {{ font-weight: bold; }}
    QLabel#status {{ color: {TEXT_2}; }}
    QLabel#status[error="true"] {{ color: {ERROR}; }}
    QLineEdit#yrange {{ background: {CHIP}; border: {px(1)}px solid {CHIP}; border-radius: {px(4)}px;
                        padding: {px(3)}px {px(6)}px; }}
    QLineEdit#yrange:focus {{ border-color: {BLUE}; }}
    QLineEdit#yrange[error="true"] {{ border-color: {ERROR}; }}
    QCheckBox {{ spacing: {px(6)}px; }}
    QCheckBox::indicator {{ width: {px(14)}px; height: {px(14)}px; border-radius: {px(3)}px;
                           border: {px(1)}px solid #B9C0CC; background: {WHITE}; }}
    QCheckBox::indicator:hover {{ border-color: {PINK_DEEP}; }}
    QCheckBox::indicator:checked {{ background: {PINK}; border-color: {PINK_DEEP}; }}
    QPlainTextEdit {{ background: #F6F8FB; border: none; border-radius: {px(6)}px;
                      font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: {px(12)}px;
                      padding: {px(6)}px; }}
    QToolBar {{ background: transparent; border: none; spacing: {px(2)}px; }}
    QToolButton {{ background: transparent; border: none; border-radius: {px(4)}px; padding: {px(3)}px; }}
    QToolButton:hover {{ background: {BLUE_PALE}; }}
    QToolButton:checked {{ background: {BLUE_PALE_2}; }}
    QSplitter::handle {{ background: transparent; }}

    QDialog {{ background: {CARD}; }}
    QComboBox, QFontComboBox {{ background: {CHIP}; border: none; border-radius: {px(5)}px;
                                padding: {px(5)}px {px(8)}px; min-width: {px(260)}px; }}
    QComboBox QAbstractItemView {{ background: {WHITE}; selection-background-color: {BLUE_PALE};
                                   selection-color: {TEXT}; }}
    QLabel#preview {{ background: {PINK_PALE}; padding: {px(9)}px {px(10)}px; border-radius: {px(6)}px; }}
    '''


# ======================================================================
#                     LOADING QT, AND THE SELF-TEST
# ======================================================================
# Libraries Qt takes from the operating system (they belong to the graphics driver, so packaged
# apps do not bundle them), and the Ubuntu/Debian package that provides each.
SYSTEM_LIBS = {'libEGL': 'libegl1', 'libGL': 'libgl1', 'libGLX': 'libglx0', 'libOpenGL': 'libopengl0',
               'libxcb': 'libxcb1', 'libxcb-cursor': 'libxcb-cursor0', 'libxkbcommon': 'libxkbcommon0',
               'libxkbcommon-x11': 'libxkbcommon-x11-0', 'libfontconfig': 'libfontconfig1', 'libdbus-1': 'libdbus-1-3'}


def load_qt():
    '''Import the first Qt binding that works: (QtCore, QtGui, QtWidgets, is_qt6).
    If none does, raise ImportError with the real reason in plain words.'''
    broken = []
    for name in ('PySide6', 'PyQt5', 'PySide2'):
        try:
            importlib.import_module(name)
        except ImportError:
            continue   # this binding is not installed
        try:
            mods = [importlib.import_module(f'{name}.{m}') for m in ('QtCore', 'QtGui', 'QtWidgets')]
            return mods[0], mods[1], mods[2], name == 'PySide6'
        except ImportError as err:
            broken.append((name, err))
    if not broken:
        raise ImportError('CG++ needs Qt for Python. Install it with:  pip install PyQt5   (or PySide6)')
    name, err = broken[0]
    msg = f'Qt ({name}) could not start: {err}'
    lib = re.search(r'(lib[\w+-]+?)\.so', str(err))
    if lib:
        pkg = SYSTEM_LIBS.get(lib.group(1))
        msg += (f'\n\nThis system is missing {lib.group(1)}.so, which Qt needs to draw windows. '
                'Desktop Linux systems normally have it; minimal installs and servers may not.')
        msg += f'\nOn Ubuntu or Debian install it with:  sudo apt install {pkg}' if pkg else \
            '\nInstall it with your package manager.'
    raise ImportError(msg)


def selftest():
    '''cgplusplus --selftest: compute the default model and check that Qt loads, without opening a window.
    Used to check packaged builds; prints the results and returns the exit code.'''
    model_ok = qt_ok = False
    try:
        p = {k: (int(v) if k in INT_KEYS else float(v)) for k, v in DEFAULTS.items()}
        names = [k for k in DUST_KEYS if p[k] > 0]
        opac = OpacityTable(dnames=names, abun=[p[k] for k in names])
        a, sed = run_disk(p['nr'], p['incl'], p['tin'], p['ms'], p['ls'], p['ts'],
                          p['rout'], p['mdisk'], p['plsig'], opac, p['chi'])
        model_ok = bool(np.all(np.isfinite(sed['lnu'])) and sed['lnu'].max() > 0)
        print('Model:', 'ok' if model_ok else 'FAILED (non-finite or empty SED)')
    except Exception:
        print('Model: FAILED')
        traceback.print_exc()
    try:
        core, _, widgets, _ = load_qt()
        app = widgets.QApplication(['cgplusplus', '-platform', 'offscreen'])   # also loads Qt's plugins
        print('Qt:', core.qVersion(), 'ok')
        qt_ok = True
        del app
    except ImportError as err:
        print('Qt: FAILED')
        print(err)
    passed = model_ok and qt_ok
    print('CG++ self-test', 'passed' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__' and '--selftest' in sys.argv:
    sys.exit(selftest())

try:
    QtCore, QtGui, QtWidgets, QT6 = load_qt()
except ImportError as _err:
    sys.exit(str(_err))

# imported after the binding, so matplotlib picks the same one
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:   # matplotlib older than 3.5
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import font_manager

Qt = QtCore.Qt
QShortcut = getattr(QtGui, 'QShortcut', None) or QtWidgets.QShortcut


def qexec(obj):
    '''exec() for every binding (PySide2 only has exec_).'''
    return obj.exec() if hasattr(obj, 'exec') else obj.exec_()



# ======================================================================
#                             SMALL WIDGETS
# ======================================================================
def button(text, slot, name='pink'):
    b = QtWidgets.QPushButton(text)
    b.setObjectName(name)
    b.setCursor(Qt.PointingHandCursor)
    b.clicked.connect(slot)
    return b


def label(text, name=None, tip=None):
    lab = QtWidgets.QLabel(text)
    if name:
        lab.setObjectName(name)
    if tip:
        lab.setToolTip(tip)
    return lab


def set_error(widget, on):
    widget.setProperty('error', 'true' if on else 'false')
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class Segmented(QtWidgets.QFrame):
    '''A segmented control; calls on_change(value) when the user picks an option.'''

    def __init__(self, options, value, on_change=None):
        super().__init__()
        self.setObjectName('segbox')
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(px(2), px(2), px(2), px(2))
        lay.setSpacing(px(2))
        self.group = QtWidgets.QButtonGroup(self)
        self.buttons = {}
        for val, text in options:
            b = QtWidgets.QPushButton(text)
            b.setObjectName('seg')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(val == value)
            b.clicked.connect(lambda _=False, v=val: on_change and on_change(v))
            self.group.addButton(b)
            self.buttons[val] = b
            lay.addWidget(b)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    def value(self):
        for val, b in self.buttons.items():
            if b.isChecked():
                return val
        return None


class Toolbar(NavigationToolbar2QT):
    # the standard matplotlib toolbar without the subplot and figure-options dialogs
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] not in ('Subplots', 'Customize')]


def hbox(*items, spacing=8, margins=(0, 0, 0, 0)):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(*[px(m) for m in margins])
    lay.setSpacing(px(spacing))
    for it in items:
        if it is None:
            lay.addStretch(1)
        elif isinstance(it, int):
            lay.addSpacing(px(it))
        else:
            lay.addWidget(it)
    return w


class FlowLayout(QtWidgets.QLayout):
    '''Lays widgets out left to right and wraps them onto new lines when space runs out.'''

    def __init__(self, parent=None, hspacing=8, vspacing=8):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self._items, self._h, self._v = [], px(hspacing), px(vspacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        try:
            return Qt.Orientations(0)
        except (AttributeError, TypeError):
            return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            if item.widget() is None or item.widget().isVisibleTo(self.parentWidget()):
                size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect, apply):
        lines, line, x = [], [], rect.x()
        for item in self._items:
            w = item.widget()
            if w is not None and not w.isVisibleTo(self.parentWidget()):
                continue
            hint = item.sizeHint()
            if line and x + hint.width() > rect.right() + 1:
                lines.append(line)
                line, x = [], rect.x()
            line.append((item, x, hint))
            x += hint.width() + self._h
        if line:
            lines.append(line)
        y = rect.y()
        for line in lines:
            height = max(h.height() for _, _, h in line)
            if apply:   # centre every item vertically within its line
                for item, x, hint in line:
                    item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y + (height - hint.height()) // 2), hint))
            y += height + self._v
        return max(0, y - self._v - rect.y())


def flow(*widgets, hspacing=12, vspacing=8):
    w = QtWidgets.QWidget()
    lay = FlowLayout(w, hspacing, vspacing)
    for it in widgets:
        lay.addWidget(it)
    return w


# ======================================================================
#                              APPLICATION
# ======================================================================
class CGWindow(QtWidgets.QMainWindow):
    def __init__(self, app, scale=None):
        super().__init__()
        self.app = app
        self.setWindowTitle('CG++')
        self.prefs = load_prefs()
        self.auto = auto_scale(app)
        self.scale = scale or self.prefs.get('scale', self.auto)
        self.font_choice = self.prefs.get('font', '')

        # model state, kept across rebuilds of the interface
        self.values = dict(DEFAULTS)
        self.ymin, self.ymax = '1e-11', '1e-7'
        self.flux_mode, self.yscale, self.struct = 'total', 'log', 'hr'
        self.obs_style, self.show_obs, self.fit_all = 'line', True, False
        self.a = self.sed = self.red = self.rtau = None
        self.obs = self.obs_name = self.model_name = None
        self.last_run = None
        self.last_log, self.last_msgs = '', []
        self.status = ('', 'idle')
        self.busy = False
        self.showing_log = False
        self.results = queue.Queue()

        self._shortcuts()
        self.apply_scale(self.scale, first=True)
        QtCore.QTimer.singleShot(150, self.run_model)

    # ------------------------------------------------------------ scale & fonts
    def family(self):
        try:
            families = QtGui.QFontDatabase.families()      # Qt 6: static
        except TypeError:
            families = QtGui.QFontDatabase().families()    # Qt 5
        if self.font_choice and self.font_choice in families:
            return self.font_choice
        return self.app.font().family()

    def apply_scale(self, scale, first=False):
        UI['scale'] = self.scale = scale
        fam = self.family()
        self.app.setStyleSheet(stylesheet(fam))
        self._setup_matplotlib(fam)
        screen = self.app.primaryScreen().availableGeometry()
        w, h = min(px(1440), int(screen.width() * 0.94)), min(px(880), int(screen.height() * 0.9))
        if first:
            self.resize(w, h)
            self.move(screen.x() + (screen.width() - w) // 2, screen.y() + (screen.height() - h) // 3)
        elif not self.isMaximized():
            self.resize(w, h)
        self.build()

    def _setup_matplotlib(self, family):
        s = self.scale
        mpl_names = {f.name for f in font_manager.fontManager.ttflist}
        matplotlib.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [family if family in mpl_names else 'DejaVu Sans', 'DejaVu Sans'],
            'font.size': 9 * s, 'axes.labelsize': 10 * s, 'xtick.labelsize': 9 * s, 'ytick.labelsize': 9 * s,
            'legend.fontsize': 9 * s, 'lines.linewidth': 1.6 * s, 'lines.markersize': 6 * s,
            'axes.linewidth': 0.8 * s, 'xtick.major.size': 3.5 * s, 'ytick.major.size': 3.5 * s,
            'xtick.minor.size': 2 * s, 'ytick.minor.size': 2 * s, 'xtick.major.width': 0.8 * s,
            'ytick.major.width': 0.8 * s, 'xtick.minor.width': 0.6 * s, 'ytick.minor.width': 0.6 * s,
            'lines.markeredgewidth': 1.0 * s, 'errorbar.capsize': 0,
            'axes.edgecolor': '#A3ABBA', 'axes.labelcolor': '#3B415E',
            'xtick.color': '#5D6479', 'ytick.color': '#5D6479',
            'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
            'legend.frameon': False, 'figure.facecolor': CARD, 'axes.facecolor': CARD,
        })

    def zoom(self, step):
        i = min(range(len(SCALES)), key=lambda k: abs(SCALES[k] - self.scale))
        i = max(0, min(len(SCALES) - 1, i + step))
        if SCALES[i] != self.scale:
            self.prefs['scale'] = SCALES[i]
            save_prefs(self.prefs)
            self.apply_scale(SCALES[i])
            self.set_status(f'Interface size {format_scale(SCALES[i])}.', 'ok')

    def _shortcuts(self):
        def add(seq, slot):
            QShortcut(QtGui.QKeySequence(seq), self).activated.connect(slot)
        add('Ctrl+R', self.run_model)
        add('Ctrl+Return', self.run_model)
        add(QtGui.QKeySequence.Open, self.open_model)
        add(QtGui.QKeySequence.Save, self.save_model)
        add(QtGui.QKeySequence.Quit, self.close)
        add(QtGui.QKeySequence.ZoomIn, lambda: self.zoom(+1))
        add('Ctrl+=', lambda: self.zoom(+1))
        add(QtGui.QKeySequence.ZoomOut, lambda: self.zoom(-1))

    # ------------------------------------------------------------ building
    def build(self):
        old = self.centralWidget()
        root = QtWidgets.QWidget()
        root.setObjectName('main')
        lay = QtWidgets.QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.entries = {}
        lay.addWidget(self._build_sidebar())
        lay.addWidget(self._build_main(), 1)
        self.setCentralWidget(root)
        if old is not None:
            old.deleteLater()
        self._restore()

    def _build_sidebar(self):
        side = QtWidgets.QWidget()
        side.setObjectName('sidebar')
        side.setFixedWidth(px(310))
        outer = QtWidgets.QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(head)
        hl.setContentsMargins(px(22), px(18), px(22), px(10))
        hl.setSpacing(px(2))
        hl.addWidget(label('CG++', 'brand'))
        hl.addWidget(label('Passive flared disk model', 'tagline'))
        hl.addSpacing(px(12))
        for colour in (BLUE, PINK, WHITE, PINK, BLUE):   # a thin flag band under the name
            stripe = QtWidgets.QFrame()
            stripe.setFixedHeight(px(3))
            stripe.setStyleSheet(f'background: {colour}; border: none;')
            hl.addWidget(stripe)
        hl.setSpacing(0)
        outer.addWidget(head)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName('sideScroll')
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        body.setObjectName('sidebarBody')
        grid = QtWidgets.QGridLayout(body)
        grid.setContentsMargins(px(22), px(4), px(14), px(12))
        grid.setHorizontalSpacing(px(6))
        grid.setVerticalSpacing(px(4))
        grid.setColumnStretch(0, 1)
        row = 0
        for title, fields in SECTIONS:
            grid.addWidget(label(title, 'section'), row, 0, 1, 3)
            row += 1
            for key, text, unit, _, tip in fields:
                if key in DUST_KEYS:
                    tip = DUST_TIP
                cell = [label(text)]
                if tip:
                    badge = label('i', 'badge', tip)
                    badge.setAlignment(Qt.AlignCenter)
                    badge.setCursor(Qt.WhatsThisCursor)
                    cell.append(badge)
                cell.append(None)
                grid.addWidget(hbox(*cell, spacing=6), row, 0)
                ent = QtWidgets.QLineEdit(self.values[key])
                ent.setObjectName('param')
                ent.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                ent.setFixedWidth(px(92))
                ent.textChanged.connect(lambda txt, k=key: self._on_edit(k, txt))
                ent.returnPressed.connect(self.run_model)
                grid.addWidget(ent, row, 1)
                u = label(unit, 'unit')
                u.setFixedWidth(px(34))
                grid.addWidget(u, row, 2)
                self.entries[key] = ent
                row += 1
        grid.setRowStretch(row, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        foot = QtWidgets.QWidget()
        fl = QtWidgets.QGridLayout(foot)
        fl.setContentsMargins(px(22), px(14), px(22), px(16))
        fl.setHorizontalSpacing(px(8))
        fl.setVerticalSpacing(px(8))
        rule = QtWidgets.QFrame()
        rule.setObjectName('rule')
        fl.addWidget(rule, 0, 0, 1, 2)
        self.run_btn = button('Run model', self.run_model, 'primary')
        fl.addWidget(self.run_btn, 1, 0, 1, 2)
        fl.addWidget(button('Open model...', self.open_model, 'dark'), 2, 0)
        fl.addWidget(button('Save model...', self.save_model, 'dark'), 2, 1)
        hint = label('Press Enter in any field to run', 'hint')
        hint.setAlignment(Qt.AlignCenter)
        fl.addWidget(hint, 3, 0, 1, 2)
        outer.addWidget(foot)
        return side

    def _card(self, title, *right):
        card = QtWidgets.QFrame()
        card.setObjectName('card')
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(px(16), px(12), px(16), px(10))
        lay.setSpacing(px(8))
        lay.addWidget(hbox(label(title, 'title'), None, *right))
        return card, lay

    def _plot(self, lay):
        fig = Figure(dpi=100)
        try:
            fig.set_layout_engine('constrained')
        except AttributeError:   # matplotlib older than 3.6
            fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        canvas.setMinimumHeight(px(160))
        toolbar = Toolbar(canvas, None, coordinates=False)
        toolbar.setIconSize(QtCore.QSize(px(20), px(20)))
        toolbar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        coords = label('', 'key')
        coords.setFixedWidth(px(170))

        def show_coords(event):
            if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
                coords.setText(f'x {event.xdata:.4g}   y {event.ydata:.4g}')
            else:
                coords.setText('')
        canvas.mpl_connect('motion_notify_event', show_coords)
        lay.addWidget(canvas, 1)
        return fig, ax, canvas, hbox(toolbar, coords, spacing=10)

    def _build_main(self):
        main = QtWidgets.QWidget()
        main.setObjectName('main')
        ml = QtWidgets.QVBoxLayout(main)
        ml.setContentsMargins(px(18), px(14), px(18), px(16))
        ml.setSpacing(px(12))

        # top bar
        self.name_lbl = label('Untitled model', 'title')
        self.dot = QtWidgets.QLabel()
        self.dot.setFixedSize(px(9), px(9))
        self.status_lbl = label('', 'status')
        ml.addWidget(hbox(self.name_lbl, 10, self.dot, self.status_lbl, None,
                          button('Settings', self.show_settings, 'ghost'),
                          button('Help', self.show_help, 'ghost'), spacing=6))

        # ---- SED card
        sed_card, sl = self._card('Spectral energy distribution', button('Save SED...', self.save_sed))
        self.seg_flux = Segmented([('total', 'Total flux'), ('components', 'Components')], self.flux_mode,
                                  lambda v: self._set('flux_mode', v))
        self.seg_scale = Segmented([('log', 'Log'), ('linear', 'Linear')], self.yscale,
                                   lambda v: self._set('yscale', v))

        self.obs_lbl = label('No observations loaded', 'muted')
        self.chk_obs = QtWidgets.QCheckBox('Show')
        self.chk_obs.setChecked(self.show_obs)
        self.chk_obs.toggled.connect(lambda v: self._set('show_obs', v))
        self.seg_obs = Segmented([('line', 'Line'), ('symbol', 'Symbols')], self.obs_style,
                                 lambda v: self._set('obs_style', v))
        self.obs_remove = button('Remove', self.clear_obs, 'ghost')
        self.obs_controls = [self.chk_obs, self.seg_obs, self.obs_remove]
        sl.addWidget(flow(self.seg_flux, self.seg_scale))
        sl.addWidget(flow(button('Load observations...', self.open_obs), self.obs_lbl,
                          self.chk_obs, self.seg_obs, self.obs_remove))

        self.fig_sed, self.ax_sed, self.cv_sed, tb = self._plot(sl)
        self.ymin_ed, self.ymax_ed = QtWidgets.QLineEdit(self.ymin), QtWidgets.QLineEdit(self.ymax)
        for ed in (self.ymin_ed, self.ymax_ed):
            ed.setObjectName('yrange')
            ed.setAlignment(Qt.AlignCenter)
            ed.setFixedWidth(px(72))
            ed.editingFinished.connect(self._range_edited)
        self.chk_fit = QtWidgets.QCheckBox('Fit to model')
        self.chk_fit.setChecked(self.fit_all)
        self.chk_fit.toggled.connect(lambda v: self._set('fit_all', v))
        yrange = hbox(label('y', 'muted'), self.ymin_ed, label('to', 'muted'), self.ymax_ed, 6,
                      self.chk_fit, button('Reset', self.reset_range, 'ghost'), spacing=6)
        sl.addWidget(flow(tb, yrange, hspacing=16))

        # ---- structure card
        st_card, stl = self._card('Disk structure')
        self.seg_struct = Segmented([('hr', 'Height'), ('T', 'Temperature'), ('sigma', 'Density'),
                                     ('tau', 'tau = 1'), ('opac', 'Opacity')], self.struct,
                                    lambda v: self._set('struct', v))
        stl.addWidget(hbox(self.seg_struct, None))
        self.fig_st, self.ax_st, self.cv_st, tb2 = self._plot(stl)
        stl.addWidget(tb2)

        # ---- summary card
        self.log_btn = button('Solver log', self.toggle_log, 'ghost')
        sm_card, sml = self._card('Model summary', self.log_btn)
        self.stack = QtWidgets.QStackedWidget()
        self.sum_widget = QtWidgets.QWidget()
        self.sum_grid = QtWidgets.QGridLayout(self.sum_widget)
        self.sum_grid.setContentsMargins(0, 0, 0, 0)
        self.sum_grid.setHorizontalSpacing(px(10))
        self.sum_grid.setVerticalSpacing(px(3))
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.stack.addWidget(self.sum_widget)
        self.stack.addWidget(self.log_text)
        sml.addWidget(self.stack, 1)

        right = QtWidgets.QSplitter(Qt.Vertical)
        right.setChildrenCollapsible(False)
        right.setHandleWidth(px(14))
        right.addWidget(st_card)
        right.addWidget(sm_card)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setSizes([px(500), px(260)])

        split = QtWidgets.QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(px(14))
        split.addWidget(sed_card)
        split.addWidget(right)
        split.setStretchFactor(0, 11)
        split.setStretchFactor(1, 9)
        split.setSizes([px(640), px(520)])
        ml.addWidget(split, 1)
        return main

    def _restore(self):
        '''Put the model state back into a freshly built interface.'''
        self.name_lbl.setText(self.model_name or 'Untitled model')
        self.set_status(*self.status)
        self.run_btn.setEnabled(not self.busy)
        self.run_btn.setText('Computing...' if self.busy else 'Run model')
        self._obs_ui()
        self._write_log(self.last_log, self.last_msgs)
        self._fill_summary(self.last_log)
        self.stack.setCurrentIndex(1 if self.showing_log else 0)
        self.log_btn.setText('Summary' if self.showing_log else 'Solver log')
        if self.sed is None:
            for ax, msg in ((self.ax_sed, 'Computing model...'), (self.ax_st, '')):
                ax.set_axis_off()
                ax.text(0.5, 0.5, msg, ha='center', va='center', color=TEXT_2, transform=ax.transAxes)
        else:
            self.redraw_sed()
            self.redraw_structure()

    def _set(self, attr, value):
        setattr(self, attr, value)
        if attr == 'struct':
            self.redraw_structure()
        else:
            self.redraw_sed()

    # ------------------------------------------------------------ status
    def set_status(self, text, kind='idle'):
        self.status = (text, kind)
        color = {'idle': SIDE_MUTED, 'busy': BLUE, 'ok': OK, 'stale': STALE, 'error': ERROR}[kind]
        self.dot.setStyleSheet(f'background: {color}; border-radius: {px(9) // 2}px;')
        self.status_lbl.setText(text)
        set_error(self.status_lbl, kind == 'error')

    def _on_edit(self, key, text):
        self.values[key] = text
        ent = self.entries.get(key)
        if ent is not None and ent.property('error') == 'true':
            set_error(ent, False)
        if self.last_run is not None and not self.busy:
            self.set_status('Parameters changed. Run the model to update.', 'stale')

    # ------------------------------------------------------------ inputs
    def collect(self):
        '''Read and validate every field. Returns a dict, or None after flagging errors.'''
        values, bad = {}, []
        for key, raw in self.values.items():
            try:
                num = float(raw.strip())
                if not np.isfinite(num):
                    raise ValueError
                if key in INT_KEYS:
                    if num != int(num) or num < 2:
                        raise ValueError
                    num = int(num)
                values[key] = num
            except ValueError:
                bad.append(key)
                set_error(self.entries[key], True)
        if bad:
            what = LABELS[bad[0]].lower()
            hint = 'a whole number of at least 2' if bad[0] in INT_KEYS else 'a number'
            more = f' ({len(bad) - 1} more field{"s" if len(bad) > 2 else ""} need fixing)' if len(bad) > 1 else ''
            self.set_status(f'{what.capitalize()} must be {hint}{more}.', 'error')
            self.entries[bad[0]].setFocus()
            return None
        if not any(values[k] > 0 for k in DUST_KEYS):
            for k in DUST_KEYS:
                set_error(self.entries[k], True)
            self.set_status('Give at least one dust species an abundance above 0.', 'error')
            return None
        return values

    # ------------------------------------------------------------ running
    def run_model(self):
        if self.busy:
            return
        params = self.collect()
        if params is None:
            return
        self.busy = True
        self.run_btn.setEnabled(False)
        self.run_btn.setText('Computing...')
        self.set_status('Computing model...', 'busy')
        self.app.setOverrideCursor(Qt.WaitCursor)
        threading.Thread(target=self._worker, args=(params,), daemon=True).start()
        QtCore.QTimer.singleShot(80, self._poll)

    def _worker(self, p):
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buf):
                warnings.simplefilter('always')
                names = [k for k in DUST_KEYS if p[k] > 0]
                opac = OpacityTable(dnames=names, abun=[p[k] for k in names])
                a, sed = run_disk(p['nr'], p['incl'], p['tin'], p['ms'], p['ls'], p['ts'],
                                  p['rout'], p['mdisk'], p['plsig'], opac, p['chi'])
            msgs = [str(w.message) for w in caught]
            self.results.put(('ok', a, sed, p, buf.getvalue(), msgs, time.perf_counter() - t0))
        except Exception:
            self.results.put(('error', traceback.format_exc(), buf.getvalue()))

    def _poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            QtCore.QTimer.singleShot(80, self._poll)
            return
        self.busy = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText('Run model')
        self.app.restoreOverrideCursor()
        if result[0] == 'error':
            _, tb, log = result
            self.last_log, self.last_msgs = log, [tb.strip()]
            self._write_log(log, self.last_msgs)
            self.set_status(f'The model failed: {tb.strip().splitlines()[-1]}', 'error')
            if not self.showing_log:
                self.toggle_log()
            return
        _, self.a, self.sed, params, log, msgs, dt = result
        self.last_run = params
        self.last_log, self.last_msgs = log, msgs
        self.red = 1 / (4 * np.pi * (params['dist'] * pc) ** 2)
        self.rtau = None
        if sys.stdout:
            sys.stdout.write(log)
        self._write_log(log, msgs)
        self._fill_summary(log)
        self.redraw_sed()
        self.redraw_structure()
        note = f'  {len(msgs)} solver warning{"s" if len(msgs) != 1 else ""}, see the log.' if msgs else ''
        self.set_status(f'Computed in {dt:.1f} s.{note}', 'stale' if msgs else 'ok')

    # ------------------------------------------------------------ SED plot
    def _range_edited(self):
        self.ymin, self.ymax = self.ymin_ed.text(), self.ymax_ed.text()
        self.redraw_sed()

    def _ylimits(self):
        try:
            lo, hi = float(self.ymin), float(self.ymax)
            if not lo < hi:
                raise ValueError
            ok = True
        except ValueError:
            ok = False
        set_error(self.ymin_ed, not ok)
        set_error(self.ymax_ed, not ok)
        if not ok:
            self.set_status('The y range needs two numbers, lower one first.', 'error')
            return None
        return lo, hi

    def redraw_sed(self):
        if self.sed is None:
            return
        sed, red = self.sed, self.red
        wav = sed['wav'] * 1e4
        total = sed['lnu'] * sed['nu'] * red
        ax = self.ax_sed
        ax.clear()
        ax.set_axis_on()
        components = self.flux_mode == 'components'
        if components:
            for key, lab, col in (('lnu_star', 'star', C_STAR), ('lnu_mid', 'midplane', C_MID),
                                  ('lnu_wall', 'wall', C_WALL), ('lnu_surf', 'surface', C_SURF)):
                ax.plot(wav, sed[key] * sed['nu'] * red, color=col, label=lab, lw=1.4 * self.scale)
            ax.plot(wav, total, color=INK, label='total', lw=2 * self.scale)
        else:
            ax.plot(wav, total, color=INK, lw=2 * self.scale)

        show_obs = self.obs is not None and self.show_obs
        if show_obs:
            self._plot_obs(ax)
        if components or show_obs:
            ax.legend(loc='best')

        ax.set_xscale('log')
        ax.set_yscale(self.yscale)
        if self.fit_all:
            ax.set_ylim(bottom=min(total), top=max(total))
            ax.set_xlim(left=min(wav), right=max(wav))
        else:
            lim = self._ylimits()
            if lim:
                ax.set_ylim(bottom=lim[0], top=lim[1])
            ax.set_xlim(left=0.1, right=1000)
        ax.set_xlabel(WAV_XLABEL)
        ax.set_ylabel(SED_YLABEL)
        self.cv_sed.draw_idle()

    def _plot_obs(self, ax):
        nu, flux, flux_err, style = self.obs
        wav = cc / nu * 1e4
        kw = dict(color=C_OBS)
        labelled = [False]

        def lab():
            if labelled[0]:
                return None
            labelled[0] = True
            return self.obs_name

        if self.obs_style == 'line':
            m = style == 0
            if m.any():
                ax.plot(wav[m], flux[m] * nu[m], label=lab(), **kw)
            m = (style != 0) & (flux_err > 0)
            if m.any():
                ax.errorbar(wav[m], flux[m] * nu[m], yerr=flux_err[m] * nu[m], label=lab(), **kw)
            m = (style != 0) & (flux_err <= 0)
            if m.any():
                ax.scatter(wav[m], flux[m] * nu[m], marker='^', label=lab(), zorder=3,
                           s=(5 * self.scale) ** 2, **kw)
        else:
            m = flux_err > 0
            if m.any():
                ax.errorbar(wav[m], flux[m] * nu[m], yerr=flux_err[m] * nu[m], label=lab(), **kw)
            m = flux_err <= 0
            if m.any():
                ax.scatter(wav[m], flux[m] * nu[m], marker='^', label=lab(), zorder=3,
                           s=(5 * self.scale) ** 2, **kw)

    def reset_range(self):
        self.ymin, self.ymax, self.fit_all = '1e-11', '1e-7', False
        self.ymin_ed.setText(self.ymin)
        self.ymax_ed.setText(self.ymax)
        self.chk_fit.setChecked(False)
        self.redraw_sed()

    # ------------------------------------------------------------ structure plot
    def redraw_structure(self):
        if self.a is None:
            return
        a, sed, ax = self.a, self.sed, self.ax_st
        kind = self.struct
        ax.clear()
        ax.set_axis_on()
        r = a.r / AU
        if kind == 'hr':
            ax.plot(r, a.Hs / a.r, label='Hs', color=C_MID, linestyle='dashed')
            ax.plot(r, a.Hp / a.r, label='Hp', color=INK)
            ax.set_ylabel(r'$H/R$')
            ax.legend()
        elif kind == 'T':
            ax.plot(r, a.Ti, label='Ti', color=INK)
            ax.plot(r, a.Ts, label='Ts', color=C_WALL, linestyle='dashed')
            ax.set_ylabel(r'$T$ (K)')
            ax.legend()
        elif kind == 'sigma':
            ax.plot(r, a.Sigma, color=INK)
            ax.set_ylabel(r'Dust surface density')
        elif kind == 'tau':
            if self.rtau is None:
                self.rtau = findtauradii(a.opac, a.Sigma, a.r, a.Ti)
            ax.plot(sed['wav'] * 1e4, self.rtau / AU, color=INK)
            ax.set_xlim(left=0.1, right=10000)
            ax.set_ylim(bottom=0.1, top=1000)
            ax.set_ylabel(r'Radius for $\tau = 1$ (AU)')
        elif kind == 'opac':
            kappa = findkappa(a.opac, 0)
            ax.plot(sed['wav'] * 1e4, kappa, color=INK)
            ax.set_xlim(left=0.1, right=1000)
            ax.set_ylabel(r'Opacity')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(WAV_XLABEL if kind in ('tau', 'opac') else r'$R$ (AU)')
        self.cv_st.draw_idle()

    # ------------------------------------------------------------ summary & log
    SUMMARY_NAMES = {
        'Mass of the disk (dust)': 'Dust mass',
        'Mass of the disk (dust+gas)': 'Gas+dust mass',
        'Radius of the inner rim': 'Inner rim',
        'Covering fraction by the wall': 'Rim covering',
        'Inner wall Hs/Hp': 'Rim Hs/Hp',
        'Covering fraction by CG disk': 'Disk covering',
        'Disk optthin to own rad at R': 'Thin (own rad.)',
        'Disk optthin to surf rad at R': 'Thin (surf. rad.)',
        'Energy conservation error': 'Energy error',
        'Full luminosity': 'Luminosity',
        'L_wall/L_tot': 'L wall / L tot',
        'L_mid/L_tot': 'L mid / L tot',
        'L_surf/L_tot': 'L surf / L tot',
        'L_scat/L_tot': 'L scat / L tot',
        'L_accr/L_tot': 'L accr / L tot',
    }

    @classmethod
    def summary_rows(cls, log):
        rows = []
        for line in log.splitlines():
            shadow = re.match(r'\s*(Full|Half) shadow at ir=\d+, R=(\S+) AU', line)
            if shadow:
                rows.append((f'{shadow.group(1)} shadow at', f'{float(shadow.group(2)):.3g} AU'))
                continue
            if ' = ' not in line:
                continue
            key, val = line.split(' = ', 1)
            key = ' '.join(key.split())
            parts = val.split(' ', 1)
            try:
                number = float(parts[0])
            except ValueError:
                rows.append((cls.SUMMARY_NAMES.get(key, key), val))
                continue
            if key in ('L_scat/L_tot', 'L_accr/L_tot') and number == 0:
                continue   # scattering and accretion are off; skip the zero rows
            unit = parts[1].strip() if len(parts) > 1 else ''
            unit = unit.replace('*4pi', 'x 4pi')
            if key == 'Full luminosity':
                unit = 'Lsun'
            rows.append((cls.SUMMARY_NAMES.get(key, key), f'{number:.3g}' + (' ' + unit if unit else '')))
        return rows

    def _fill_summary(self, log):
        while self.sum_grid.count():
            item = self.sum_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        rows = self.summary_rows(log)
        half = (len(rows) + 1) // 2
        for i, (k, v) in enumerate(rows):
            col = 0 if i < half else 3
            self.sum_grid.addWidget(label(k, 'key'), i % half, col)
            val = label(v, 'value')
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.sum_grid.addWidget(val, i % half, col + 1)
        if rows:
            self.sum_grid.setColumnMinimumWidth(2, px(16))
            self.sum_grid.setColumnStretch(0, 1)
            self.sum_grid.setColumnStretch(3, 1)
            self.sum_grid.setRowStretch(half, 1)

    def _write_log(self, log, msgs):
        self.log_text.clear()
        self.log_text.appendPlainText(log.rstrip('\n'))
        for m in msgs:
            self.log_text.appendHtml(f'<span style="color:{ERROR}">'
                                     + m.replace('&', '&amp;').replace('<', '&lt;').replace('\n', '<br>')
                                     + '</span>')

    def toggle_log(self):
        self.showing_log = not self.showing_log
        self.stack.setCurrentIndex(1 if self.showing_log else 0)
        self.log_btn.setText('Summary' if self.showing_log else 'Solver log')

    # ------------------------------------------------------------ files
    @staticmethod
    def _remember_folder(path):
        global LAUNCH_DIR
        LAUNCH_DIR = os.path.dirname(os.path.abspath(path))

    @staticmethod
    def _path(result):
        return result[0] if isinstance(result, tuple) else result

    def save_model(self):
        start = os.path.join(LAUNCH_DIR, self.model_name or '')
        path = self._path(QtWidgets.QFileDialog.getSaveFileName(self, 'Save model', start,
                                                                'CG++ model (*.model);;All files (*)'))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += '.model'
        current = dict(self.values, ymin=self.ymin, ymax=self.ymax)
        try:
            with open(path, 'w') as file:
                for key, prefix in MODEL_FILE_LINES:
                    print(f'{prefix}{current[key]}', file=file)
        except OSError as err:
            self.set_status(f'Could not save the model: {err.strerror}.', 'error')
            return
        self._remember_folder(path)
        self.model_name = os.path.basename(path)
        self.name_lbl.setText(self.model_name)
        self.set_status(f'Saved {self.model_name}.', 'ok')

    def open_model(self):
        path = self._path(QtWidgets.QFileDialog.getOpenFileName(self, 'Open model', LAUNCH_DIR,
                                                                'CG++ model (*.model);;All files (*)'))
        if not path:
            return
        out = {}
        try:
            with open(path, 'r') as file:
                for line in file.readlines():
                    if ':' not in line:
                        continue
                    key, value = [entry.strip() for entry in line.split(':', 1)]
                    out[key] = value
            new = {}
            for key, lab in MODEL_FILE_LINES:
                name = lab.split(':')[0]
                if name not in out:
                    continue
                num = float(out[name])
                if key == 'nr':
                    new[key] = f'{int(num)}'
                elif key in ('ymin', 'ymax'):
                    new[key] = f'{num:.2e}'
                else:
                    new[key] = f'{num}'
        except (OSError, ValueError) as err:
            self.set_status(f'Could not read {os.path.basename(path)}: {err}', 'error')
            return
        for key, val in new.items():
            if key == 'ymin':
                self.ymin = val
                self.ymin_ed.setText(val)
            elif key == 'ymax':
                self.ymax = val
                self.ymax_ed.setText(val)
            else:
                self.entries[key].setText(val)
        self._remember_folder(path)
        self.model_name = os.path.basename(path)
        self.name_lbl.setText(self.model_name)
        self.run_model()

    def open_obs(self):
        path = self._path(QtWidgets.QFileDialog.getOpenFileName(self, 'Load observations', LAUNCH_DIR,
                                                                'Observations (*.dat);;All files (*)'))
        if not path:
            return
        try:
            data = np.loadtxt(path, skiprows=3)
            self.obs = (data[:, 0], data[:, 1], data[:, 2], data[:, 3])
        except (OSError, ValueError, IndexError):
            self.set_status(f'Could not read {os.path.basename(path)}: expected 4 columns '
                            f'(nu, flux, error, style) after 3 header lines.', 'error')
            return
        self._remember_folder(path)
        self.obs_name = os.path.basename(path)
        self.show_obs = True
        self.chk_obs.setChecked(True)
        self._obs_ui()
        self.redraw_sed()

    def _obs_ui(self):
        loaded = self.obs is not None
        self.obs_lbl.setText(self.obs_name if loaded else 'No observations loaded')
        self.obs_lbl.setObjectName('' if loaded else 'muted')
        self.obs_lbl.style().unpolish(self.obs_lbl)
        self.obs_lbl.style().polish(self.obs_lbl)
        for w in self.obs_controls:
            w.setVisible(loaded)

    def clear_obs(self):
        self.obs = self.obs_name = None
        self._obs_ui()
        self.redraw_sed()

    def save_sed(self):
        if self.sed is None:
            self.set_status('Run the model before saving its SED.', 'error')
            return
        path = self._path(QtWidgets.QFileDialog.getSaveFileName(self, 'Save SED',
                                                                os.path.join(LAUNCH_DIR, 'sed.out'),
                                                                'All files (*)'))
        if not path:
            return
        sed, red = self.sed, self.red
        try:
            with open(path, 'w') as file:
                print('#wavelength     total          star          wall       midplane       surface', file=file)
                for inu in range(len(sed['wav'])-1, -1, -1):
                    print(f"{sed['wav'][inu]*1e4:.4e}   {sed['lnu'][inu]*sed['nu'][inu]*red:.5e}   {sed['lnu_star'][inu]*sed['nu'][inu]*red:.5e}   "
                          f"{sed['lnu_wall'][inu]*sed['nu'][inu]*red:.5e}   {sed['lnu_mid'][inu]*sed['nu'][inu]*red:.5e}   {sed['lnu_surf'][inu]*sed['nu'][inu]*red:.5e}", file=file)
        except OSError as err:
            self.set_status(f'Could not save the SED: {err.strerror}.', 'error')
            return
        self.set_status(f'Saved SED to {os.path.basename(path)}.', 'ok')

    # ------------------------------------------------------------ dialogs
    def _dialog(self, title):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(px(26), px(22), px(26), px(20))
        lay.setSpacing(px(4))
        return dlg, lay

    def show_settings(self):
        dlg, lay = self._dialog('Settings')
        lay.addWidget(label('Settings', 'title'))
        lay.addSpacing(px(12))
        lay.addWidget(label('<b>Interface size</b>'))
        seg = Segmented([(s, format_scale(s)) for s in SCALES],
                        min(SCALES, key=lambda s: abs(s - self.scale)))
        lay.addWidget(seg)
        lay.addWidget(label(f'Suggested for this screen: {format_scale(self.auto)}. '
                            f'Ctrl + and Ctrl - also change it.', 'key'))
        lay.addSpacing(px(14))
        lay.addWidget(label('<b>Font</b>'))
        combo = QtWidgets.QFontComboBox()
        combo.setCurrentFont(QtGui.QFont(self.family()))
        lay.addWidget(combo, 0, Qt.AlignLeft)
        preview = label('Disk mass 0.05 Msun, T_in 1500 K, inclination 60 deg', 'preview')
        lay.addSpacing(px(6))
        lay.addWidget(preview)

        def show_preview(font=None):
            preview.setStyleSheet(f'font-family: "{combo.currentFont().family()}";')
        combo.currentFontChanged.connect(show_preview)
        show_preview()
        lay.addWidget(label('Plots use the same font when matplotlib can find it.', 'key'))
        reset = button('Use the system font', lambda: combo.setCurrentFont(QtGui.QFont(self.app.font().family())),
                       'ghost')
        lay.addSpacing(px(16))

        def apply():
            fam = combo.currentFont().family()
            self.font_choice = '' if fam == self.app.font().family() else fam
            new_scale = seg.value()
            self.prefs.update(scale=new_scale, font=self.font_choice)
            save_prefs(self.prefs)
            dlg.accept()
            self.apply_scale(new_scale)
            self.set_status('Settings applied.', 'ok')
        lay.addWidget(hbox(reset, None, button('Cancel', dlg.reject, 'ghost'), button('Apply', apply, 'apply'),
                           spacing=6))
        qexec(dlg)

    def show_help(self):
        dlg, lay = self._dialog('About CG++')
        brand = label('CG++', 'title')
        brand.setStyleSheet(f'font-size: {px(29)}px;')
        lay.addWidget(brand)
        lay.addWidget(label('A Python port of CG+ (Dullemond, Dominik & Natta 2001)', 'muted'))
        lay.addSpacing(px(8))
        notes = [
            ('Parameters', 'Hover the small i next to a label for what it means. Rim Hs/Hp is computed by '
                           'the model when set to 0 or below. Albedo has no effect, as scattering is turned off.'),
            ('Running', 'Press Enter in any field, or Ctrl+R. The window stays responsive while the model '
                        'computes; solver messages appear under Solver log.'),
            ('Observations', 'Load a .dat file with 3 header lines and columns nu, flux, error, style. '
                             'Use Show to toggle them, Remove to unload.'),
            ('Files', 'Models are saved in the same .model format as before. Save SED writes the total and '
                      'component fluxes as columns.'),
            ('Layout', 'Drag the gaps between the panels to resize them.'),
            ('Size and font', 'Change them under Settings, or use Ctrl + and Ctrl - for the size. '
                              f'Your choice is remembered in {PREFS_PATH}.'),
            ('License', 'Free software under the GNU General Public License, version 3 or later. '
                        'You may share and modify it under the same terms; it comes with no warranty.'),
            ('Contact', 'lis.zwicky@csfk.org'),
        ]
        for title, text in notes:
            lay.addSpacing(px(6))
            lay.addWidget(label(f'<b>{title}</b>'))
            body = label(text, 'muted')
            body.setWordWrap(True)
            body.setFixedWidth(px(480))
            lay.addWidget(body)
        lay.addSpacing(px(14))
        close = button('Close', dlg.accept, 'apply')
        lay.addWidget(hbox(None, close))
        close.setDefault(True)
        qexec(dlg)


def find_asset(name):
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        path = os.path.join(base, 'assets', name)
        if os.path.isfile(path):
            return path
    return None


def main():
    scale = None
    if '--scale' in sys.argv:
        try:
            scale = float(sys.argv[sys.argv.index('--scale') + 1])
        except (IndexError, ValueError):
            print('Usage: cgplusplus [--scale 1.5] [--selftest]')
    if not QT6:
        # Qt 5 needs these before the application exists to honour the OS display scaling
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('CG++')
    icon = find_asset('cgplusplus.png')
    if icon:
        app.setWindowIcon(QtGui.QIcon(icon))
    win = CGWindow(app, scale)
    win.show()
    sys.exit(qexec(app))


if __name__ == '__main__':
    main()
