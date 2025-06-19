import numpy as np
import os
import tkinter as tk
from tkinter import ttk
# import tkinter.font as tkFont
from dataclasses import dataclass
from typing import List
from scipy.special import erf
import warnings
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
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

window = tk.Tk()
# print(tkFont.names())
# custom_font = tkFont.Font(name='TkDefaultFont', size=12)
window.geometry(f'1450x800')
fig = plt.figure(1)
fig.set_tight_layout(True)
a, sed = run_disk(30, 60, 1500, 2.4, 47, 9520, 400, 0.05, -2, OpacityTable(dnames=['silicate']), 0)
dpc = 144
red = 1 / (4 * np.pi * (dpc * pc) ** 2)
plt.plot(sed['wav']*1e4, sed['lnu']*sed['nu']*red, color='black')
plt.xlim(left=0.1, right=1000)
plt.ylim(top=1e-7, bottom=1e-11)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ ($\mu$m)')
plt.ylabel(r'$\nu F_\nu$ (erg/s/cm$^2$)')

fig2 = plt.figure(2)
fig2.set_tight_layout(True)
plt.ion()
plt.plot(a.r/AU, a.Hs/a.r, label='Hs', color='black', linestyle='dashed')
plt.plot(a.r/AU, a.Hp/a.r, label='Hp', color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R$ (AU)')
plt.ylabel(r'$H/R$')
plt.legend()

side_frame = tk.Frame(window, width=160, height=800, padx=3)
side_frame.grid(column=0)
plot_frame = tk.Frame(window, width=800, height=500, padx=10, highlightbackground='gray', highlightthickness=2)
plot_frame.grid(column=1, row=0)
buttons_frame = tk.Frame(window, width=800, height=100, padx=3, pady=3, highlightbackground='gray', highlightthickness=2)
buttons_frame.grid(column=1, row=1)
newplot_frame = tk.Frame(window, width=800, height=100, padx=3, pady=3)
newplot_frame.grid(column=1, row=2)
file_frame = tk.Frame(window, width=800, height=100, padx=3, pady=3)
file_frame.grid(column=1, row=3)
input_frame = tk.Frame(window, width=800, height=100, padx=3, pady=3, highlightbackground='gray', highlightthickness=2)
input_frame.grid(column=1, row=4)
dust_frame = tk.Frame(window, width=800, height=100, padx=3, pady=3, highlightbackground='gray', highlightthickness=2)
dust_frame.grid(column=1, row=5)

plot2_frame = tk.Frame(window, width=600, height=300, padx=10, highlightbackground='gray', highlightthickness=2)
plot2_frame.grid(column=2, row=0)
plot2_button_frame = tk.Frame(window, width=600, height=100, padx=10, highlightbackground='gray', highlightthickness=2)
plot2_button_frame.grid(column=2, row=1)

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
plot_widget = canvas.get_tk_widget()
toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.update()

canvas2 = FigureCanvasTkAgg(fig2, master=plot2_frame)
plot2_widget = canvas2.get_tk_widget()
toolbar2 = NavigationToolbar2Tk(canvas2, plot2_frame)
toolbar2.update()
#
# canvas.get_tk_widget()

def update():
    global a, sed
    nr = int(nr_var.get())
    incl = float(incl_var.get())
    mstar = float(ms_var.get())
    lstar = float(ls_var.get())
    tstar = float(ts_var.get())
    rout = float(rout_var.get())
    mdisk = float(mdisk_var.get())
    plsig = float(plsig_var.get())
    tin = float(tin_var.get())
    chiwall = float(chi_var.get())
    dust_names = []
    dust_abun = []
    if float(silicate_var.get())>0:
        dust_names.append('silicate')
        dust_abun.append(float(silicate_var.get()))
    if float(forsterite_var.get())>0:
        dust_names.append('forsterite')
        dust_abun.append(float(forsterite_var.get()))
    if float(silbeta1_var.get())>0:
        dust_names.append('silbeta1')
        dust_abun.append(float(silbeta1_var.get()))
    if float(carbon_var.get())>0:
        dust_names.append('carbon')
        dust_abun.append(float(carbon_var.get()))
    if float(ice_var.get())>0:
        dust_names.append('ice')
        dust_abun.append(float(ice_var.get()))
    opac = OpacityTable(dnames=dust_names, abun=dust_abun)

    a, sed = run_disk(nr, incl, tin, mstar, lstar, tstar, rout, mdisk, plsig, opac, chiwall)
    dpc = float(dist_var.get())
    red = 1 / (4 * np.pi * (dpc * pc) ** 2)
    plt.figure(1)
    fig.clear()
    plt.plot(sed['wav']*1e4, sed['lnu']*sed['nu']*red, color='black')
    if not is_totflux.get():
        plt.plot(sed['wav'] * 1e4, sed['lnu'] * sed['nu'] * red, color='black')
        plt.plot(sed['wav'] * 1e4, sed['lnu_star'] * sed['nu'] * red, label='star')
        plt.plot(sed['wav'] * 1e4, sed['lnu_mid'] * sed['nu'] * red, label='midplane')
        plt.plot(sed['wav'] * 1e4, sed['lnu_wall'] * sed['nu'] * red, label='wall')
        plt.plot(sed['wav'] * 1e4, sed['lnu_surf'] * sed['nu'] * red, label='surface')
        plt.legend()

    if show_allpoints.get():
        plt.ylim(bottom=min(sed['lnu'] * sed['nu'] * red), top=max(sed['lnu'] * sed['nu'] * red))
        plt.xlim(left=min(sed['wav'] * 1e4), right=max(sed['wav'] * 1e4))
    else:
        plt.ylim(bottom=float(ymin_var.get()), top=float(ymax_var.get()))
        plt.xlim(left=0.1, right=1000)
    plt.xscale('log')
    plt.yscale(yscale_var.get())

    plt.xlabel(r'$\lambda$ ($\mu$m)')
    plt.ylabel(r'$\nu F_\nu$ (erg/s/cm$^2$)')

    change_plottype()

def change_yscale():
    plt.figure(1)
    plt.yscale(yscale_var.get())

def change_plotstyle():
    plt.figure(1)
    if is_totflux.get():
        fig.clear()
        plt.plot(sed['wav'] * 1e4, sed['lnu'] * sed['nu'] * red, color='black')
    else:
        fig.clear()
        plt.plot(sed['wav'] * 1e4, sed['lnu'] * sed['nu'] * red, color='black')
        plt.plot(sed['wav']*1e4, sed['lnu_star']*sed['nu']*red, label='star')
        plt.plot(sed['wav']*1e4, sed['lnu_mid']*sed['nu']*red, label='midplane')
        plt.plot(sed['wav']*1e4, sed['lnu_wall']*sed['nu']*red, label='wall')
        plt.plot(sed['wav']*1e4, sed['lnu_surf']*sed['nu']*red, label='surface')
        plt.legend()

    plt.yscale(yscale_var.get())
    plt.xscale('log')
    if show_allpoints.get():
        plt.ylim(bottom=min(sed['lnu'] * sed['nu'] * red), top=max(sed['lnu'] * sed['nu'] * red))
        plt.xlim(left=min(sed['wav'] * 1e4), right=max(sed['wav'] * 1e4))
    else:
        plt.ylim(bottom=float(ymin_var.get()), top=float(ymax_var.get()))
        plt.xlim(left=0.1, right=1000)
    plt.xlabel(r'$\lambda$ ($\mu$m)')
    plt.ylabel(r'$\nu F_\nu$ (erg/s/cm$^2$)')

def showallpoints():
    plt.figure(1)
    if show_allpoints.get():
        plt.ylim(bottom=min(sed['lnu'] * sed['nu'] * red), top=max(sed['lnu'] * sed['nu'] * red))
        plt.xlim(left=min(sed['wav']*1e4), right=max(sed['wav']*1e4))
    else:
        plt.ylim(bottom=float(ymin_var.get()), top=float(ymax_var.get()))
        plt.xlim(left=0.1, right=1000)

def reset_button():
    ymin_var.set(value=1e-11)
    ymax_var.set(value=1e-7)
    showallpoints()

def change_plottype():
    plt.figure(2)
    if plot_type.get() == 'hr':
        fig2.clear()
        plt.plot(a.r/AU, a.Hs / a.r, label='Hs', color='black', linestyle='dashed')
        plt.plot(a.r/AU, a.Hp / a.r, label='Hp', color='black')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$R$ (AU)')
        plt.ylabel(r'$H/R$')
        plt.legend()
    elif plot_type.get() == 'T':
        fig2.clear()
        plt.plot(a.r/AU, a.Ti, label='Ti', color='black')
        plt.plot(a.r/AU, a.Ts, label='Ts', color='black', linestyle='dashed')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$R$ (AU)')
        plt.ylabel(r'$T$ (K)')
        plt.legend()
    elif plot_type.get() == 'sigma':
        fig2.clear()
        plt.plot(a.r/AU, a.Sigma, color='black')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$R$ (AU)')
        plt.ylabel(r'Dust Surface Density')
    elif plot_type.get() == 'tau':
        rtau = findtauradii(a.opac, a.Sigma, a.r, a.Ti)
        fig2.clear()
        plt.plot(sed['wav']*1e4, rtau/AU, color='black')
        plt.xscale('log')
        plt.xlim(left=0.1, right=10000)
        plt.yscale('log')
        plt.ylim(bottom=0.1, top=1000)
        plt.xlabel(r'$\lambda$ ($\mu$m)')
        plt.ylabel(r'Radius for $\tau = 1$ (AU)')

def save_model():
    save_frame = tk.Frame(window, width=600, height=200, padx=10, highlightbackground='gray',
                                  highlightthickness=2)
    save_frame.grid(column=2, row=2)
    tk.Label(save_frame, text=f'Save current settings in ').grid(row=0, column=0)
    model_name = tk.StringVar(value='.model')
    tk.Entry(save_frame, textvariable=model_name, width=20).grid(row=0, column=1)

    def save_button():
        with open(model_name.get(), 'w') as file:
            print(f'Nr:      {nr_var.get()}', file=file)
            print(f'Distance[pc]:       {dist_var.get()}', file=file)
            print(f'Inclination[degrees]:       {incl_var.get()}', file=file)
            print(f'Starmass[MS]:       {ms_var.get()}', file=file)
            print(f'Starluminosity[LS]:       {ls_var.get()}', file=file)
            print(f'Startemperature[K]:       {ts_var.get()}', file=file)
            print(f'Outer_disk-radius[AU]:       {rout_var.get()}', file=file)
            print(f'Mdisk:    {mdisk_var.get()}', file=file)
            print(f'Plsigma:      {plsig_var.get()}', file=file)
            print(f'silicate:       {silicate_var.get()}', file=file)
            print(f'carbon:       {carbon_var.get()}', file=file)
            print(f'forsterite:       {forsterite_var.get()}', file=file)
            print(f'ice:       {ice_var.get()}', file=file)
            print(f'silbeta1:       {silbeta1_var.get()}', file=file)
            print(f'Inner Disk temp.:       {tin_var.get()}', file=file)
            print(f'Chi:       {chi_var.get()}', file=file)
            print(f'Albedo:       {albedo_var.get()}', file=file)
            print(f'Yrange[0]:   {ymin_var.get()}', file=file)
            print(f'Yrange[1]:   {ymax_var.get()}', file=file)

    def cancel_button():
        save_frame.grid_forget()
        save_frame.destroy()

    tk.Button(save_frame, text="Save", command=save_button).grid(row=1, column=0)
    tk.Button(save_frame, text="Cancel", command=cancel_button).grid(row=1, column=1)

def load_model():
    out = {}
    with open(model_var.get(), 'r') as file:
        for line in file.readlines():
            key, value = [entry.strip() for entry in line.split(':')]
            out[key] = value

    nr_var.set(f"{(int(float(out['Nr'])))}")
    dist_var.set(f"{float(out['Distance[pc]'])}")
    incl_var.set(f"{float(out['Inclination[degrees]'])}")
    ms_var.set(f"{float(out['Starmass[MS]'])}")
    ls_var.set(f"{float(out['Starluminosity[LS]'])}")
    ts_var.set(f"{float(out['Startemperature[K]'])}")
    rout_var.set(f"{float(out['Outer_disk-radius[AU]'])}")
    mdisk_var.set(f"{float(out['Mdisk'])}")
    plsig_var.set(f"{float(out['Plsigma'])}")
    silicate_var.set(f"{float(out['silicate'])}")
    carbon_var.set(f"{float(out['carbon'])}")
    forsterite_var.set(f"{float(out['forsterite'])}")
    ice_var.set(f"{float(out['ice'])}")
    silbeta1_var.set(f"{float(out['silbeta1'])}")
    tin_var.set(f"{float(out['Inner Disk temp.'])}")
    chi_var.set(f"{float(out['Chi'])}")
    albedo_var.set(f"{float(out['Albedo'])}")
    ymin_var.set(f"{float(out['Yrange[0]']):.2e}")
    ymax_var.set(f"{float(out['Yrange[1]']):.2e}")

    update()

def load_obs(event):
    data = np.loadtxt(obs_var.get(), skiprows=3)
    nu = data[:, 0]
    flux = data[:, 1]
    flux_err = data[:, 2]
    style = data[:, 3]
    wav = cc/nu * 1e4

    plt.figure(1)
    change_plotstyle()
    if is_line.get():
        line_mask = style == 0
        plt.plot(wav[line_mask], flux[line_mask]*nu[line_mask], color='gray')

        error_mask = (style != 0) & (flux_err > 0)
        plt.errorbar(wav[error_mask], flux[error_mask]*nu[error_mask], yerr=flux_err[error_mask]*nu[error_mask], color='gray')

        point_mask = (style != 0) & (flux_err <= 0)
        plt.scatter(wav[point_mask], flux[point_mask]*nu[point_mask], color='gray', marker='^')
    else:
        error_mask = (flux_err > 0)
        plt.errorbar(wav[error_mask], flux[error_mask] * nu[error_mask], yerr=flux_err[error_mask] * nu[error_mask],
                     color='gray')

        point_mask = (flux_err <= 0)
        plt.scatter(wav[point_mask], flux[point_mask] * nu[point_mask], color='gray', marker='^')

def load_obs2():
    if obs_var.get() == '':
        return
    data = np.loadtxt(obs_var.get(), skiprows=3)
    nu = data[:, 0]
    flux = data[:, 1]
    flux_err = data[:, 2]
    style = data[:, 3]
    wav = cc / nu * 1e4

    plt.figure(1)
    change_plotstyle()
    if is_line.get():
        line_mask = style == 0
        plt.plot(wav[line_mask], flux[line_mask] * nu[line_mask], color='gray')

        error_mask = (style != 0) & (flux_err > 0)
        plt.errorbar(wav[error_mask], flux[error_mask] * nu[error_mask], yerr=flux_err[error_mask] * nu[error_mask],
                     color='gray')

        point_mask = (style != 0) & (flux_err <= 0)
        plt.scatter(wav[point_mask], flux[point_mask] * nu[point_mask], color='gray', marker='^')
    else:
        error_mask = (flux_err > 0)
        plt.errorbar(wav[error_mask], flux[error_mask] * nu[error_mask], yerr=flux_err[error_mask] * nu[error_mask],
                     color='gray')

        point_mask = (flux_err <= 0)
        plt.scatter(wav[point_mask], flux[point_mask] * nu[point_mask], color='gray', marker='^')

def save_sed():
    dpc = float(dist_var.get())
    red = 1 / (4 * np.pi * (dpc * pc) ** 2)
    with open('sed.out', 'w') as file:
        print('#wavelength     total          star          wall       midplane       surface', file=file)
        for inu in range(len(sed['wav'])-1, -1, -1):
            print(f"{sed['wav'][inu]*1e4:.4e}   {sed['lnu'][inu]*sed['nu'][inu]*red:.5e}   {sed['lnu_star'][inu]*sed['nu'][inu]*red:.5e}   "
                  f"{sed['lnu_wall'][inu]*sed['nu'][inu]*red:.5e}   {sed['lnu_mid'][inu]*sed['nu'][inu]*red:.5e}   {sed['lnu_surf'][inu]*sed['nu'][inu]*red:.5e}", file=file)

def help_button():
    popup = tk.Toplevel()
    popup.wm_title('Help')

    textmessage = 'Welcome to CG++, here are some notes for this interface\n\n' \
                  'Variables that can be misunderstood:\n\n' \
                  'N_r --- number of radial cells in a grid\n\n' \
                  'disk size --- outer radius of the disk\n\n' \
                  'plsig --- power law exponent for surface density, Sigma(R) = Sigma0*(R/Rc)**plsigma\n\n' \
                  'chi* --- Hs/Hp (scale heights) ratio (only for the inner rim!!)\n\n' \
                  'albedo* does not do anything as scattering is turned off\n\n' \
                  '* means if the value of parameter is <=0 then it will be calculated by the script\n\n' \
                  'To remove observations from the plot click either Total flux or Components\n\n' \
                  'To put them back click either Line or Symbol\n\n' \
                  'My email if something is really bad: lis.zwicky@csfk.org'

    tk.Label(popup, text=textmessage).grid(row=0, column=0)

    b = ttk.Button(popup, text="Okay", command=popup.destroy)
    b.grid(row=1, column=0)

def showopac():

    fig3 = plt.figure(3)
    if show_opac.get():
        fig3.set_tight_layout(True)
        plt.ion()
        kappa = findkappa(a.opac, 0)
        plt.plot(sed['wav'] * 1e4, kappa, color='black')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(left=0.1, right=1000)
        plt.xlabel(r'$\lambda$ ($\mu$m)')
        plt.ylabel(r'Opacity')
    else:
        plt.close(fig=fig3)


plot_widget.pack()
plot2_widget.pack()
tk.Button(side_frame,text="Quit",command=exit).grid(row=0, column=0, padx=5)
tk.Button(side_frame,text="Help",command=help_button).grid(row=1, column=0, padx=5)

yscale_var = tk.StringVar(value='log')
tk.Radiobutton(buttons_frame, text='Y lin', variable=yscale_var, value='linear', command=change_yscale).grid(row=0, column=0)
tk.Radiobutton(buttons_frame, text='Y log', variable=yscale_var, value='log', command=change_yscale).grid(row=0, column=1)

is_line = tk.BooleanVar(value=True)
tk.Radiobutton(buttons_frame, text='Line', variable=is_line, value=True, command=load_obs2).grid(row=0, column=2)
tk.Radiobutton(buttons_frame, text='Symbol', variable=is_line, value=False, command=load_obs2).grid(row=0, column=3)

is_totflux = tk.BooleanVar(value=True)
tk.Radiobutton(buttons_frame, text='Total flux', variable=is_totflux, value=True, command=change_plotstyle).grid(row=0, column=4)
tk.Radiobutton(buttons_frame, text='Components', variable=is_totflux, value=False, command=change_plotstyle).grid(row=0, column=5)

show_opac = tk.BooleanVar(value=False)
show_allpoints = tk.BooleanVar(value=False)
tk.Checkbutton(buttons_frame, text='Show opacity', variable=show_opac, onvalue=True, offvalue=False,
               command=showopac).grid(row=1, column=0, columnspan=2)
tk.Checkbutton(buttons_frame, text='View all points', variable=show_allpoints, onvalue=True, offvalue=False,
               command=showallpoints).grid(row=1, column=2, columnspan=2)
tk.Label(buttons_frame, text='ymin-ymax').grid(row=1, column=4)
ymin_var = tk.StringVar(value='1e-11')
tk.Entry(buttons_frame, textvariable=ymin_var, width=10).grid(row=1, column=5)
ymax_var = tk.StringVar(value='1e-7')
tk.Entry(buttons_frame, textvariable=ymax_var, width=10).grid(row=1, column=6)

tk.Button(newplot_frame,text="New plot",command=update).grid(row=0, column=0)
tk.Button(newplot_frame,text="Reset",command=reset_button).grid(row=0, column=1)
tk.Button(newplot_frame,text="Save SED",command=save_sed).grid(row=0, column=2)

obs_var = tk.StringVar()
tk.Label(file_frame, text=f'obs file: ').grid(row=0, column=0)
obs_box = ttk.Combobox(file_frame,textvariable=obs_var, values=find_dat())
obs_box.grid(row=0, column=1)
obs_box.bind('<<ComboboxSelected>>', load_obs)
tk.Button(file_frame,text="Save model",command=save_model).grid(row=0, column=2)
model_var = tk.StringVar()
tk.Button(file_frame,text="Load model: ",command=load_model).grid(row=0, column=3)
model_box = ttk.Combobox(file_frame,textvariable=model_var, values=find_model())
model_box.grid(row=0, column=4)


tk.Label(input_frame, text='N_r').grid(row=0, column=0)
nr_var = tk.StringVar(value='30')
tk.Entry(input_frame, textvariable=nr_var, width=10).grid(row=0, column=1)

tk.Label(input_frame, text='star mass (Msun)').grid(row=0, column=2, padx=5)
ms_var = tk.StringVar(value='2.4')
tk.Entry(input_frame, textvariable=ms_var, width=10).grid(row=0, column=3)

tk.Label(input_frame, text='disk size (AU)').grid(row=0, column=4)
rout_var = tk.StringVar(value='400')
tk.Entry(input_frame, textvariable=rout_var, width=10).grid(row=0, column=5)

tk.Label(input_frame, text='distance (pc)').grid(row=1, column=0)
dist_var = tk.StringVar(value='144.0')
tk.Entry(input_frame, textvariable=dist_var, width=10).grid(row=1, column=1)

tk.Label(input_frame, text='star lum (Lsun)').grid(row=1, column=2, padx=5)
ls_var = tk.StringVar(value='47.0')
tk.Entry(input_frame, textvariable=ls_var, width=10).grid(row=1, column=3)

tk.Label(input_frame, text='disk mass (Msun)').grid(row=1, column=4)
mdisk_var = tk.StringVar(value='0.05')
tk.Entry(input_frame, textvariable=mdisk_var, width=10).grid(row=1, column=5)

tk.Label(input_frame, text='incl (deg)').grid(row=2, column=0)
incl_var = tk.StringVar(value='60.0')
tk.Entry(input_frame, textvariable=incl_var, width=10).grid(row=2, column=1)

tk.Label(input_frame, text='star temp (K)').grid(row=2, column=2)
ts_var = tk.StringVar(value='9520.0')
tk.Entry(input_frame, textvariable=ts_var, width=10).grid(row=2, column=3)

tk.Label(input_frame, text='plsig').grid(row=2, column=4)
plsig_var = tk.StringVar(value='-2.0')
tk.Entry(input_frame, textvariable=plsig_var, width=10).grid(row=2, column=5)

tk.Label(dust_frame, text='silicate').grid(row=0, column=0)
silicate_var = tk.StringVar(value='1.00')
tk.Entry(dust_frame, textvariable=silicate_var, width=10).grid(row=0, column=1)

tk.Label(dust_frame, text='carbon').grid(row=0, column=2, padx=5)
carbon_var = tk.StringVar(value='0.00')
tk.Entry(dust_frame, textvariable=carbon_var, width=10).grid(row=0, column=3)

tk.Label(dust_frame, text='T_in (K)').grid(row=0, column=4)
tin_var = tk.StringVar(value='1500')
tk.Entry(dust_frame, textvariable=tin_var, width=10).grid(row=0, column=5)

tk.Label(dust_frame, text='forsterite').grid(row=1, column=0)
forsterite_var = tk.StringVar(value='0.00')
tk.Entry(dust_frame, textvariable=forsterite_var, width=10).grid(row=1, column=1)

tk.Label(dust_frame, text='ice').grid(row=1, column=2, padx=5)
ice_var = tk.StringVar(value='0.00')
tk.Entry(dust_frame, textvariable=ice_var, width=10).grid(row=1, column=3)

tk.Label(dust_frame, text='chi*').grid(row=1, column=4)
chi_var = tk.StringVar(value='0')
tk.Entry(dust_frame, textvariable=chi_var, width=10).grid(row=1, column=5)

tk.Label(dust_frame, text='silbeta1').grid(row=2, column=0)
silbeta1_var = tk.StringVar(value='0.00')
tk.Entry(dust_frame, textvariable=silbeta1_var, width=10).grid(row=2, column=1)

# tk.Label(dust_frame, text='star temp (K)').grid(row=2, column=2)
# ts_var = tk.StringVar(value='9520.0')
# tk.Entry(dust_frame, textvariable=ts_var, width=10).grid(row=2, column=3)

tk.Label(dust_frame, text='albedo*').grid(row=2, column=4)
albedo_var = tk.StringVar(value='0')
tk.Entry(dust_frame, textvariable=albedo_var, width=10).grid(row=2, column=5)

plot_type = tk.StringVar(value='hr')
tk.Radiobutton(plot2_button_frame, text='Plot temperature(R)', variable=plot_type, value='T', command=change_plottype).grid(row=0, column=0)
tk.Radiobutton(plot2_button_frame, text='Plot height(R)', variable=plot_type, value='hr', command=change_plottype).grid(row=1, column=0)
tk.Radiobutton(plot2_button_frame, text='Plot surface density(R)', variable=plot_type, value='sigma', command=change_plottype).grid(row=0, column=1)
tk.Radiobutton(plot2_button_frame, text='Plot tau', variable=plot_type, value='tau', command=change_plottype).grid(row=1, column=1)


window.mainloop()