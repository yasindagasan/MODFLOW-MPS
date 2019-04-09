#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:52:56 2018

@author: dagasany
"""
import flopy
import flopy.utils.binaryfile as bf
import numpy as np

import os
import sys
import shutil

# Alternative
# Alternative
# Alternative
model_ws_trueK =os.path.join("modflow", "modflowFiles")

modelname_trueK = 'true_flow'

exe_name = os.path.join(os.getcwd(), 'modflow/exeMF/mf2005')

mf_trueK = flopy.modflow.Modflow(modelname=modelname_trueK, exe_name=exe_name,
                                model_ws=model_ws_trueK)




# --------------------
# Model domain and grid definition
Lx   =  100. # Length in x direction
Ly   =  100. # Length in y direction
ztop =  0.   # Top elevation of the layer
zbot = -50.  # Bottom elevation of the layer
nlay =  1    # number of layers
nrow =  100  # number of rows (number of grid nodes in y direction)
ncol =  100  # number of columns (number of grid nodes in x direction)
wcol =  50   # x index for the well to discharge or recharge
wrow =  50   # y index for the well to discharge or recharge

delr = Lx/ncol # Calculate the length of each grid ALONG the rows
delc = Ly/nrow # Calculate the length of each grid ALONG the columns
delv = (ztop - zbot) / nlay # Calculate the length of the grid along layers
botm = np.linspace(ztop, zbot, nlay + 1) # Vector for Bottom and top Elevations
wpt = ((wcol+0.5)*delr, Lx - ((wrow + 0.5)*delr))

# Create the discretization object (grid)
dis_trueK = flopy.modflow.ModflowDis(mf_trueK, nlay, nrow,
                                    ncol, delr=delr, delc=delc,
                                    top=ztop, botm=botm[1])

# Variables for the BAS (basic) package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32) # Create an empty 1s array
ibound[:, :, 0]  = -1 # Constant head values at the left boundary
ibound[:, :, -1] = -1 # Constant head values at the right boundary
strt = np.ones((nlay, nrow, ncol), dtype=np.float32) # Create an empty 1s array
strt[:, :, 0]  = 1. # Boundary conditions: Head values at the left boundary
strt[:, :, -1] = 0. # Boundary conditions: Head values at the right boundary
bas_trueK = flopy.modflow.ModflowBas(mf_trueK, ibound=ibound, strt=strt)

# Add the well package
# Remember to use zero-based layer, row, column indices!


def flowSimforTrue (realNum, real, pumpRate=-0.003):

    pumping_rate = pumpRate
    wel_sp = [[0, wrow, wcol, pumping_rate]] # lay, row, col index, pumping rate
    stress_period_data = {0: wel_sp} # define well stress period {period, well info dictionary}
    wel_trueK = flopy.modflow.ModflowWel(mf_trueK, stress_period_data=stress_period_data)



    # Add OC (Output Control Option) package to the MODFLOW model
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc_trueK = flopy.modflow.ModflowOc(mf_trueK, stress_period_data=spd, compact=True)


    # Add PCG (Preconditioned-Conj. Gradient Solver) package to the MODFLOW model
    pcg_trueK = flopy.modflow.ModflowPcg(mf_trueK) # Solver


    # Add LPF (Layer Property Flow) package to the MODFLOW model
    lpf_trueK = flopy.modflow.ModflowLpf(mf_trueK, hk=real.val[realNum,:,:,:], vka=0., ipakcb=53)
    # Write the MODFLOW model input files into the model directory
    mf_trueK.write_input()
    success_trueK, buff_trueK = mf_trueK.run_model()
    hds = bf.HeadFile(model_ws_trueK+"/"+ modelname_trueK + '.hds')
    head = hds.get_data(totim=1.0)
    if success_trueK:
        print("Flow for True Field", 1, ":Success!")
    else:
        print("There is a problem with the Field Flow", 1, ":Error!")

    return head;
