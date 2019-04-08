#!/usr/bin/python3
#-*- coding: utf-8 -*-

# At start in the current directory:
# .
# |
# +--pts.gslib      : conditioning points for DeeSse simulation
# +--test.py        : this file
# +--ti.gslib       : training image for DeeSse simulation
#
# Run this python script in interactive mode (in a python shell):
# ----
# with open("test.py") as f: s = f.read()
#
# exec(s)
# ---
#set_env unine_LICENSE = /home/local/UNINE.CH/dagasany/Documents/PostDoc/Software/DeeSse_Python/
# Import for the MPS
#import tkinter
import numpy as np
import os 
import sys
sys.path.insert(0, '../')
import mps.deesse.deesse as deesse
import mps.deesse.deesseinterface as dsi
from   mps.deesse.deesseinterface import DeesseInput
import mps.data.img as img

import matplotlib.pyplot as plt
import mps.view.imgplot as imgplt
import mps.view.customcolors as ccol
import flopy.utils.binaryfile as bf


import func.flowSimforReals as fwReal
import func.flowSimforTrue as fwTrue

import flopy


#####################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           #################
########## MPS Simulations############
######################################

# Set missing value
# -----------------
missing_value = deesse.MPDS_MISSING_VALUE # -9999999

# Read files
# ----------
# Read the TI (Img Class)
ti_filename = 'data/ti_strebelle_K.gslib'
ti = img.readImageGslib(ti_filename, missing_value=missing_value)

# Set input for deesse
# --------------------
# DeeSse input parameters to create the true K field
deesse_input_true = DeesseInput(
    nx=128,ny=128,nz=1,nv=1,varname='facies',
    nTI=1,TI=ti,
    searchNeighborhoodParameters=dsi.SearchNeighborhoodParameters(rx=40.,ry=40.),
    nneighboringNode=30,
    distanceThreshold=0.02,
    maxScanFraction=0.25,
    npostProcessingPathMax=1,
    seed=1234,
    nrealization=20)

# DeeSse input parameters to create the realisations for training the ML
deesse_input_realisations = DeesseInput(
    nx=128,ny=128,nz=1,nv=1,varname='facies',
    nTI=1,TI=ti,
    searchNeighborhoodParameters=dsi.SearchNeighborhoodParameters(rx=40.,ry=40.),
    nneighboringNode=30,
    distanceThreshold=0.02,
    maxScanFraction=0.25,
    npostProcessingPathMax=1,
    seed=444,
    nrealization=10000)

# Do deesse simulations to generate the true field and the realisations of it
# --------------------
# Do deesse simulations to generate the true field
doSimul=True

if doSimul:
    trueKList = dsi.deesseRun(deesse_input_true)
    trueK     = img.gatherImages(trueKList)
    # Do deesse simulation to generate the realisations
    simKList  = dsi.deesseRun(deesse_input_realisations)
    simK      = img.gatherImages(simKList)
    # Write the simulations and true field in GSLIB files
    img.writeImageGslib(trueK,filename= ("results/GSLIB/128*128/trueK.gslib"))
    img.writeImageGslib(simK,filename= ("results/GSLIB/128*128/simK.gslib"))
else:
    # Import the fields
    trueK_filename = 'results/GSLIB/trueK.gslib'
    simK_filename  = 'results/GSLIB/simK.gslib'
    trueK = img.readImageGslib(trueK_filename, missing_value=missing_value)
    simK  = img.readImageGslib(simK_filename, missing_value=missing_value)

######################################
######### Flow Simulations############
######################################

# --------------------
# Assign name and create modflow model object
#modelname_trueK = 'modflow/modflowFiles/true_flow'
#modelname_simK = 'modflow/modflowFiles/sim_flow'
#mf_trueK = flopy.modflow.Modflow(modelname_trueK, exe_name='modflow/exeMF/mf2005',)
#mf_simK = flopy.modflow.Modflow(modelname_simK, exe_name='modflow/exeMF/mf2005')


# Alternative 
model_ws_trueK =os.path.join("modflow", "modflowFiles")
model_ws_simK  =os.path.join("modflow", "modflowFiles")

modelname_trueK = 'true_flow'
modelname_simK  = 'sim_flow'

exe_name = os.path.join(os.getcwd(), 'modflow/exeMF/mf2005')

mf_trueK = flopy.modflow.Modflow(modelname=modelname_trueK, exe_name=exe_name,
                                model_ws=model_ws_trueK)
mf_simK = flopy.modflow.Modflow(modelname=modelname_simK, exe_name=exe_name,
                                 model_ws=model_ws_simK)

trueKindex = 4
# --------------------
# Model domain and grid definition
Lx   =  128. # Length in x direction
Ly   =  128. # Length in y direction 
ztop =  0.   # Top elevation of the layer
zbot = -50.  # Bottom elevation of the layer 
nlay =  1    # number of layers 
nrow =  128  # number of rows (number of grid nodes in y direction)
ncol =  128  # number of columns (number of grid nodes in x direction)
wcol =  64   # x index for the well to discharge or recharge
wrow =  64   # y index for the well to discharge or recharge

delr = Lx/ncol # Calculate the length of each grid ALONG the rows
delc = Ly/nrow # Calculate the length of each grid ALONG the columns
delv = (ztop - zbot) / nlay # Calculate the length of the grid along layers
botm = np.linspace(ztop, zbot, nlay + 1) # Vector for Bottom and top Elevations
wpt = ((wcol+0.5)*delr, Lx - ((wrow + 0.5)*delr)) # Well location

# Create the discretization object (grid)
dis_trueK = flopy.modflow.ModflowDis(mf_trueK, nlay, nrow, 
                                     ncol, delr=delr, delc=delc,
                                     top=ztop, botm=botm[1])
dis_simK = flopy.modflow.ModflowDis(mf_simK, nlay, nrow,
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
bas_simK  = flopy.modflow.ModflowBas(mf_simK, ibound=ibound, strt=strt)

# Add LPF (Layer Property Flow) package to the MODFLOW model
lpf_trueK = flopy.modflow.ModflowLpf(mf_trueK, hk=trueK.val[trueKindex,:,:,:], vka=0., ipakcb=53)
lpf_simK  = flopy.modflow.ModflowLpf(mf_simK, hk=simK.val[trueKindex,:,:,:], vka=0., ipakcb=53)

# Add the well package
# Remember to use zero-based layer, row, column indices!
pumping_rate = -0.003
wel_sp = [[0, wrow, wcol, pumping_rate]] # lay, row, col index, pumping rate
stress_period_data = {0: wel_sp} # define well stress period {period, well info dictionary}
wel_trueK = flopy.modflow.ModflowWel(mf_trueK, stress_period_data=stress_period_data)
wel_simK = flopy.modflow.ModflowWel(mf_simK, stress_period_data=stress_period_data)


# Add OC (Output Control Option) package to the MODFLOW model
spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
oc_trueK = flopy.modflow.ModflowOc(mf_trueK, stress_period_data=spd, compact=True)
oc_simK  = flopy.modflow.ModflowOc(mf_simK, stress_period_data=spd, compact=True)

# Add PCG (Preconditioned-Conj. Gradient Solver) package to the MODFLOW model
pcg_trueK = flopy.modflow.ModflowPcg(mf_trueK) # Solver
pcg_simK = flopy.modflow.ModflowPcg(mf_simK) # Solver

# Write the MODFLOW model input files into the model directory
mf_trueK.write_input()
mf_simK.write_input()

# Run the MODFLOW model and check if the operation is successful
if doSimul:
    success_trueK, buff_trueK = mf_trueK.run_model()
    success_simK,  buff_simK = mf_simK.run_model()

######################################
######### Flow Sim for Reals##########
######################################

#-----------
# Create the flow simulations for each of the permeability realisations


if doSimul:
    
    
    
    
    for i in range(simK.val.shape[0]):
        flow_simK = img.copyImg(simK)
        flow_simK.remove_allvar()
        temp_simK = img.copyImg(simK)
        temp_simK.remove_allvar()

        flow_simK.append_var(fwReal.flowSimforReals(i,simK,
                                                      pumpRate=pumping_rate))
        img.writeImageGslib(flow_simK,filename="results/GSLIB/128*128/flow_simK/flow_%05.d.gslib" % i)
        
        temp_simK.append_var(simK.val[i,:,:,:])
        img.writeImageGslib(temp_simK,filename="results/GSLIB/128*128/simK/real_%05.d.gslib" % i)

        #flow_trueK.append_var(fwTrue.flowSimforTrue(trueKindex,trueK,
         #                                        pumpRate=pumping_rate))
    #img.writeImageGslib(flow_trueK,filename= ("results/GSLIB/128*128/flow_trueK.gslib"))
    #img.writeImageGslib(flow_simK,filename= ("results/GSLIB/128*128/flow_simK.gslib"))

    
    for i in range(trueK.val.shape[0]):
        temp_trueK = img.copyImg(trueK)
        temp_trueK.remove_allvar()
        flow_trueK = img.copyImg(trueK)
        flow_trueK.remove_allvar()
        
        flow_trueK.append_var(fwReal.flowSimforReals(i,trueK,
                                                      pumpRate=pumping_rate))
        img.writeImageGslib(flow_trueK,filename="results/GSLIB/128*128/flow_trueK/flow_%05.d.gslib" % i)
        
        temp_trueK.append_var(simK.val[i,:,:,:])
        img.writeImageGslib(temp_trueK,filename="results/GSLIB/128*128/trueK/real_%05.d.gslib" % i)

       

        #flow_trueK.append_var(fwTrue.flowSimforTrue(trueKindex,trueK,
         #                                        pumpRate=pumping_rate))
else:
    flow_trueK_filename = 'results/GSLIB/flow_trueK.gslib'
    flow_simK_filename  = 'results/GSLIB/flow_simK.gslib'
    flow_trueK = img.readImageGslib(flow_trueK_filename, 
                                         missing_value=missing_value)
    flow_simK  = img.readImageGslib(flow_simK_filename, 
                                          missing_value=missing_value)

    

    


#flowSim_reals = list(range(simK.val.shape[0]));
#for i in range(simK.val.shape[0]):
#    flowSim_reals[i]=fwReal.flowSimforReals(i,simK,pumpRate=pumping_rate)
    
######################################
######### Plot The Variables##########
######################################

headsInd = flow_simK.sample(spacing=20) # get the index values for the measurement pts.


# Display
# -------
categ = ti.get_unique() # get the unique ctegories of the TI

# Set colors for categories
categ_col = ccol.clr_chart_list_1[0:len(categ)]

# Draw TI
fig, ax = plt.subplots(2, 2, figsize=(12,10))
plt.subplot(2, 2, 1)
imgplt.drawImage2D(ti,
                   categ=True, categVal=categ, categCol=categ_col,
                   title='TI')

plt.subplot(2, 2, 2)
imgplt.drawImage2D(trueK,iv=trueKindex,
                   categ=True, categVal=categ, categCol=categ_col,
                   title='True Field')
plt.plot(wpt[0],wpt[1],'v',markersize=12, color="black")
plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=14, color="black")


for i in list(range(2)) :
    plt.subplot(2, 2, i+3)
    imgplt.drawImage2D(simK,iv=i,
                       categ=True, categVal=categ, categCol=categ_col,
                       title='sim #{}'.format(i))
    plt.plot(wpt[0],wpt[1],'v',markersize=12, color="black")
    plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=14, color="black")
plt.savefig('results/plots/TI_and_realisations.png')


###############################
######### True Flow Res #######
###############################

## Post process the results

fig_flow, ax = plt.subplots(2, 2, figsize=(12,10))
plt.subplot(2, 2, 1, aspect='equal')
hds = bf.HeadFile(model_ws_trueK+"/"+ modelname_trueK + '.hds')
head = hds.get_data(totim=1.0)
levelsDefine = np.arange(np.min(head),np.max(head),0.2)
extent = (delr / 2, Lx - delr / 2, Ly - delc / 2, delc / 2)
#plt.contour(head[0, :, :], levels=levelsDefine, extent=extent)
plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=14, color="black")

imgplt.drawImage2D(trueK,iv=trueKindex,
                   categ=True, categVal=categ, categCol=categ_col,
                   title='True Field', colorbar_pad_fraction=0.2)
plt.plot(wpt[0],wpt[1],'k^:',markersize=12)



plt.subplot(2, 2, 2, aspect='equal')
hds = bf.HeadFile( model_ws_trueK+"/"+ modelname_trueK+'.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])
levels = np.linspace(0, 10, 5) # contour spacing 

cbb = bf.CellBudgetFile( model_ws_trueK+"/"+ modelname_trueK+'.cbc')
kstpkper_list = cbb.get_kstpkper()
frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]

modelmap = flopy.plot.ModelMap(model=mf_trueK, layer=0)
qm = modelmap.plot_ibound()
#cs = modelmap.contour_array(head, levels=levelsDefine, colors='b')
quiver = modelmap.plot_discharge(frf, fff, head=head, istep=5, jstep=6)
#plt.clabel(cs, inline=1, fontsize=12)
plt.plot(wpt[0],wpt[1],'r^:',markersize=12)
plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=14, color="red")


modelmap = flopy.plot.ModelMap(model=mf_trueK, layer=0)
qm = modelmap.plot_ibound()
plt.subplot(2, 2, 3, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf_trueK, layer=0)
qm = modelmap.plot_ibound()
cs = modelmap.contour_array(head, levels=levelsDefine, colors='red')
quiver = modelmap.plot_discharge(frf, fff, head=head, istep=5, jstep=6)
plt.clabel(cs, inline=1, fontsize=12)


plt.subplot(2, 2, 4, aspect='equal')
plt.imshow(head[0, ::-1, :], origin="lower")
plt.colorbar()
plt.title("Head values (True K)")
#hds.plot()
#plt.clabel(cs, inline=0, fontsize=8,)
#quadmesh = modelmap.plot_bc('WEL', kper=5, plotAll=True)
contours=plt.contour(head[0, ::-1, :],3, origin="lower", levels=levelsDefine,colors="black") 
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=14, color="black")
plt.savefig('results/plots/flow_simulations_trueK.png')

###############################
######### Simu Flow Res #######
###############################

nrowIm =4
ncolIm =4


levelsDefine = np.arange(np.min(head),np.max(head),0.4)

fig_reals, ax = plt.subplots(6, 6, figsize=(12,12))
fig_reals.suptitle("Title for whole figure\n")
plt.subplot(nrowIm, ncolIm, 1, aspect='equal')
imgplt.drawImage2D(trueK,iv=trueKindex,
                   categ=True, categVal=categ, categCol=categ_col,
                   title='True Field')

plt.plot(wpt[0],wpt[1],'k^:',markersize=12)

plt.subplot(nrowIm, ncolIm, 2, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf_trueK, layer=0)
qm = modelmap.plot_ibound()
quiver = modelmap.plot_discharge(frf, fff, head=head, istep=5, jstep=6)
plt.plot(wpt[0],wpt[1],'r^:', markersize=12)
plt.title("Flow Directions")

plt.subplot(nrowIm, ncolIm, 3, aspect='equal',)
a=plt.imshow(head[0, ::-1, :], origin="lower")
ccol.add_colorbar(a)
plt.title("True Heads")

plt.subplot(nrowIm, ncolIm, 4, aspect='equal')
a=plt.imshow(head[0, ::-1, :], origin="lower")
contours=plt.contour(head[0, ::-1, :], origin="lower", levels=levelsDefine,colors="black") 
plt.clabel(contours, inline=True, fontsize=8)
ccol.add_colorbar(a)
plt.title("True Heads")

for i in range(6):
    plt.subplot(nrowIm, ncolIm, 2*i+5, aspect='equal')
    imgplt.drawImage2D(simK,iv=i,
                       categ=True, categVal=categ, categCol=categ_col,
                       title='sim #{}'.format(i))
    plt.plot(wpt[0],wpt[1],'v',markersize=8, color="black")
    plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=5, color="black")

    #plt.plot(wpt[0],wpt[1],'k^:',markersize=12)
    plt.subplot(nrowIm, ncolIm, 2*i+6, aspect='equal')
    a=img.readImageGslib(filename="results/GSLIB/128*128/flow_trueK/flow_%05.d.gslib" % i)
    headsInd = a.sample(spacing=16) # get the index values for the measurement pts.

    imgplt.drawImage2D(a,
                       categ=False,
                       title='Heads for #{}'.format(i))
    plt.plot(wpt[0],wpt[1],'v',markersize=8, color="black")
    plt.plot(headsInd[:,0],headsInd[:,1],'.',markersize=5, color="black")

    
    
    #a=plt.imshow(flowSim_reals[i][0, ::-1, :], origin="lower")
    #ccol.add_colorbar(a)
    #plt.title('Heads for #{}'.format(i) )
plt.tight_layout()    
plt.savefig('results/plots/flow_simulations_realisations.png')

