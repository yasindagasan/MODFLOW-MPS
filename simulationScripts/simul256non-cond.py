# import
import numpy as np
import matplotlib.pyplot as plt

# import from package 'geone'
from geone import img
import geone.deesseinterface as dsi
from geone.deesse_core.deesse import MPDS_MISSING_VALUE # constant for missing value
import sys
sys.path.insert(0, '../')
# import the custom functions
from func import forward256 as forward
from func import progressBar


# Read the TI (Img Class)
ti_filename = 'data/ti_strebelle_K.gslib' # strebelle with K values
ti = img.readImageGslib(ti_filename, missing_value=MPDS_MISSING_VALUE)

# DeeSse input parameters to create the true K field
deesse_input_true = dsi.DeesseInput(
    nx=256,ny=256,nz=1,nv=1,varname='facies',
    nTI=1,TI=ti,
    searchNeighborhoodParameters=dsi.SearchNeighborhoodParameters(rx=40.,ry=40.),
    nneighboringNode=50,
    distanceThreshold=0.02,
    maxScanFraction=0.25,
    npostProcessingPathMax=1,
    seed=14,
    nrealization=1)

# DeeSse input parameters to create the realisations for training the ML
deesse_input_realisations = dsi.DeesseInput(
    nx=256,ny=256,nz=1,nv=1,varname='facies',
    nTI=1,TI=ti,
    searchNeighborhoodParameters=dsi.SearchNeighborhoodParameters(rx=40.,ry=40.),
    nneighboringNode=50,
    distanceThreshold=0.02,
    maxScanFraction=0.25,
    npostProcessingPathMax=1,
    seed=444,
    nrealization=700)

# well location
wpt = ((64+0.5), 128 - ((64 + 0.5))) # Well location
# perform the DS simulation to create the reference K field
trueKList = dsi.deesseRun(deesse_input_true)
trueK     = trueKList["sim"]
trueK     = img.gatherImages(trueK,varInd=0)

# write the reference as a GSLIB file
img.writeImageGslib(trueK,filename= ("results/GSLIB/256*256_non-cond/trueK.gslib"))

# Do DS simulations to create
# --------------------
doSimul=True

if doSimul:
    simKList  = dsi.deesseRun(deesse_input_realisations)
    simK      = simKList["sim"]
    simK      = img.gatherImages(simK)
    img.writeImageGslib(simK,filename= ("results/GSLIB/256*256_non-cond/simKAll.gslib"))

    # Write the simulations in seperate GSLIB files
    for i in range(simK.val.shape[0]):
        temp_simK = img.copyImg(simK)
        temp_simK.remove_allvar()
        temp_simK.append_var(simK.val[i,:,:,:])
        img.writeImageGslib(temp_simK,
                            filename="results/GSLIB/256*256_non-cond/realisations/real_%05.d.gslib" % i)
else:
    # Import the fields
    trueK_filename = 'results/GSLIB/256*256_non-cond/trueK.gslib'
    simK_filename  = 'results/GSLIB/256*256_non-cond/simKAll.gslib'
    trueK = img.readImageGslib(trueK_filename, missing_value=MPDS_MISSING_VALUE)
    simK  = img.readImageGslib(simK_filename, missing_value=MPDS_MISSING_VALUE)

if doSimul:
    mf_trueK=forward.Modflow()
    # flow simulation for the reference parameter field
    flow_trueK = img.copyImg(trueK)
    flow_trueK.remove_allvar()
    predTrueK=mf_trueK.predict(model=trueK,imod=0,modelName="forwTrueK")
    flow_trueK.append_var(predTrueK[0])
    img.writeImageGslib(flow_trueK,filename="results/GSLIB/256*256_non-cond/flow_trueK.gslib" )

    mf_simK=forward.Modflow()
    # flow simulations for realisations
    for i in range(simK.val.shape[0]):
        flow_simK = img.copyImg(simK)
        flow_simK.remove_allvar()
        predSimK=mf_simK.predict(model=simK,imod=i,modelName="forwSimK")
        flow_simK.append_var(predSimK[0])
        img.writeImageGslib(flow_simK,
                            filename="results/GSLIB/256*256_non-cond/flowRealisations/flow_%05.d.gslib" % i)
        progressBar.update_progress(i / simK.val.shape[0])
else:
    flow_trueK_filename = 'results/GSLIB/flow_trueK.gslib'
    flow_simK_filename  = 'results/GSLIB/flow_simK.gslib'
    flow_trueK = img.readImageGslib(flow_trueK_filename, missing_value=MPDS_MISSING_VALUE)
    flow_simK  = img.readImageGslib(flow_simK_filename, missing_value=MPDS_MISSING_VALUE)
