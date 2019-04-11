import flopy
import flopy.utils.binaryfile as bf
import numpy as np

import os

class Modflow():
    def __init__(self):
        # trained image dimensions that should be input in the generator model
        self.dref = np.array([
            0.7066451358, 0.0817266470, 0.3714137884, -0.0815579066,
            -0.4690818427, 0.1376200029, 0.0277398729, 0.6678132149,
            0.1162949964
        ])
        self.err_tol = np.array([0.07])

        self.obs_loc = [(20, 80), (80, 80), (35, 65), (65, 65), (50, 50),
                        (35, 35), (65, 35), (20, 20), (80, 20)]

        self.img_res = (128, 128)

        # define the discretisation

        # Model domain and grid definition
        self.Lx = 128.  # Length in x direction
        self.Ly = 128.  # Length in y direction
        self.ztop = 0.  # Top elevation of the layer
        self.zbot = -1.  # Bottom elevation of the layer
        self.nlay = 1  # number of layers
        self.nrow = 128  # number of rows (number of grid nodes in y direction)
        self.ncol = 128  # number of columns (number of grid nodes in x direction)
        self.wcol = 64  # x index for the well to discharge or recharge
        self.wrow = 64  # y index for the well to discharge or recharge
        self.delr = self.Lx / self.ncol  # Calculate the length of each grid ALONG the rows
        self.delc = self.Ly / self.nrow  # Calculate the length of each grid ALONG the columns
        self.delv = (self.ztop -self.zbot) / self.nlay  # Calculate the length of the grid along layers
        self.botm = np.linspace(self.ztop, self.zbot, self.nlay + 1)  # Vector for Bottom and top Elevations
        self.wpt = ((self.wcol + 0.5) * self.delr, self.Lx - ((self.wrow + 0.5) * self.delr))  # Well location
        self.pumping_rate = -0.003

        #  directory to store the modflow input files
        self.model_ws = os.path.join("modflow/modflowFiles")



        # directory for the modflow executable
        self.exe_name = os.path.join(os.getcwd(), 'modflow/exeMF/mf2005')



    def predict(self, model,imod=0,modelName='forwdModel'):
        """ _RUN_FORWARD_MODFLOW(...) implements the forward operator using MODFLOW software
                :param model:   (m-tuple) Tuple of 'MType' instances
                :param imod:    (int) Model index
                :return:        (M, ndarray) Set of predicted observations
                """
        # name of the modflow name
        if imod==0:
            self.modelname = modelName + str(imod)
        else:
            self.modelname = modelName + "Tempsim"

        # create modflow forward model
        mf_forwdModel = flopy.modflow.Modflow(
            modelname=self.modelname, exe_name=self.exe_name, model_ws=self.model_ws)


        # Create the discretization object (grid)
        dis_forwdModel = flopy.modflow.ModflowDis(
            mf_forwdModel,
            self.nlay,
            self.nrow,
            self.ncol,
            delr=self.delr,
            delc=self.delc,
            top=self.ztop,
            botm=self.botm[1])

        # Variables for the BAS (basic) package
        ibound = np.ones((self.nlay, self.nrow, self.ncol),
                         dtype=np.int32)  # Create an empty 1s array
        ibound[:, :, 0] = -1  # Constant head values at the left boundary
        ibound[:, :, -1] = -1  # Constant head values at the right boundary
        strt = np.ones((self.nlay, self.nrow, self.ncol),
                       dtype=np.float32)  # Create an empty 1s array
        strt[:, :,0] = 1.  # Boundary conditions: Head values at the left boundary
        strt[:, :,-1] = 0.  # Boundary conditions: Head values at the right boundary
        bas_forwdModel = flopy.modflow.ModflowBas(mf_forwdModel, ibound=ibound, strt=strt)

        # Assign the permeability values

        # Add LPF (Layer Property Flow) package to the MODFLOW model
        lpf_forwdModel = flopy.modflow.ModflowLpf(mf_forwdModel, hk=model.val[imod,:,:,:], vka=0., ipakcb=53)

        # Add the well package
        # Remember to use zero-based layer, row, column indices!
        wel_sp = [[0, self.wrow, self.wcol,self.pumping_rate]]  # lay, row, col index, pumping rate
        stress_period_data = {0: wel_sp}  # define well stress period {period, well info dictionary}
        wel_forwdModel = flopy.modflow.ModflowWel(mf_forwdModel, stress_period_data=stress_period_data)

        # Add OC (Output Control Option) package to the MODFLOW model
        spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget'] }
        oc_mf_forwdModel = flopy.modflow.ModflowOc(mf_forwdModel, stress_period_data=spd, compact=True)

        # Add PCG (Preconditioned-Conj. Gradient Solver) package to the MODFLOW model
        pcg_mf_forwdModel = flopy.modflow.ModflowPcg(mf_forwdModel)  # Solver

        # Write the MODFLOW model input files into the model directory
        mf_forwdModel.write_input()

        # run the forward MODFLOW model
        mf_forwdModel.run_model(silent=True)

        # read the head values
        hds = bf.HeadFile(self.model_ws + "/" + self.modelname + '.hds')
        head = hds.get_data(totim=1.0)

        return head, mf_forwdModel
