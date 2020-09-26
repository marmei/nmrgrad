
from .klayout_py_grad_coil import (KlayoutPyGradCoil,
                                   discretise_wire)

from .klayout_grad import (arrayToPointList, 
                           placeRoundPath,
                           placeText,
                           placeArc,
                           KlayoutPyFeature)

import nmrgrad
import numpy as np
try:
    import pya
except:
    # import without klayout GUI
    import klayout.db as pya

class StraightGradCoil(KlayoutPyGradCoil):
    TargetCellName = "YZGrad"

    LayerViaInfosCoil = None
    LayerInfoIntercon = [pya.LayerInfo(0, 0),
                         pya.LayerInfo(2, 0)] # the fluidic cooling layer

    LayerInfo = pya.LayerInfo(5, 0)
    InterConnectTop = False

    xPosCondStartZ = 3.10e6 # at the connector
    xPosCondEndZ = 3.10e6 # at the curve
    viaYcorrectZ = 100e3

    xPosCondStartY = 3.0e6
    xPosCondEndY = 3.2e6

    width = 100e3 # um
    widthArc = 100e3

    connector01 = 1.2e6
    connector02 = 2.4e6

    widthVia = 250e3
    lenVia = 250e3

    yOffsetVia = 90e3

    def __init__(self, TopCellName, GradCoil, chipType="Bot"):
        KlayoutPyGradCoil.__init__(self, TopCellName, GradCoil)
        self.removeCell()

        self.z0 = eval(GradCoil["@zNOTCal_top"]) # in mm

        if GradCoil["@type"] == "AndersonZ":
            self.TargetCellName = "GZ" + str(self.z0) + "A"
        if GradCoil["@type"] == "AndersonY":
            self.TargetCellName = "GY" + str(self.z0) + "A"
        if "@cWArc" in GradCoil:
            self.widthArc = eval(GradCoil["@cWArc"])
        if "@cWidth" in GradCoil:
            self.width    = eval(GradCoil["@cWidth"])
        if "@dVia" in GradCoil:
            self.widthVia = eval(GradCoil["@dVia"])
            self.lenVia   = eval(GradCoil["@dVia"])
        if "@viaYcorrectZ" in GradCoil:
            self.viaYcorrectZ = eval(GradCoil["@viaYcorrectZ"])
        if "@resistMinWidth" in GradCoil:
            self.resistMinWidth = eval(GradCoil["@resistMinWidth"])

        if chipType == "Top":
            self.InterConnectTop = True

        if "@dViaOld" in GradCoil:
            self.dViaOld = eval(GradCoil["@dViaOld"])
        else:
            self.dViaOld = eval(GradCoil["@dVia"])

        if "@dViaOut" in GradCoil:
            self.LayerViaInfos = [pya.LayerInfo(0, 0), pya.LayerInfo(2, 0),
                                  pya.LayerInfo(3, 0), pya.LayerInfo(4, 0)]
            self.LayerViaInfosCoil = [pya.LayerInfo(5, 1), pya.LayerInfo(5, 0)]
            self.dViaOut = eval(GradCoil["@dViaOut"])

        self.removeCell()
        if GradCoil["@type"] == "AndersonY" and chipType == "Top":
            self.makeCell(trans=pya.Trans.new(pya.Trans.R180, -0.0, -0.0))
        else:
            self.makeCell()

    @staticmethod
    def wire_to_conductor(cond, z_not, mirror_chip=1.0, mirror_y=1.0):
        """
        return mapped array for/from Klayout to Bfield calculatin
        with given zNot (e.g. xNot separation)
        cond: np.array([[x1, y1], ... [xn, yn]])
        """
        cond = discretise_wire(cond)
        return np.asarray([cond[:,0] * 1e-3 * mirror_chip,
                           cond[:,1] * 1e-3 * mirror_y,
                           np.zeros_like(cond[:, 0]) - mirror_chip * z_not * 1e-3]).swapaxes(0,1)

    def wire_to_round_path(self, pts, radius=500, LayerInfo=None):
        helperCell = KlayoutPyFeature(self.TargetCellName)
        helperCell.TargetCellName = "helperCell_del_Grad" + str(np.random.rand())
        helperCell.removeCell()
        helperCell.makeCell()

        if LayerInfo is None:
            LayerInfo = self.LayerInfo

        placeRoundPath(helperCell.TargetCell,
                       helperCell._ly,
                       radius=radius,
                       width=self.width * 1e-3,
                       pts=pts,
                       LayerInfo=LayerInfo)

    def makeInterconnect(self, connector, pos,
                         trans_angle=pya.Trans.R0, addlength=200e3):
        trans = pya.Trans(trans_angle, 0.0, 0.0)

        helperCell = self._ly.create_cell("helperCellAndersonGrad")
        self.TargetCell.insert(pya.CellInstArray.new(helperCell.cell_index(), trans))
        if connector == 0.0:
            vz = -1.0
        else:
            vz = 1.0

        array1 = np.array([[self.topChipWidth, connector + .5 * vz * self.widthConnector],
                           [self.topChipWidth - self.lenConnector - addlength,
                            connector + .5 * vz * self.widthConnector],
                           [pos[0] - 0.43 * self.lenVia, pos[1] + vz * self.yOffsetVia],
                           [pos[0] - 0.43 * self.lenVia, pos[1]- vz * 0.43 * self.lenVia],
                           [pos[0] + 0.43 * self.lenVia, pos[1]- vz * 0.43 * self.lenVia],
                           [pos[0] + 0.43 * self.lenVia, pos[1]],
                           [self.topChipWidth - self.lenConnector, connector - .5 * vz * self.widthConnector],
                           [self.topChipWidth, connector - .5 * vz * self.widthConnector]])

        for LayerInfo in self.LayerInfoIntercon:
            poly = pya.Polygon(arrayToPointList(array1))
            polyShape = poly.round_corners(125e3, 125e3, 32)
            helperCell.shapes(self._ly.layer(LayerInfo)).insert(polyShape)

        # LayerViaInfosCoil
        self.makeVia(pos, helperCell)
        if self.LayerViaInfosCoil != None:
            self.LayerViaInfos = self.LayerViaInfosCoil
            self.widthVia = self.dViaOut
            self.lenVia = self.dViaOut
            self.makeVia(pos, helperCell)

        self.TargetCell.flatten(-1, True)

    def make_anderson_z_grad(self):
        z0 = self.z0 * 1e3 ## convert to nm
        if self.zNOT_top_real is not None:
            z0real = self.zNOT_top_real * 1e3

        via01 = np.array([self.xPosCondStartZ,  z0 - self.viaYcorrectZ]) ## via on the right side
        self.makeVia(via01)
        self.makeVia(via01 * np.array([1, -1])) # other via simmilar

        self.makeInterconnect(self.connector01, via01, trans_angle=pya.Trans.M90)

        ## mirror the one interconnect
        if self.InterConnectTop is True:
            self.makeInterconnect(self.connector01, via01, trans_angle=pya.Trans.R180)
        else:
            self.makeInterconnect(0.0, via01, trans_angle=pya.Trans.R180)

        # make additional paths to add additional conductor with at the edges
        Zgrad_coil = np.asarray([via01,
                                 np.array([ self.xPosCondStartZ - self.viaYcorrectZ *.07, z0]),
                                 np.array([-self.xPosCondStartZ + self.viaYcorrectZ *.07, z0]),
                                 via01 * np.array([-1, 1])]) * 1e-3

        Zgrad_coil2 = np.asarray([via01,
                                  np.array([via01[0], z0]),
                                  np.array([-via01[0], z0]),
                                  via01 * np.array([-1, 1])]) * 1e-3

        Zgrad_coil3 = np.asarray([via01,
                                  np.array([ self.xPosCondStartZ - self.viaYcorrectZ * 2.2, z0]),
                                  np.array([-self.xPosCondStartZ + self.viaYcorrectZ * 2.2, z0]),
                                  via01 * np.array([-1, 1])]) * 1e-3

        Zgrad_coil4 = np.asarray([via01,
                                  np.array([ self.xPosCondStartZ - self.viaYcorrectZ * 5, z0]),
                                  np.array([-self.xPosCondStartZ + self.viaYcorrectZ * 5, z0]),
                                  via01 * np.array([-1, 1])]) * 1e-3

        self.wire_to_round_path(Zgrad_coil, radius=self.width * 20e-3)
        self.wire_to_round_path(Zgrad_coil2, radius=self.width * 1e-3)
        self.wire_to_round_path(Zgrad_coil3, radius=self.width * 10e-3)
        self.wire_to_round_path(Zgrad_coil4, radius=self.width * 10e-3)

        self.wire_to_round_path(self.mirrorArrayXaxis(Zgrad_coil), radius=self.width * 20e-3)
        self.wire_to_round_path(self.mirrorArrayXaxis(Zgrad_coil2), radius=self.width * 1e-3)
        self.wire_to_round_path(self.mirrorArrayXaxis(Zgrad_coil3), radius=self.width * 10e-3)
        self.wire_to_round_path(self.mirrorArrayXaxis(Zgrad_coil4), radius=self.width * 10e-3)

        arc_intecon = placeArc(self.TargetCell, self._ly,
                               trans=pya.Trans(pya.Trans.R0, self.xPosCondStartZ, 0),
                               r1=(z0 - self.viaYcorrectZ - .5 * self.widthVia) / 1e3,
                               r2=(z0 - self.viaYcorrectZ + .5 * self.widthVia) / 1e3,
                               angle_start=-90.0, angle_end=90.0)

        self.TargetCell.flatten(-1, True)

        intercon_01 = np.asarray([via01,
                                  [3800e3, self.connector01],
                                  [7200e3, self.connector01]]) * -1e-3
        intercon_02 = np.asarray([via01 * np.array([1.0, -1.0]), # Zgrad_coil[0],
                                  [via01[0], 0],
                                  [4000e3, 0]]) * -1e-3
        intercon_03 = np.asarray([[4000e3, 0],
                                  [7200e3, 0]]) * -1e-3

        # build up the conductor model for field calculation
        coilZ = np.vstack([
            self.wire_to_conductor(intercon_01[::-1], self.z0 + 100, mirror_y=-1.0),
                            self.wire_to_conductor(Zgrad_coil[::-1], self.z0),

                            self.wire_to_conductor(arc_intecon[::-1], self.z0 + 100),

                            self.wire_to_conductor(Zgrad_coil, self.z0, mirror_y=-1.0),
                            # TODO, add to list.
                            self.wire_to_conductor(intercon_01, self.z0 + 100),
                            self.wire_to_conductor(intercon_01[::-1], - self.z0 - 100),
                            self.wire_to_conductor(Zgrad_coil[::-1], - self.z0, mirror_y=-1.0),
                            self.wire_to_conductor(arc_intecon, -self.z0 - 100),
                            self.wire_to_conductor(Zgrad_coil, -self.z0),
                            self.wire_to_conductor(intercon_02, -self.z0 - 100),
                            self.wire_to_conductor(intercon_03, self.z0 + 100)
                           ])

        if self.zNOT_top_real != None:
            base_layer = 1080

            coilZ = np.vstack([self.wire_to_conductor(intercon_01[::-1], base_layer, mirror_y=-1.0),
                               self.wire_to_conductor(Zgrad_coil[::-1], self.zNOT_top_real),
                               self.wire_to_conductor(Zgrad_coil, self.zNOT_top_real, mirror_y=-1.0),
                               self.wire_to_conductor(intercon_01, base_layer),
                               self.wire_to_conductor(intercon_01[::-1], - base_layer),
                               self.wire_to_conductor(Zgrad_coil[::-1], - self.zNOT_top_real, mirror_y=-1.0),
                               self.wire_to_conductor(Zgrad_coil, -self.zNOT_top_real),
                               self.wire_to_conductor(intercon_02, -base_layer),
                               self.wire_to_conductor(intercon_03, base_layer)])

        self.conductors = [[coilZ, {"color":"b", "zorder":210, "current":1.0}]]
        self.condPlot3D = self.conductors

    def make_anderson_y_grad(self):
        # calculate the conductors
        GradY = nmrgrad.uniaxial.AndersonLinGradY()
        GradYStr, CondY01, CondY02 = GradY.max_grad_z_comb(self.z0 * 1e-3,
                                                           separation=(self.resistMinWidth + self.width) * 1e-6) * np.array([1, 1e6, 1e6])
        z0 = self.z0 * 1e3 # convert to nm

        # make the chip-chip interconnects
        via01 = [self.xPosCondStartY, z0 + self.dViaOld * 2.8]
        self.makeInterconnect(self.connector02, via01, trans_angle=pya.Trans.M90)

        # make the 1st conductor (outer or top)
        CondY02_Ypos_end = np.array([ self.xPosCondStartY - 1.5 * self.viaYcorrectZ, CondY02])
        thickerVia = -1 * np.array([self.width * 0.5, 0])
        CondY02_Ypos_01 = np.asarray([CondY02_Ypos_end + np.array([30e3, 0]),
                                      np.array([-self.xPosCondStartY + \
                                                self.viaYcorrectZ, CondY02])
                                      - thickerVia,
                                      via01 * np.array([-1, 1]) - thickerVia])

        CondY02_Ypos_02  = np.array([CondY02_Ypos_end + 45e3 * np.array([1, 0]),
                                     np.array([-self.xPosCondStartY + self.viaYcorrectZ,
                                               CondY02]) + thickerVia,
                                     via01 * np.array([-1, 1]) + thickerVia])
        r1_right = (CondY02 - (z0 - self.viaYcorrectZ) - .5 * self.width -25e3) / 1e3
        r2_right = r1_right + self.widthArc / 1e3

        xArch_right = self.xPosCondStartZ + .5 * self.lenVia - .5 * self.widthVia
        yArch_right = (z0 - self.viaYcorrectZ)

        xarchRight = CondY02 - (z0 - self.viaYcorrectZ) + xArch_right
        rad_centre = CondY02 - (z0 - self.viaYcorrectZ)

        CondY02ToArch = np.array([ CondY02_Ypos_end + np.array([0, self.width * .5]),
                                      CondY02_Ypos_end - np.array([0, self.width * .5]),
                                      np.array([xArch_right, r1_right * 1e3 + yArch_right]),
                                      np.array([xArch_right, r2_right * 1e3 + yArch_right])])

        # connect to the arch to the inner conductors CondY01
        ArchtoCondY01 = np.array([np.array([xArch_right,
                                            yArch_right - r1_right * 1e3 - self.widthArc * .5]),
                                  np.array([xArch_right - 0.75 * self.lenVia, yArch_right \
                                            - r1_right * 1e3 - self.widthArc * .5])])

        CondY01_Via = np.array([xArch_right - .75 * self.lenVia, yArch_right 
                                - r1_right * 1e3 - self.widthArc * .5 + .5 * (self.width - self.widthArc)]) # start from via
        CondY01_Start = np.array([xArch_right - self.lenVia, CondY01])
        CondY01_bot   = np.array([CondY01_Via * np.array([1, -1]),
                                  CondY01_Start * np.array([1, -1]),
                                  CondY01_Start * np.array([-1, -1]),
                                  CondY01_Via * np.array([-1, -1])])

        #CondY01_pos_left_RingCon
        CondY01_top = np.array([CondY01_Via * np.array([1, 1]),
                                CondY01_Start * np.array([1, 1]),
                                CondY01_Start * np.array([-1, 1]), CondY01_Via * np.array([-1, 0])
                                + (self.xPosCondStartZ + .5 * self.lenVia - .5 * self.widthVia) \
                                * np.array([0, 0]) \
                                + (CondY01_Via - self.width - 85e3) * np.array([0, -1])])

        r_centre_outer = CondY01_top[3][1] - yArch_right * -1 - self.width * 0.5

        # connector from the innerConductor CondY01 posY (1) lower to the outerRing
        CondY01_pos_left_RingCon = np.array([np.array([-xArch_right + 0.75 * self.lenVia,
                                                       -yArch_right + r_centre_outer + self.widthArc * .5]),
                                             np.array([-xArch_right,
                                                       -yArch_right + r_centre_outer + self.widthArc * .5])])

        # connector from the innerConductor CondY01 negY (1) lower to the Via02
        arrayConArchInner = np.array([np.array([-xArch_right + .75 * self.lenVia,
                                                -1 * (yArch_right - r1_right * 1e3 - self.widthArc * .5)]),
                                      np.array([-xArch_right,
                                                -1 * (yArch_right - r1_right * 1e3 - self.widthArc * .5)])])

        # (***) left lower arch to via
        via02 = np.array([xArch_right + r2_right * 1e3
                          - 0.5 * self.dViaOld, CondY02 + 100e3]) # z0 + self.viaYcorrectZ + 0.5 * self.widthVia])

        self.makeInterconnect(self.connector02, via02, trans_angle=pya.Trans.R180)

        arrayInnerVia02con = np.array([via02 * np.array([-1, 0])
                                       - np.array([.5 * self.lenVia - .5 * self.widthArc, 0]
                                                  + np.array([0, yArch_right * 1])),
                                       via02 * np.array([-1, -1])
                                       - np.array([.5 * self.lenVia - .5 * self.widthArc, 0])])

        arch_lowX = -via02[0] - .5 * self.lenVia + self.widthVia * .5
        arch_lowY = -via02[1]

        r_outer_lowX = (.5 * self.widthVia + self.resistMinWidth) * 1e-3

        moveItIntercon = arrayInnerVia02con[0][0] - self.widthArc - self.resistMinWidth * 1.5
        OuterInterconVia02 = np.array([[moveItIntercon, arrayInnerVia02con[0][1]],
                                       [moveItIntercon, arrayInnerVia02con[1][1]]])

        VerticalConnectIntercon = np.array([[-via02[0] + (.5 * self.lenVia - .5 * self.widthVia),
                                             -via02[1] - r_outer_lowX * 1e3 - .5 * self.widthArc],
                                            [-via02[0] - (.5 * self.lenVia - .5 * self.widthVia),
                                             -via02[1] - r_outer_lowX * 1e3 - .5 * self.widthArc]])

        # swap the conductor on the other side (outer) top Conductor
        CondY02_Ypos_01_mirror = self.mirrorArrayXaxis(CondY02_Ypos_01[:-2])
        CondY02_Ypos_01_mirror = np.append(CondY02_Ypos_01_mirror,
                                           CondY02_Ypos_01_mirror[-1::] * np.array([-1.0, 1.0]), axis=0)

        CondY02_Ypos_01_mirror = np.append(CondY02_Ypos_01_mirror,
                                           np.array([[VerticalConnectIntercon[0][0]
                                                      + r_outer_lowX * 1e3
                                                      + self.width * .5,
                                                      - via02[1]]]), axis=0)

        def shrExt(arrayS, value=100e3):
            ## shirnk, extend value
            new_array = list()
            for n in arrayS:
                if n[0] > 0.0:
                    new_array.append([n[0] + value , n[1]])
                if n[0] < 0.0:
                    new_array.append([n[0] - value , n[1]])
            return np.asarray(new_array)

        std_path_cond = np.vstack([CondY02_Ypos_02[::-1],
                                   [[xarchRight+ 100*10**3, CondY02_Ypos_02[::-1][-1][1]]],
                                   [[xarchRight+ 100*10**3, CondY01_top[0][1]]],
                                   CondY01_top[:-1],
                                   [CondY01_top[-1][0], CondY01_top[-1][1]], # CondY01_pos_left_RingCon[::-1][0][1]],
                                   [OuterInterconVia02[0][0] - 10 * 10**3, CondY01_top[-1][1]],
                                   OuterInterconVia02 - np.array([10, r_outer_lowX + .5 * self.widthArc * 1e-3,]) * 1e3,
                                   [[CondY02_Ypos_01_mirror[::-1][0][0],
                                     OuterInterconVia02[1][1] - r_outer_lowX * 1e3 - .5 * self.widthArc]],
                                   CondY02_Ypos_01_mirror[::-1],
                                   [[xarchRight+ 100e3, -1.0 * CondY02_Ypos_02[::-1][-1][1]]],
                                   [[xarchRight+ 100e3, -1.0 * CondY01_top[0][1]]],
                                   self.mirrorArrayXaxis( CondY01_top )[:-1],
                                   arrayConArchInner,
                                   [[arrayInnerVia02con[0][0] ,arrayConArchInner[-1][1]]],
                                   arrayInnerVia02con ]) * 1e-3

        # around the Zgrad vias: -> xarchRight
        add_path_cond = np.vstack([CondY02_Ypos_01[::-1],
                                   [[xarchRight + 200e3, CondY02_Ypos_02[::-1][-1][1]]],
                                   [[xarchRight + 200e3, CondY01_top[0][1]]],
                                   shrExt(CondY01_top[:-1], -140e3),
                                   [CondY01_top[-1][0], CondY01_top[-1][1]], # CondY01_pos_left_RingCon[::-1][0][1]],
                                   [OuterInterconVia02[0][0] - 80e3, CondY01_top[-1][1]],
                                   shrExt(OuterInterconVia02 - np.array([0, r_outer_lowX + .5 * self.widthArc * 1e-3 + 40]) * 1e3, value=80e3),
                                   [[CondY02_Ypos_01_mirror[::-1][0][0],
                                     OuterInterconVia02[1][1] - r_outer_lowX * 1e3 - .5 * self.widthArc - 40e3]],
                                   shrExt(CondY02_Ypos_01_mirror[::-1], -100e3),
                                   [[xarchRight + 200e3, -1.0 * CondY02_Ypos_02[::-1][-1][1]]],
                                   [[xarchRight + 200e3, -1.0 * CondY01_top[0][1]]],
                                   shrExt(self.mirrorArrayXaxis( CondY01_top )[:-1], -140e3),
                                   arrayConArchInner,
                                   [[ arrayInnerVia02con[0][0] , arrayConArchInner[-1][1]]],
                                   arrayInnerVia02con ]) * 1e-3

        shift_via = arrayInnerVia02con + np.array([150e3, 0])
        add_path_via = np.vstack([[arrayInnerVia02con[0]],
                                  [[arrayInnerVia02con[0][0], shift_via[1][1] + 150e3]],
                                  [shift_via[1]]]) * 1e-3


        self.wire_to_round_path(std_path_cond, radius=200)
        self.wire_to_round_path(add_path_cond, radius=200)
        self.wire_to_round_path(add_path_via, radius=1500)
        
        self.TargetCell.flatten(-1, True)
        Ygrad_coil = std_path_cond

        self.conductors = []

        if self.zNOT_top_real != None:
            self.conductors.append([self.wire_to_conductor(Ygrad_coil, self.zNOT_top_real),
                                    {"color":"b", "zorder":210, "current":-1.0}])
            self.conductors.append([self.wire_to_conductor(Ygrad_coil, self.zNOT_top_real,
                                                           mirror_chip=-1.0),
                                    {"color":"b", "zorder":210, "current":1.0}])
        else:
            self.conductors.append([self.wire_to_conductor(Ygrad_coil, self.z0),
                                    {"color":"b", "zorder":210, "current":-1.0}])
            self.conductors.append([self.wire_to_conductor(Ygrad_coil, self.z0, mirror_chip=-1.0),
                                    {"color":"b", "zorder":210, "current":1.0}])

        self.condPlot3D = self.conductors

def AndersonYGrad(GradCoil, TargetCellName,
                  chipType="Bot", remove=False,
                  dir_grad_output=None):

    Grad = StraightGradCoil(TargetCellName, GradCoil, chipType=chipType)

    if dir_grad_output is not None:
        Grad.dir_grad_output = dir_grad_output

    Grad.make_anderson_y_grad()
    Grad.LayerInfo = pya.LayerInfo(5, 1)
    Grad.make_anderson_y_grad()
    Grad.mergeShapes()

    if chipType == "Bot":
        Grad.export_conductor_model()

    for LayerInfo in Grad.LayerViaInfos:
        Grad.mergeShapes(LayerInfo=LayerInfo)

    placeText (Grad.TargetCell,
               Grad._ly,
               text=Grad.TargetCellName,
               mag=300,
               trans=pya.Trans(pya.Trans.M45, 3200e3, 5000e3),
               LayerInfo=pya.LayerInfo(0, 0))
    return Grad

def AndersonZGrad(GradCoil, TargetCellName,
                  chipType="Bot", remove=False,
                  dir_grad_output=None):

    Grad = StraightGradCoil(TargetCellName, GradCoil, chipType=chipType)

    if dir_grad_output is not None:
        Grad.dir_grad_output = dir_grad_output

    Grad.make_anderson_z_grad()
    Grad.LayerInfo = pya.LayerInfo(5, 1)
    Grad.make_anderson_z_grad()
    Grad.mergeShapes()

    for LayerInfo in Grad.LayerViaInfos:
        Grad.mergeShapes(LayerInfo=LayerInfo)

    if chipType == "Bot":
        Grad.export_conductor_model()

    placeText(Grad.TargetCell,
              Grad._ly,
              text=Grad.TargetCellName,
              mag=300,
              trans=pya.Trans(pya.Trans.M45, 3600e3, 5000e3),
              LayerInfo=pya.LayerInfo(0, 0))
    return Grad
