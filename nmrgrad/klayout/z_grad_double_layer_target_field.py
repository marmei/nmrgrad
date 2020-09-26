
from .klayout_py_grad_coil import (KlayoutPyGradCoil,
                                   discretise_wire)
from .klayout_grad import (import_svg_polyline, placeRoundPath, placeText,
                           arrayToPointList, KlayoutPyFeature)

import pya
import numpy as np

class CCoilGapsZ_YZ(KlayoutPyGradCoil):
    interconDistBot = None
    interconDistTop = None
    shiftEndIcon = -12 # remove these points from the interconnect at the end

    def __init__(self, TopCellName, GradCoil):
        KlayoutPyGradCoil.__init__(self, TopCellName, GradCoil)
        self.TargetCellName  = "GZ{}_{}_red".format(self.zNOT_bot, self.zNOT_top)

        if "@interconDistBot" in GradCoil:
            self.interconDistBot = eval(GradCoil["@interconDistBot"])
        if "@interconDistTop" in GradCoil:
            self.interconDistTop = eval(GradCoil["@interconDistTop"])

        if "@zNOTCal_top_real" in GradCoil:
            self.zNOT_top = eval(GradCoil["@zNOTCal_top_real"])
        if "@zNOTCal_bot_real" in GradCoil:
            self.zNOT_bot = eval(GradCoil["@zNOTCal_bot_real"])

    def read_grad_streamlines(self, GradCoil):
        self.blue_cc01, self.red_cc01 = import_svg_polyline(self.dir_grad_coils
                                                            + GradCoil["@fileCoil_bot"])
        self.blue_gaps_cc01, self.red_gaps_cc01 = import_svg_polyline(self.dir_grad_coils
                                                                      + GradCoil["@fileGaps_bot"])
        self.blue_cc00, self.red_cc00 = import_svg_polyline(self.dir_grad_coils
                                                            + GradCoil["@fileCoil_top"])
        self.blue_gaps_cc00, self.red_gaps_cc00 = import_svg_polyline(self.dir_grad_coils
                                                                      + GradCoil["@fileGaps_top"])

    @staticmethod
    def KlayoutToBfield(cond, z_not):
        """
        return mapped array for/from Klayout to Bfield calculatin
        with given zNot (e.g. xNot separation)
        cond: np.array([[x1, y1], ... [xn, yn]])
        """
        return np.asarray([np.zeros_like(cond[:, 0]) + z_not * 1e-3,
                           cond[:, 0] * 1e-6,
                           cond[:, 1] * 1e-6]).swapaxes(0, 1)

    def simuGradModel(self):
        """
        Plots the Field for the gradient coils Peter created
        This method depends on KlayoutToBfield which is created in the
        subclass to map the 3D array.

        color and zorder is for mathplotlib
        """
        self.conductors = []
        for cond in self.blue_cc01:
            self.conductors.append([self.KlayoutToBfield(cond, -1.0 * self.ccSep01),
                                    {"color":"b", "zorder":210}])
            self.conductors.append([self.KlayoutToBfield(cond, self.ccSep01),
                                    {"color":"r", "zorder":190}])
        for cond in self.blue_cc00:
            self.conductors.append([self.KlayoutToBfield(cond, -1.0 * self.ccSep00),
                                    {"color":"b", "zorder":200}])
            self.conductors.append([self.KlayoutToBfield(cond, self.ccSep00),
                                    {"color":"r", "zorder":195} ])
        for cond in self.red_cc01:
            self.conductors.append([self.KlayoutToBfield(cond, -1.0 * self.ccSep01),
                                    {"color":"b", "zorder":210}])
            self.conductors.append([self.KlayoutToBfield(cond, self.ccSep01),
                                    {"color":"r", "zorder":190, "current":-1.0}])
        for cond in self.red_cc00:
            self.conductors.append([self.KlayoutToBfield(cond, -1.0 * self.ccSep00),
                                    {"color":"b", "zorder":200}])
            self.conductors.append([self.KlayoutToBfield(cond, self.ccSep00),
                                    {"color":"r", "zorder":195, "current":-1.0}])
        self.condPlot3D = self.conductors # for simulation

    def makeInterconnect(self, InterConnectBaseArr, LayerInfo=None):
        ## make the Interconnection for the Bot coil
        inter_con_top = np.zeros_like(InterConnectBaseArr)

        if str(type(LayerInfo)) != "<class 'pya.LayerInfo'>":
            LayerInfo = self.LayerInfoIntercon[0]

        # Make the interconnection wire
        inter_con_top[0] = np.array([self.SingleArray_cc01[-1] * 1e-3])
        inter_con_top[1] = np.array([self.SingleArray_cc01[-1][0] * 1e-3, 0.0])
        inter_con_top[2] = np.array([self.SingleArray_cc01[-1][0] * 1e-3, 0.0])
        inter_con_top[3] = np.array([self.SingleArray_cc01[-1][0] * 1e-3, 0.0])
        InterConnectBot = inter_con_top + InterConnectBaseArr
        inter_con_top = inter_con_top + InterConnectBaseArr * np.array([1.0, -1.0])

        self.RoundPathF(inter_con_top, 300, 400, LayerInfo)
        self.RoundPathF(InterConnectBot[0:3], 300, 400)
        self.RoundPathF(InterConnectBot[0:2], 300, 400 + 60, self.LayerSU8TopLitho)
        self.RoundPathF(InterConnectBot[1:], 300, 400, LayerInfo)

        # make the via Interconnection - use the list of LayerViaInfos
        self.RoundPathF(InterConnectBot[1:3], 500, 400, self.LayerViaInfos)

        InterConnectBot01 = discretise_wire(InterConnectBot[:][0:3]).T
        self.inter_con_bot01 = np.asarray([np.zeros_like(InterConnectBot01[0])
                                           + self.zNOT_bot * 1e-3,
                                           np.flipud(InterConnectBot01[0]) * 1e-3,
                                           np.flipud(InterConnectBot01[1]) * 1e-3])

        InterConnectBot02 = discretise_wire(InterConnectBot[:][2:]).T
        self.inter_con_bot02 = np.asarray([np.zeros_like(InterConnectBot02[0])
                                           + self.zNOT_top * 1e-3,
                                           np.flipud(InterConnectBot02[0]) * 1e-3,
                                           np.flipud(InterConnectBot02[1]) * 1e-3])

        inter_con_top = discretise_wire(inter_con_top)

        self.inter_con_top = np.asarray([np.zeros_like(inter_con_top[:, 0])
                                         + self.zNOT_top * 1e-3,
                                         inter_con_top[:, 0] * 1e-3,
                                         inter_con_top[:, 1] * 1e-3])

    def conductorModelBcal(self):
        self.condPlot3D = []
        self.conductors = []

        ## first side of self.conductors
        xPos_top = np.flipud(self.SingleArray_top)[:, 0] * 1e-6
        yPos_top = np.flipud(self.SingleArray_top)[:, 1] * 1e-6
        zPos_top = np.zeros_like(xPos_top) + (self.zNOT_top * 1e-3)

        xPos_cc01 = self.SingleArray_cc01[:, 0] * 1e-6
        yPos_cc01 = self.SingleArray_cc01[:, 1] * 1e-6
        zPos_cc01 = np.zeros_like(xPos_cc01) + (self.zNOT_bot * 1e-3) # zNot is in um

        InterconnectStart01_zNotPos = self.inter_con_bot01
        InterconnectStart02_zNotPos = self.inter_con_bot02
        InterconnectEnd_zNotPos = self.inter_con_top

        BotChipConductorSet = np.asarray([np.flipud(np.append(zPos_top, zPos_cc01)),
                                          np.flipud(np.append(xPos_top, xPos_cc01)),
                                          np.flipud(np.append(yPos_top, yPos_cc01))])

        self.conductors.append([np.hstack([InterconnectStart02_zNotPos,
                                           InterconnectStart01_zNotPos,
                                           BotChipConductorSet,
                                           InterconnectEnd_zNotPos]).swapaxes(0, 1),
                                {"current": 1.0}])

        self.condPlot3D.append([InterconnectStart02_zNotPos.swapaxes(0, 1),
                                {"color":"g", "alpha": 1.0, "zorder": 90}])

        self.condPlot3D.append([InterconnectStart01_zNotPos.swapaxes(0,1),
                                {"color":"g", "alpha": 1.0, "zorder": 90}])

        self.condPlot3D.append([BotChipConductorSet.swapaxes(0,1)[0:len(xPos_cc01)+1],
                                {"color":"orangered", "alpha": 0.7, "zorder": 90}])

        self.condPlot3D.append([BotChipConductorSet.swapaxes(0,1)[len(xPos_cc01)+1:-1],
                                {"color":"r", "alpha":1.0, "zorder": 90}])

        self.condPlot3D.append([InterconnectEnd_zNotPos.swapaxes(0,1),
                                {"color":"g", "alpha": 1.0, "zorder": 90}])

        ## now we shift the conductors to the Plot Coordinate System
        InterconnectEnd_zNotPos_Top = (np.flipud(InterconnectEnd_zNotPos).T
                                       * np.array([1.0, -1.0, -1.0])).T
        TopChipConductorSet = (np.flipud(BotChipConductorSet).T
                               * np.array([1.0, -1.0, -1.0 ])).T
        InterconnectStart01_zNotPos_Top = (np.flipud(InterconnectStart01_zNotPos).T
                                           * np.array([1.0, -1.0, -1.0])).T
        InterconnectStart02_zNotPos_Top = (np.flipud(InterconnectStart02_zNotPos).T
                                           * np.array([1.0, -1.0, -1.0])).T

        TopChipComplementSet = np.hstack([InterconnectStart02_zNotPos_Top,
                                          InterconnectStart01_zNotPos_Top,
                                          TopChipConductorSet,
                                          InterconnectEnd_zNotPos_Top])

        def helperSS(arrA):
            return np.array([arrA[2],arrA[1],arrA[0]]).swapaxes(0,1)

        self.conductors.append([helperSS(TopChipComplementSet), {"current": 1.0}])

        self.condPlot3D.append([helperSS(InterconnectEnd_zNotPos_Top)[::-1],
                                {"color":"g", "alpha":1.0, "zorder":190}])

        self.condPlot3D.append([helperSS(TopChipConductorSet)[len(xPos_cc01)+1:-1][::-1],
                                {"color":"slateblue", "alpha":1.0, "zorder":199}])

        self.condPlot3D.append([helperSS(TopChipConductorSet)[0:len(xPos_cc01)+1][::-1],
                                {"color":"mediumblue", "alpha":1.0, "zorder":200}])

        self.condPlot3D.append([helperSS(InterconnectStart01_zNotPos_Top)[::-1],
                                {"color":"g", "alpha":1.0, "zorder":190}])

        self.condPlot3D.append([helperSS(InterconnectStart02_zNotPos_Top)[::-1],
                                {"color":"g", "alpha":1.0, "zorder":190}])

    def makeGradLabel(self):
        """ make the gradient coil label """
        placeText(self.TargetCell,
                  self._ly,
                  text="Gz" + str(self.zNOT_bot),
                  mag=600,
                  trans = pya.Trans(pya.Trans.M45, -6800e3, -500e3),
                  LayerInfo=pya.LayerInfo(0, 0))

        placeText(self.TargetCell,
                  self._ly,
                  text="Gz" + str(self.zNOT_top),
                  mag=600,
                  trans=pya.Trans(pya.Trans.M45, -5800e3, -500e3),
                  LayerInfo=pya.LayerInfo(0, 0))

def makeCoilGapsZ_YZ(GradCoil, TargetCellName,
                     dir_grad_output=None, dir_grad_coils=None,
                     chipType="Bot", remove=False):
    """ make the Zgradient coil and interconnects """

    Grad = CCoilGapsZ_YZ(TargetCellName, GradCoil)

    if dir_grad_coils is not None:
        Grad.dir_grad_coils = dir_grad_coils

    if dir_grad_output is not None:
        Grad.dir_grad_output = dir_grad_output

    if remove is True:
        Grad.removeCell() # clear the grad coil cell

    Grad.read_grad_streamlines(GradCoil)

    Grad.makeCell()

    ptsIntercon = Grad.InterConPts(Grad.cIntercon)

    # only select the blue coil/gaps
    listOfCoils_cc01 = Grad.GapsIntoCoilList(Grad.red_cc01,
                                             ptsIntercon[:Grad.shiftEndIcon],
                                             interconDistMax=Grad.interconDistBot)
    listOfCoils_gaps_cc01 = Grad.GapsIntoCoilList(Grad.red_gaps_cc01,
                                                  ptsIntercon[:Grad.shiftEndIcon],
                                                  interconDistMax=Grad.interconDistBot)

    # make the coil curves
    SingleArray_cc01 = Grad.makeSingleCurve(listOfCoils_cc01, ptsIntercon[:Grad.shiftEndIcon])
    Grad.SingleArray_cc01 = np.append(SingleArray_cc01, ptsIntercon[Grad.shiftEndIcon:], axis=0)

    listOfCoils_top = Grad.GapsIntoCoilList(Grad.red_cc00,
                                            ptsIntercon[:Grad.shiftEndIcon],
                                            interconDistMax=Grad.interconDistTop)
    listOfCoils_gaps_top = Grad.GapsIntoCoilList(Grad.red_gaps_cc00,
                                                 ptsIntercon[:Grad.shiftEndIcon],
                                                 interconDistMax=Grad.interconDistTop)

    # make the coil curves
    SingleArray_top = Grad.makeSingleCurve(listOfCoils_top,
                                           ptsIntercon[:Grad.shiftEndIcon])
    SingleArray_top = np.append(SingleArray_top, ptsIntercon[Grad.shiftEndIcon:],
                                axis=0)

    # make the gaps curves
    SingleArray_gaps_cc01 = Grad.makeSingleCurve(listOfCoils_gaps_cc01,
                                                 ptsIntercon, shift=True,
                                                 cIntercon=Grad.cIntercon)
    SingleArray_gaps_top = Grad.makeSingleCurve(listOfCoils_gaps_top,
                                                ptsIntercon, shift=True,
                                                cIntercon=Grad.cIntercon)

    # mirror the top coil and gaps
    Grad.SingleArray_top = SingleArray_top * np.array([1.0, -1.0])
    SingleArray_gaps_top = SingleArray_gaps_top * np.array([1.0, -1.0])

    # make the gradient paths in Klayout
    Grad.makeCoilTrack(Grad.SingleArray_cc01, width=Grad.widthInner)
    Grad.makeCoilTrack(Grad.SingleArray_top, width=Grad.widthInner,
                       LayerInfo=Grad.LayerInfoIntercon[0])
    Grad.makeCoilTrack(Grad.SingleArray_top, width=Grad.widthInner,
                       LayerInfo=Grad.LayerInfoIntercon[1])

    # make the shifted gaps
    Grad.makeCoilTrack(SingleArray_gaps_cc01, width=Grad.widthGaps,
                       LayerInfo=Grad.LayerInfoGaps)

    # merge the coil shapes of bot layer
    Grad.mergeCoilShapes()

    # make top layer gaps and merge top layer
    Grad.makeCoilTrack(SingleArray_gaps_top, width=Grad.widthGaps,
                       LayerInfo=Grad.LayerInfoGaps)
    Grad.makeCoilTrack(SingleArray_gaps_top, width=Grad.widthGaps,
                       LayerInfo=Grad.LayerInfoGapsTop)
    Grad.mergeCoilShapes(LayerInfoOut=Grad.LayerInfoIntercon[0],
                         LayerInfo=Grad.LayerInfoIntercon[0])

    Grad.LayerInfoGaps = Grad.LayerInfoGapsTop
    Grad.mergeCoilShapes(LayerInfoOut=Grad.LayerInfoIntercon[1],
                         LayerInfo=Grad.LayerInfoIntercon[1])


    Grad.makeVia(ptsIntercon[0], vertical=True, trans=pya.Trans(pya.Trans.R0, 0.0, 0.0))

    InterConnect = np.array([[0, -100],
                             [0, -1100],
                             [0, -1400],
                             [0, -3000],
                             [-3000, -3560],
                             [-2000, -3560]]) * np.array([-1.0, 1.0])

    Grad.makeInterconnect(InterConnect)

    InterConnect = np.array([[0, -100],
                             [0, -1100],
                             [0, -1400],
                             [0, -1700],
                             [-1850, -2100],
                             [-1150, -2400],
                             [-1150, -3560],
                             [-2450, -3560]]) * np.array([-1.0, 1.0])

    Grad.makeInterconnect(InterConnect, LayerInfo=Grad.LayerInfoIntercon[1])

    if chipType == "Bot":
        Grad.simuGradModel()
        Grad.export_conductor_model(addName="_discrete_")
        Grad.conductorModelBcal()
        Grad.export_conductor_model()

    return Grad
