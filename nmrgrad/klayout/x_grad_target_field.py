
from .klayout_py_grad_coil import (KlayoutPyGradCoil,
                                   discretise_wire)
from .klayout_grad import (import_svg_polyline,
                           placeRoundPath,
                           placeText,
                           arrayToPointList,
                           KlayoutPyFeature)
try:
    import pya
except ImportError:
    # without klayout gui
    import klayout.db as pya
import numpy as np


class CCoilGapsX_XY(KlayoutPyGradCoil):

    LayerViaInfos = [pya.LayerInfo(0, 0), pya.LayerInfo(2, 0),
                     pya.LayerInfo(3, 0), pya.LayerInfo(4, 0)]

    ViaShift = np.array([0.0, 0.0]) * 1e-3

    skipPtsStart = 1
    skipPtsEnd = 1
    skipPtsGapStartRight = 1
    skipPtsGapStartLeft = 1
    skipPtsGapEnd = 1
    skipPtsGapEndLeft = 1

    extraPtCoilCoilIcon = np.array([[0, -2100e3]])
    interconDistRight = None
    interconDistLeft = None
    InterConnectPathWidth = 400
    shiftInterconMode = False

    def __init__(self, TopCellName, GradCoil):
        KlayoutPyGradCoil.__init__(self, TopCellName, GradCoil)
        self.TargetCellName = "GX" + str(self.zNOT_bot)

        if "@ViaShift" in GradCoil:
            self.ViaShift = eval(GradCoil["@ViaShift"])
        if "@InterConnectPathWidth" in GradCoil:
            self.InterConnectPathWidth = eval(
                GradCoil["@InterConnectPathWidth"])
        if "@shiftInterconMode" in GradCoil:
            self.shiftInterconMode = eval(GradCoil["@shiftInterconMode"])

        if "@zNOTCal_top_real" in GradCoil:
            self.zNOT_top_real = eval(GradCoil["@zNOTCal_top_real"])
            self.zNOTCal_top_real = eval(GradCoil["@zNOTCal_top_real"])
        else:
            self.zNOT_top_real = None
            self.zNOTCal_top_real = None
        if "@zNOTCal_bot_real" in GradCoil:
            self.zNOT_bot_real = eval(GradCoil["@zNOTCal_bot_real"])
            self.zNOTCal_bot_real = eval(GradCoil["@zNOTCal_bot_real"])
        else:
            self.zNOT_bot_real = None
            self.zNOTCal_bot_real = None

        if "@skipPtsStart" in GradCoil:
            self.skipPtsStart = eval(GradCoil["@skipPtsStart"])
        if "@skipPtsEnd" in GradCoil:
            self.skipPtsEnd = eval(GradCoil["@skipPtsEnd"])
        if "@skipPtsGapStartRight" in GradCoil:
            self.skipPtsGapStartRight = eval(GradCoil["@skipPtsGapStartRight"])
        if "@skipPtsGapStartLeft" in GradCoil:
            self.skipPtsGapStartLeft = eval(GradCoil["@skipPtsGapStartLeft"])
        if "@skipPtsGapEnd" in GradCoil:
            self.skipPtsGapEnd = eval(GradCoil["@skipPtsGapEnd"])
        if "@skipPtsGapEndLeft" in GradCoil:
            self.skipPtsGapEndLeft = eval(GradCoil["@skipPtsGapEnd"])
        if "@extraPtCoilCoilIcon" in GradCoil:
            self.extraPtCoilCoilIcon = eval(GradCoil["@extraPtCoilCoilIcon"])
        if "@interconDistRight" in GradCoil:
            self.interconDistRight = eval(GradCoil["@interconDistRight"])
        if "@interconDistLeft" in GradCoil:
            self.interconDistLeft = eval(GradCoil["@interconDistLeft"])
        if "@shiftGapsRedBlue" in GradCoil:
            if eval(GradCoil["@shiftGapsRedBlue"]):
                self.shiftGapsRedBlue = True
            else:
                self.shiftGapsRedBlue = False
        else:
            self.shiftGapsRedBlue = False

    def read_grad_streamlines(self, GradCoil):
        # read the curves from the SVG Plots
        self.red_cc01, self.blue_cc01 = import_svg_polyline(
            self.dir_grad_coils + GradCoil["@fileCoil"])
        self.red_gaps_cc01, self.blue_gaps_cc01 = import_svg_polyline(
            self.dir_grad_coils + GradCoil["@fileGaps"])

    @staticmethod
    def KlayoutToBfield(cond, zNot):
        """
        return mapped array for/from Klayout to Bfield calculatin
        with given zNot (e.g. xNot separation)
        cond: np.array([[x1, y1], ... [xn, yn]])
        """
        return np.asarray([cond[:, 0] * 1e-6,
                           cond[:, 1] * 1e-6,
                           np.zeros_like(cond[:, 0])
                           + zNot * 1e-3]).swapaxes(0, 1)

    def simuGradModel(self):
        """
        Plots the Field for the gradient coils from the streamlines.

        This method depends on KlayoutToBfield which is created in the subclass
        to map the 3D array.
        """
        self.conductors = []
        for cond in self.blue_cc01:
            self.conductors.append([self.KlayoutToBfield(cond, -1.0
                                                         * self.ccSep01),
                                    {"color": "b", "zorder": 210,
                                     "current": -1.0}])
            self.conductors.append([self.KlayoutToBfield(cond, self.ccSep01),
                                    {"color": "r", "zorder": 190,
                                     "current": -1.0}])
        for cond in self.red_cc01:
            self.conductors.append([self.KlayoutToBfield(
                cond,
                -1.0 * self.ccSep01),
                {"color": "b", "zorder": 210, "current": -1.0}])
            self.conductors.append([self.KlayoutToBfield(cond, self.ccSep01),
                                    {"color": "r", "zorder": 190,
                                     "current": -1.0}])
        self.condPlot3D = self.conductors  # for simulation

    def makeViaStartEnd(self):
        """
        create Vias at the end positions of the coil
        """
        self.makeVia(self.SingleCoilArray[0] + self.ViaShift * 1e3,
                     vertical=True, trans=pya.Trans(pya.Trans.R0, 0.0, 0.0))
        self.makeVia(self.SingleCoilArray[-1] + self.ViaShift * 1e3,
                     vertical=True, trans=pya.Trans(pya.Trans.R0, 0.0, 0.0))

    def makeInterconnect(self, InterConnectBaseArr,
                         direction="Right", LayerInfo=None):
        InterConnect = np.zeros_like(InterConnectBaseArr)

        # Overlay the Interconnection to the base array
        if direction == "Right":
            InterConnect[0] = np.array([self.SingleCoilArray[0] * 1e-3
                                        + self.ViaShift])
            InterConnect[1] = np.array([self.SingleCoilArray[0][0]
                                        * 1e-3, 0.0])
            InterConnect = InterConnect + InterConnectBaseArr
        elif direction == "Left":
            InterConnect[0] = np.array([self.SingleCoilArray[-1] * 1e-3
                                        + self.ViaShift])
            InterConnect[1] = np.array([self.SingleCoilArray[-1][0] * 1e-3,
                                        0.0])
            InterConnect = InterConnect + InterConnectBaseArr * [-1.0, 1.0]

        # make the right interconnection Path
        Intecon = KlayoutPyFeature(self.TargetCellName)
        Intecon.TargetCellName = "intercon_dummy" + str(np.random.rand())
        Intecon.removeCell()
        Intecon.makeCell()
        if str(type(LayerInfo)) != "<class 'pya.LayerInfo'>":
            LayerInfo = self.LayerInfoIntercon[0]

        placeRoundPath(Intecon.TargetCell,
                       Intecon._ly,
                       radius=400,
                       width=self.InterConnectPathWidth,
                       pts=InterConnect,
                       LayerInfo=LayerInfo,
                       RadPath=self.widthVia * 1e-3)

        InterConnect = discretise_wire(InterConnect)

        if self.zNOTCal_top_real != None:
            return np.array([InterConnect.T[0],  # x
                             InterConnect.T[1],  # y
                             np.zeros_like(InterConnect.T[0]) + self.zNOT_top_real]) * 1e-3
        else:
            return np.array([InterConnect.T[0],  # x
                             InterConnect.T[1],  # y
                             np.zeros_like(InterConnect.T[0])
                             + self.zNOT_top]) * 1e-3

    def makeGradLabel(self):
        """ make the gradient coil label """
        placeText(self.TargetCell,
                  self._ly,
                  text=self.TargetCellName,
                  mag=300,
                  trans=pya.Trans(pya.Trans.M45, -3800e3, 5000e3),
                  LayerInfo=self.LayerInfo)

    def conductorModelBcal(self, InterConnectRightComp,
                           InterConnectLeftComp, addName="RR"):
        """ creat internal conductor models for Bfield calculation """
        self.condPlot3D = []
        self.conductors = []

        xPos_bot = self.SingleCoilArray[:, 0] * 1e-6
        xPos_top = np.flipud(self.SingleCoilArray)[:, 0] * 1e-6

        yPos_bot = self.SingleCoilArray[:, 1] * 1e-6
        yPos_top = np.flipud(self.SingleCoilArray)[:, 1] * -1e-6

        if self.zNOTCal_bot_real is not None:
            zPos_top = np.zeros_like(
                xPos_top) + (self.zNOT_bot_real * 1e-3)  # zNot is in um
            zPos_bot = np.zeros_like(xPos_bot) - (self.zNOT_bot_real * 1e-3)
        else:
            zPos_top = np.zeros_like(xPos_top) + \
                (self.zNOT_bot * 1e-3)  # zNot is in um
            zPos_bot = np.zeros_like(xPos_bot) - (self.zNOT_bot * 1e-3)

        InterconLeftTop = discretise_wire(
                np.array([InterConnectLeftComp[0][::-1],
                          InterConnectLeftComp[1][::-1] * -1.0,
                          InterConnectLeftComp[2][::-1]]))

        InterconRightTop = discretise_wire(
                np.array([InterConnectRightComp[0],
                          InterConnectRightComp[1] * -1.0,
                          InterConnectRightComp[2]]))

        TopChipConductorSet = np.asarray([xPos_top, yPos_top, zPos_top])
        self.conductors.append([np.hstack([InterconLeftTop,
                                           TopChipConductorSet,
                                           InterconRightTop]).swapaxes(0, 1),
                                {"current": -1.0}])

        BotChipConductorSet = np.asarray([xPos_bot, yPos_bot, zPos_bot])
        InterconLeftBot = np.array([InterConnectLeftComp[0],
                                    InterConnectLeftComp[1],
                                    InterConnectLeftComp[2] * -1.0])

        InterconRightBot = np.array([InterConnectRightComp[0][::-1],
                                     InterConnectRightComp[1][::-1],
                                     InterConnectRightComp[2][::-1] * -1.0])

        self.conductors.append([np.hstack([InterconRightBot,
                                           BotChipConductorSet,
                                           InterconLeftBot]).swapaxes(0, 1),
                                {"current": -1.0}])

        def helperSS(arrA, appendStart=None, appendEnd=None):
            arrayIn = [arrA[0], arrA[1], arrA[2]]
            if appendStart is None and appendEnd is None:
                return np.array(arrayIn).swapaxes(0, 1)
            elif appendEnd is None:
                return np.append(arrayIn,
                                 np.array([appendStart[::, 0]]).T,
                                 axis=1).swapaxes(0, 1)
            elif appendStart is None:
                return np.append(arrayIn,
                                 np.array([appendEnd[::, -1]]).T,
                                 axis=1).swapaxes(0, 1)

        self.condPlot3D.append([helperSS(BotChipConductorSet),
                                {"color": "orangered", "zorder": 291}])
        self.condPlot3D.append([helperSS(InterconRightBot,
                                         BotChipConductorSet),
                                {"color": "forestgreen", "zorder": 291}])

        self.condPlot3D.append([helperSS(np.fliplr(InterconLeftBot),
                                         appendStart=np.fliplr(
                                                 BotChipConductorSet)),
                                {"color": "forestgreen", "zorder": 291}])

        self.condPlot3D.append([helperSS(InterconLeftTop,
                                         appendStart=TopChipConductorSet),
                                {"color": "g", "zorder": 390}])

        self.condPlot3D.append([helperSS(np.fliplr(InterconRightTop),
                                         appendStart=np.fliplr(
                                                 TopChipConductorSet)),
                                {"color": "g", "zorder": 390}])

        self.condPlot3D.append([helperSS(TopChipConductorSet), {
                               "color": "b", "zorder": 390}])


def makeCoilGapsX_XY(GradCoil, TargetCellName,
                     dir_grad_output=None, dir_grad_coils=None,
                     chipType="Bot", remove=False, debug=False):
    """
    make a single Xgradient coil and interconnects
    """
    Grad = CCoilGapsX_XY(TargetCellName, GradCoil)

    if dir_grad_coils is not None:
        Grad.dir_grad_coils = dir_grad_coils

    if dir_grad_output is not None:
        Grad.dir_grad_output = dir_grad_output

    if remove is True:
        Grad.removeCell()  # clear the grad cell before

    Grad.read_grad_streamlines(GradCoil)

    if Grad.shiftGapsRedBlue is False:
        Grad.red_gaps_cc01 = []
        for gaps in Grad.blue_gaps_cc01:
            Grad.red_gaps_cc01.append(gaps * np.array([-1.0, 1.0]))
    else:
        # create the red gaps from the blue!
        Grad.blue_gaps_cc01 = []
        for gaps in Grad.red_gaps_cc01:
            Grad.blue_gaps_cc01.append(gaps * np.array([-1.0, 1.0]))

    Grad.makeCell()

    ptsIntercon = Grad.InterConPts(Grad.cIntercon)

    # only select the blue coil/gaps
    listOfCoilsRight = Grad.GapsIntoCoilList(
            Grad.blue_cc01, ptsIntercon,
            interconDistMax=Grad.interconDistRight)
    listOfGapsRight = Grad.GapsIntoCoilList(
            Grad.blue_gaps_cc01, ptsIntercon,
            interconDistMax=Grad.interconDistRight)

    listOfCoilsLeft = Grad.GapsIntoCoilList(
            Grad.red_cc01, ptsIntercon,
            interconDistMax=Grad.interconDistLeft)[::-1]
    listOfGapsLeft = Grad.GapsIntoCoilList(
            Grad.red_gaps_cc01, ptsIntercon,
            interconDistMax=Grad.interconDistLeft,
            invert=True, skipLast=True)

    # make the gradient curve
    SingleCoilArrayRight = Grad.makeSingleCurve(
            listOfCoilsRight, ptsIntercon,
            connectEndPts=False)[Grad.skipPtsStart::]
    ptsStartLeftCoil = np.append(SingleCoilArrayRight[-3:-1:],
                                 Grad.extraPtCoilCoilIcon, axis=0)
    SingleCoilArrayLeft = Grad.makeSingleCurve(
            listOfCoilsLeft,
            ptsIntercon,
            startPts=ptsStartLeftCoil)[:-1 * Grad.skipPtsStart]

    SingleArrayGapsRight = Grad.makeSingleCurve(
            listOfGapsRight,
            ptsIntercon,
            connectEndPts=False)[Grad.skipPtsGapStartRight::]

    SingleArrayGapsLeft = Grad.makeSingleCurve(
            listOfGapsLeft,
            ptsIntercon,
            connectEndPts=False,
            startPts=True)[Grad.skipPtsGapEndLeft:-Grad.skipPtsGapEndLeft]

    # add extra gaps for the left-right coil separation
    if "@extraGapCoilCoil" in GradCoil:
        extraGapMid = eval(GradCoil["@extraGapCoilCoil"])
        Grad.TargetCell.shapes(Grad._ly.layer(Grad.LayerInfoGaps)).insert(
            pya.Path(arrayToPointList(extraGapMid), Grad.widthGaps))

    # correct the first winding of the left coil
    Grad.skipLeftCorrect = -25
    SingleArrayGapsLeft = np.append(
            SingleArrayGapsLeft[0:Grad.skipLeftCorrect],
            np.array([SingleArrayGapsLeft[Grad.skipLeftCorrect]
                      - np.array([0.0, 1000e3])]), axis=0)

    Grad.SingleCoilArray = SingleCoilArrayRight[:-2]
    Grad.SingleCoilArray = np.append(
            Grad.SingleCoilArray, SingleCoilArrayLeft, axis=0)

    # append the via positions
    Grad.SingleCoilArray = np.append(
            np.array([eval(GradCoil["@posViaRight"])]) * 1e3,
            Grad.SingleCoilArray, axis=0)
    Grad.SingleCoilArray = np.append(
            Grad.SingleCoilArray,
            np.array([eval(GradCoil["@posViaLeft"])]) * 1e3, axis=0)

    Grad.makeCoilTrack(Grad.SingleCoilArray, width=Grad.widthInner)

    Grad.makeCoilTrack(SingleArrayGapsRight, width=Grad.widthGaps,
                       LayerInfo=Grad.LayerInfoGaps)

    Grad.makeCoilTrack(SingleArrayGapsLeft, width=Grad.widthGaps,
                       LayerInfo=Grad.LayerInfoGaps)

    Grad.mergeCoilShapes()
    Grad.makeViaStartEnd()

    if Grad.shiftInterconMode is True:
        LayerInfo01 = Grad.LayerInfoIntercon[1]
        LayerInfo02 = Grad.LayerInfoIntercon[0]
    else:
        LayerInfo01 = Grad.LayerInfoIntercon[0]
        LayerInfo02 = Grad.LayerInfoIntercon[1]

    # 1st interconnect (around the fluidic reservour)
    InterConnect = np.array([[0.0, 0.0],
                             [0.0, -1800.0],
                             [-3550.00, -1800],
                             [-3550, -2900], [-3000, -3560],
                             [-1950, -3560]])

    InterConnectRightComp = Grad.makeInterconnect(InterConnect,
                                                  LayerInfo=LayerInfo01)

    InterConnectLeftComp = Grad.makeInterconnect(InterConnect,
                                                 direction="Left",
                                                 LayerInfo=LayerInfo01)

    # 2nd interconnect (around the fluidic reservour)
    InterConnect = np.array([[0.0, 0.0],
                             [0.0, -1800.0],
                             [-900.0, -2300.0],
                             [-900.0, -2900.0],
                             [-900, -3560],
                             [-2450, -3560]])

    InterConnectRightComp = Grad.makeInterconnect(InterConnect,
                                                  LayerInfo=LayerInfo02)

    InterConnectLeftComp = Grad.makeInterconnect(InterConnect,
                                                 direction="Left",
                                                 LayerInfo=LayerInfo02)

    # calculate the field
    if chipType == "Bot":
        Grad.simuGradModel()
        Grad.export_conductor_model(addName="_discrete_")
        Grad.conductorModelBcal(InterConnectRightComp, InterConnectLeftComp)
        Grad.export_conductor_model()

    # Generate Labels for the gradient coil
    Grad.makeGradLabel()
    return Grad
