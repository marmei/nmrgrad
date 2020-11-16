
import pickle
import numpy as np

try:
    import pya
except ImportError:
    # without klayout gui
    import klayout.db as pya

from .klayout_grad import (KlayoutPyFeature,
                           arrayToPointList,
                           bspline,
                           centre_array_at_pts,
                           array_idx_dist,
                           min_dist_array_at_pts)

class KlayoutPyGradCoil(KlayoutPyFeature):
    """
    Main Class to create the coils from a conductor spline / Anderson model

    """
    TargetCellName = "klayoutPyGradCoil"

    dir_grad_coils = "GradCoils/"
    dir_grad_output = "GradOutput/"

    LayerInfo = pya.LayerInfo(0, 0)
    LayerInfoGaps = pya.LayerInfo(1, 0)
    LayerInfoGapsTop = pya.LayerInfo(2, 0)
    LayerInfoIntercon = [pya.LayerInfo(5, 0),
                         pya.LayerInfo(5, 1)]

    InterconPtsSamples = 100
    interconDistMax = 0.55e6

    widthInner = 200e3
    widthMiddle = 580e3
    widthOuter = 200e3
    widthGaps = 85e3
    widthIntercon = 400e3
    widthDebug = 100e3

    # empty lists for the gradient coils
    red_cc00 = []
    blue_cc00 = []
    red_gaps_cc00 = []
    blue_gaps_cc00 = []
    red_cc01 = []
    blue_cc01 = []
    red_gaps_cc01 = []
    blue_gaps_cc01 = []

    # linearity plot along axis ROI
    ModelLin = None
    CondLin = None

    conductors = []

    def __init__(self, TopCellName, GradCoil):
        KlayoutPyFeature.__init__(self, TopCellName)

        # read config
        if "@widthOuter" in GradCoil:
            self.widthOuter = GradCoil["@widthOuter"]
        if "@widthMiddle" in GradCoil:
            self.widthMiddle = GradCoil["@widthMiddle"]

        self.zNOT_bot = None
        self.zNOTCal_bot = None

        if "@zNOTCal_bot" in GradCoil:
            self.zNOT_bot = eval(GradCoil["@zNOTCal_bot"])
            # coil centre separation 01
            self.ccSep01 = eval(GradCoil["@zNOTCal_bot"])
            self.zNOTCal_bot = eval(GradCoil["@zNOTCal_bot"])

        if "@zNOTCal_top" in GradCoil:
            self.zNOT_top = eval(GradCoil["@zNOTCal_top"])
            # coil centre separation 00
            self.ccSep00 = eval(GradCoil["@zNOTCal_top"])
            self.zNOTCal_top = eval(GradCoil["@zNOTCal_top"])

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

        if "@cIntercon" in GradCoil:
            self.cIntercon = eval(GradCoil["@cIntercon"])

        if "@widthOuter" in GradCoil:
            self.widthOuter = GradCoil["@widthOuter"]

        if "@widthMiddle" in GradCoil:
            self.widthMiddle = GradCoil["@widthMiddle"]

        if "@dVia" in GradCoil:
            self.widthVia = eval(GradCoil["@dVia"])
            self.lenVia = eval(GradCoil["@dVia"])

        if "@trans" in GradCoil:
            self.makeCell(trans=eval(GradCoil["@trans"]))

        if "@type" in GradCoil:
            self.GradType = str(GradCoil["@type"])
        else:
            self.GradType = ""

    def makeCoilTrack(self,
                      scipy_array_list,
                      width,
                      LayerInfo=None,
                      trans=pya.Trans.new(pya.Point.new(0.0, 0.0))):
        """
        produces the gradient coil in a helper cell and flatten them.

        """
        if str(type(LayerInfo)) != "<class 'pya.LayerInfo'>":
            LayerInfo = self.LayerInfo

        if self.TargetCell is None:
            self.makeCell()

        if self._ly.has_cell("helperCellAmark"):
            raise Exception("Helper Cell does exist in makeCoilTrack.")

        helperCell = self._ly.create_cell("helperCellAmark")
        self.TargetCell.insert(
            pya.CellInstArray.new(helperCell.cell_index(), trans))

        helperCell.shapes(self._ly.layer(
                LayerInfo)).insert(pya.Path(arrayToPointList(scipy_array_list),
                                            width, .5 * width, .5 * width,
                                            True))
        self.TargetCell.flatten(-1, True)

    def mergeCoilShapes(self, LayerInfoOut=None, LayerInfo=None):
        """
        Merge the coil shapes and renders the gradient coil in klayout.

        """
        if str(type(LayerInfoOut)) != "<class 'pya.LayerInfo'>":
            LayerInfoOut = self.LayerInfo
        elif LayerInfoOut is None:
            LayerInfoOut = self.LayerInfo

        if str(type(LayerInfo)) != "<class 'pya.LayerInfo'>":
            LayerInfo = self.LayerInfo
        elif LayerInfo is None:
            LayerInfo = self.LayerInfo

        self.mergeShapes()
        self.mergeShapes(LayerInfo=self.LayerInfoGaps)
        self.shapeBoolean(self.LayerInfoGaps, LayerInfo, LayerInfoOut)
        self.removeShapeLayer(self.LayerInfoGaps)

    def InterConPts(self, cv):
        """
        cv = [[x1, y1], ..., [xn, yn]]
        cv len must be at least be 4 or % equal 2.
        returns a straight line which produces the interconnection points
        """
        if len(cv) == 2:
            cv = np.asarray(cv)
            interSecPts = np.array([
                np.linspace(cv[0][0], cv[1][0],
                            self.InterconPtsSamples),
                np.linspace(cv[0][1], cv[0][1],
                            self.InterconPtsSamples)]) * 1e3
            return np.swapaxes(interSecPts, 0, 1)
        # elif len(cv) >= 4:
        return bspline(cv, n=self.InterconPtsSamples) * 1e3

    def GapsIntoCoilList(self, gapList, ptsIntercon,
                         interconDistMax=None, invert=False,
                         skipFirst=False, skipLast=False):
        """
        Place gaps into the circular coil patterns.

        The interconneciton curve "ptsIntercon" can be any
        curve but must intersect with the gradient coil in oder to create the
        gaps.
        """
        if interconDistMax is None:
            distMaxIcon = self.interconDistMax
        else:
            distMaxIcon = interconDistMax

        listOfGaps = []
        for coilIdx in range(len(gapList)):
            coil = gapList[coilIdx]
            if coil.shape[0] >= 80 \
                    and min_dist_array_at_pts(coil, ptsIntercon) < 50e3:
                if skipFirst is True:
                    skipFirst = False
                    coilArray = array_idx_dist(
                        centre_array_at_pts(coil, ptsIntercon),
                        distMax=distMaxIcon)
                elif skipLast and coilIdx == len(gapList)-1:
                    coilArray = array_idx_dist(
                        centre_array_at_pts(coil,
                                            ptsIntercon),
                        distMax=distMaxIcon, start=True)
                else:
                    coilArray = array_idx_dist(
                        centre_array_at_pts(coil, ptsIntercon),
                        distMax=distMaxIcon)

                if invert:
                    coilArray = np.flipud(coilArray)
                listOfGaps.append(coilArray)

        return listOfGaps

    def makeSingleCurve(self, listOfGaps, ptsIntercon, shift=False,
                        cIntercon=[], startPts=False, connectEndPts=True):
        """
        interconncts the gaps curve's and returns
        a point list of the single, 'interconnected' curve.

        shiftStartEnd is the shift vector for the spline. sciyp.array([x, y])
        startPts : Start point to start with the Spline for the curve.
        If len(startPts) exceeds ...  no bspline interpolation is done.
        """
        shiftStartEnd = np.array([0.0, 0.0])
        if (shift is True) and (cIntercon != []):
            # shift the gaps by :
            shiftDist = .75 * (self.widthInner + self.widthGaps)
            xDist = (cIntercon[-1][0] - cIntercon[0][0]) * 1.0
            yDist = (cIntercon[-1][1] - cIntercon[0][1]) * 1.0

            # calculate the angle by wich to shift
            if xDist == 0.0:
                shift = np.array([shiftDist, 0.0])
            else:
                anglTan = np.tan(yDist / xDist)
                # make the sign convention for the tangents
                signX = 1.0
                signY = 1.0
                if (xDist < 0) and (yDist < 0):
                    signY = -1.0
                    signX = -1.0
                elif (xDist < 0) and (yDist > 0):
                    signY = -1.0
                    signX = -1.0
                elif (xDist < 0) and (yDist > 0):
                    signY = 1.0
                    signX = -1.0
                shiftStartEnd = np.array(
                    [np.sin(anglTan) * signX,
                     np.cos(anglTan) * signY * -1.0]) * shiftDist

        ptsInterconX = np.swapaxes(ptsIntercon, 0, 1)[0] + shiftStartEnd[0]
        ptsInterconY = np.swapaxes(ptsIntercon, 0, 1)[1] + shiftStartEnd[1]

        ptsIntercon = np.swapaxes(np.array([ptsInterconX, ptsInterconY]), 0, 1)

        ptsInterconX_end = np.swapaxes(ptsIntercon, 0, 1)[0] \
            - shiftStartEnd[0] * 2.0
        ptsInterconY_end = np.swapaxes(ptsIntercon, 0, 1)[1] \
            - shiftStartEnd[1] * 2.0

        ptsIntercon_end = np.swapaxes(np.array([
            ptsInterconX_end, ptsInterconY_end]), 0, 1)

        # interconnect the curves
        for coilIdx in range(len(listOfGaps)):
            if coilIdx == 0 and type(startPts) == bool:  # Start
                if startPts is True:
                    new_array = listOfGaps[coilIdx][3:-2]
                else:
                    spline = np.append(ptsIntercon[0:2],
                                       listOfGaps[coilIdx][:2], axis=0)
                    new_array = np.append(bspline(spline),
                                          listOfGaps[coilIdx][3:-2], axis=0)
            elif coilIdx == len(listOfGaps)-1:  # End
                spline = np.append(listOfGaps[coilIdx-1][-2:],
                                   listOfGaps[coilIdx][:2], axis=0)
                new_array = np.append(new_array, bspline(spline),
                                      axis=0)
                new_array = np.append(new_array, listOfGaps[coilIdx][3:-2],
                                      axis=0)
                if connectEndPts:
                    spline = np.append(listOfGaps[coilIdx][-2:],
                                       ptsIntercon_end[-2:], axis=0)
                    new_array = np.append(new_array, bspline(spline), axis=0)
            else:  # in between
                if type(startPts) == bool:
                    spline = np.append(listOfGaps[coilIdx-1][-2:],
                                       listOfGaps[coilIdx][:2], axis=0)
                    new_array = np.append(new_array, bspline(spline), axis=0)
                else:
                    spline = np.append(startPts, listOfGaps[coilIdx][:2],
                                       axis=0)
                    new_array = bspline(spline)
                    startPts = False
                new_array = np.append(new_array, listOfGaps[coilIdx][3:-2],
                                      axis=0)  # ref before
        return new_array

    def makeViaInterConInterCon(self):
        self.widthVia = 150e3
        self.lenVia = 150e3
        self.makeVia([1950e3, -3500e3])
        self.makeVia([2400e3, -3500e3])

    def export_conductor_model(self, addName=""):
        """
        Export the coil path and design parameters.

        """
        fname_out = self.dir_grad_output + "CModel_" \
            + self.TopCell.name + "_" \
            + self.TargetCellName + "_" + addName + self.GradType + ".p"
        pickle.dump(self.conductors, open(fname_out, "wb"))

        fname_out_params = self.dir_grad_output + "GParam_" \
            + self.TopCell.name + "_" \
            + self.TargetCellName + "_" + addName + self.GradType + ".p"
        params_out = {"GType": self.GradType,
                      "zNOT_top_real": self.zNOT_top_real,
                      "zNOTCal_top_real": self.zNOTCal_top_real,
                      "zNOT_top": self.zNOT_top,
                      "zNOTCal_top": self.zNOTCal_top,
                      "zNOT_bot_real": self.zNOT_bot_real,
                      "zNOTCal_bot_real": self.zNOTCal_bot_real,
                      "zNOT_bot": self.zNOT_bot,
                      "zNOTCal_bot": self.zNOTCal_bot,
                      "NAME": self.TopCell.name,
                      "Target": self.TargetCellName}
        pickle.dump(params_out, open(fname_out_params, "wb"))


def discretise_wire(wire, len_max=50):
    """
    Builds-up a numpy array with supplemental ponts to allow for \
    Biot Savart Integration.

    Parameters
    ----------
    wire : ndarray
        DESCRIPTION.
    len_max : int, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    ndarray
        The wire with added discretisation points.

    """
    discretised_wire = []

    # TODO: convert to 3D array discretistaion.
    for w_idx, w_elem in enumerate(wire):
        if w_idx == len(wire)-1:
            # append last point to list
            discretised_wire.append(w_elem)
            break

        len_pt_to_pt = np.sqrt((wire[w_idx+1][0] - w_elem[0])**2
                               + (wire[w_idx+1][1] - w_elem[1])**2)

        if len_pt_to_pt > len_max:
            x_step = np.linspace(w_elem[0], wire[w_idx+1][0],
                                 int(len_pt_to_pt / len_max + 1),
                                 endpoint=False)
            y_step = np.linspace(w_elem[1], wire[w_idx+1][1],
                                 int(len_pt_to_pt / len_max + 1),
                                 endpoint=False)

            for idx, _ in enumerate(x_step):
                discretised_wire.append([x_step[idx], y_step[idx]])
        else:
            discretised_wire.append(wire[w_idx])

    return np.asarray(discretised_wire)
