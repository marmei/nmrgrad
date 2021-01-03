
from .klayout_grad import (import_svg_polyline,
                           placeRoundPath,
                           placeText,
                           arrayToPointList,
                           pathPolyRound,
                           KlayoutPyFeature)
import numpy as np

try:
    import pya
except ImportError:
    # without klayout gui
    import klayout.db as pya


class FluidicChannels(KlayoutPyFeature):
    TopCellName = "Fluidics_Quarter"

    # positions of the reservour
    x1, y1 = 2700.00, 2700.00  # in um
    x2, y2 = 0.00, 4000.00  # in um

    LayerInfo = pya.LayerInfo(3, 0)  # base Layer Where to make the Channels
    LayerInfoSepMerge = pya.LayerInfo(0, 0)
    LayerInfoFCroute = []
    nMainCircPoints = 100
    widthMain = 290 * 2e3

    # width of the spacers for the main fluidic connector
    widthSpacer = 240e3

    # Fludic connectors
    nInlets = 19
    widthChannel = 140e3
    widthChannelSpacer = 180e3
    lenFconnector = 250e3
    round_inlet = 200e3
    round_outlet = 200e3

    # fluidic router, routing area
    channelLeft = -2500e3
    channelRight = 2500e3
    channelTop = 2000e3
    channelBottom = -2000e3
    resistMinWidth = 100e3 * 2

    outlets = None
    skipOutlet = []

    channelYknick = 800.12e3
    bboxViaSepList = []
    viaExtraSep = 50e3
    splitChannel = False
    SkipPrimerBool = False
    SpacerInvScale = 2

    splitIdxStart = 4
    splitIdxEnd = -5

    # defines the value of which to scale the cooling channel shapes
    baseSizeShapes = 25e3

    # internal storrage of gradient coils with intersecting vias
    gradientCoils = []

    # The Layers where to draw the fluidic system
    LSkipPrimer = []
    LFchannels = []
    LEncapsulation = []

    def __init__(self, TopCellName, cooling, gradientCoils=[]):
        self.TargetCellName = TopCellName + "_" + cooling["@cellName"]
        KlayoutPyFeature.__init__(self, TopCellName)

        if "@skipViaRoute" in cooling:
            self.gradientCoils = []
        else:
            self.gradientCoils = gradientCoils

        if "@resistMinWidth" in cooling:
            self.resistMinWidth = eval(cooling["@resistMinWidth"])
        if "@cSpacer" in cooling:
            self.widthChannelSpacer = eval(cooling["@cSpacer"])
        if "@skipOutlet" in cooling:
            self.skipOutlet = eval(cooling["@skipOutlet"])
        if "@nInlets" in cooling:
            self.nInlets = eval(cooling["@nInlets"])
        if "@cWidth" in cooling:
            self.addSpacerScale = eval(cooling["@cWidth"])
        if cooling["@type"] == "middle":
            self.LBottomChannel = [2]
            self.LFchannels = [3]
            self.LEncapsulation = [4, 6, 7, 8]
            if "@cWidth" in cooling:
                self.widthChannel = eval(cooling["@cWidth"]) \
                    - ((self.LFchannels[0] - 1) * self.baseSizeShapes)
        if cooling["@type"] == "comb":
            self.SpacerInvScale = 5.5
            self.LBottomChannel = [2, 4]
            self.LFchannels = [6]
            self.LSkipPrimer = [6]
            self.LEncapsulation = [4, 3, 7, 8]
            self.resistMinWidth = 100e3 * 2 + \
                2 * self.baseSizeShapes

            if "@channelTop" in cooling:
                self.channelTop = eval(cooling["@channelTop"])
            if "@channelBottom" in cooling:
                self.channelBottom = eval(cooling["@channelBottom"])
            self.widthChannel = eval(cooling["@cWidth"]) - \
                8 * self.baseSizeShapes

            self.round_outlet = 200e3
            self.round_inlet = 200e3
            self.widthMain = 290 * 2e3 - 120e3
            self.widthSpacer = 140e3

        self.removeCell()
        self.makeCell()

        # get all layers and sort them
        for n in self._ly.layer_indices():
            self.LayerInfoFCroute.append(self._ly.get_info(n))
        self.LayerInfoFCroute.sort(key=lambda x: x.layer * -1)

    def makeXYPointsFromRad(self, nPoints):
        """
        calculates the angles where to place the outlets from the main channel
        nPoints defines the number of xPoints, yPoints and angles to return

        @ Points : Number of points for the outlets
        returns np.array([[xPos1, yPos1], ...,  [xPosN, yPosN]]), \
            [angle1, ..., angleN]
        """
        alpha = np.arctan((self.y2 - self.y1) / self.x1)
        radius = self.x1 / np.sin(alpha)
        angles = np.linspace(-alpha, alpha, nPoints)

        xPoint = np.sin(angles) * radius * 1e3
        yPoint = (np.cos(angles) * radius -
                  np.cos(angles[-1]) * radius + self.y1) * 1e3

        return np.swapaxes(np.array([xPoint, yPoint]), 0, 1), angles

    def makeMainSuppylChannel(self, sizeShape=0.0):
        """
        make the main supply channel
        & sets the outlet positions.
        """
        if self.TargetCell is None:
            self.makeCell()
        points, angles = self.makeXYPointsFromRad(self.nMainCircPoints)
        if sizeShape != 0.0:
            channelWidth = self.widthMain + sizeShape * 2
        else:
            channelWidth = self.widthMain
        mainChannel = pya.Path(arrayToPointList(points),
                               channelWidth,
                               0.5*channelWidth,
                               0.5*channelWidth, True)
        self.insert(mainChannel)
        mainChannel = pya.Path(arrayToPointList(points * [1, -1]),
                               channelWidth,
                               0.5*channelWidth,
                               0.5*channelWidth, True)
        self.insert(mainChannel)

    def makePrimers(self, sizeShape=0.0, primerTip=False):
        """
        make single channel supplies Target supply channel
        """
        addScale = np.absolute(sizeShape)

        width = np.absolute(self.widthChannel + 2 * addScale)

        # the scripted shape of the primer
        conPoly = \
            np.array([[-0.5 * width, -self.lenFconnector - addScale],
                      [0.5 * width, -self.lenFconnector - addScale],
                      [0.5 * width,  0 - addScale],
                      [3.0 * width,  0 - addScale],
                      [3.0 * width,  0.5 * self.lenFconnector],
                      [-3.0 * width,  0.5 * self.lenFconnector],
                      [-3.0 * width,  0 - addScale],
                      [-0.5 * width,  0 - addScale],
                      [-0.5 * width, -self.lenFconnector - addScale],
                      [-0.5 * width, -self.lenFconnector - addScale]])
        if primerTip == True:
            conPoly = np.array([[0.5 * width,
                                 -self.lenFconnector - addScale],
                                [0.5 * width,
                                 + addScale - 0.5 * self.addSpacerScale
                                 - self.baseSizeShapes],
                                [-0.5 * width,
                                 + addScale - 0.5 * self.addSpacerScale
                                 - self.baseSizeShapes],
                                [-0.5 * width,
                                 -self.lenFconnector - addScale]])

        poly = pya.Polygon.new(arrayToPointList(conPoly))

        # compute the positions of the primers and channel outputs
        points_inlets, angles_inlet = self.makeXYPointsFromRad(
            self.nInlets + 2)
        outlets = list()
        skip_outlet_counter = 0

        # rotate each primer by complex rotation
        for n in range(1, len(points_inlets+2)-1):
            correct_displ = (self.widthMain * 0.5 -
                             np.cos(angles_inlet[n]) * self.widthMain * 0.5)
            xICp = points_inlets[n][0]
            yICp = points_inlets[n][1] - correct_displ - self.widthMain * 0.5
            TransformICP = pya.ICplxTrans().new(1.0,  # scale
                                                # angle
                                                -1 * \
                                                angles_inlet[n] / np.pi * 180,
                                                False,
                                                xICp,  # move x
                                                yICp)  # move y
            polyShape = poly.round_corners(
                self.round_outlet, self.round_inlet, 32)

            # skip the outlets
            if skip_outlet_counter not in self.skipOutlet \
                    and (self.SkipPrimerBool is False):
                polyPlace = polyShape.transformed(TransformICP)
                self.insert(polyPlace)
                polyPlaceMirror = polyPlace.transform(
                    pya.Trans(pya.Trans.M0, -0.0, -0.0))
                self.insert(polyPlaceMirror)
            outlets.append(np.array([xICp - np.sin(angles_inlet[n])
                                     * (self.lenFconnector),
                                     yICp -
                                     np.cos(angles_inlet[n]) *
                                     (self.lenFconnector),
                                     angles_inlet[n],
                                     xICp -
                                     np.sin(angles_inlet[n]) *
                                     (self.lenFconnector) * 0.5,
                                     yICp - np.cos(angles_inlet[n])
                                     * (self.lenFconnector) * 0.5]))
            skip_outlet_counter += 1
        # returns the positions of the primed outlets.
        return np.asarray(outlets)

    def makeChannelsYXpos(self):
        """
        returns the evenly spaced channel xpositions for an internal defined
        number of inlets.
        """
        return np.linspace(self.channelLeft, self.channelRight, self.nInlets)

    def makeChannelsY(self, channels, ypos, sizeShape):
        """
        make the cooling channels in y-direction. Outputs are none

        @channelPos: np.array([[channel_1_x1, ..., channel_n_x1], ...,
                               [channel_1_xn, ..., channel_n_xn]])
        @ypos:       np.array([ y1, ...,  yn])
        """
        if self.outlets is None:
            raise Exception(
                "run self.makePrimers first or define parameter outputs!")
        lenPoints = len(channels)
        for Idx in range(len(channels[0])):
            # for a small angle, go straight
            if np.absolute(self.outlets[Idx][2]) <= 0.01:
                channelPath = [self.outlets[Idx], [
                    channels[0][Idx], self.channelYknick]]
            else:
                yCannelMidd = (self.outlets[Idx][1]
                               + (self.outlets[Idx][0] - channels[0][Idx])
                               / np.tan(-self.outlets[Idx][2]))

                xCannelMidd = channels[0][Idx]
                channelPath = [self.outlets[Idx],
                               [xCannelMidd, yCannelMidd],
                               [xCannelMidd, ypos[0]]]

            for pointIdx in range(1, lenPoints):
                channelPath.append([channels[pointIdx][Idx], ypos[pointIdx]])
            widthChannel = self.widthChannel + 2 * sizeShape
            channel_poly = pya.Path(arrayToPointList(
                channelPath), widthChannel, 0.0, 0.0, True)
            self.insert(channel_poly)

    def makeChannelsYSpline(self, channels, ypos, sizeShape=0.0):
        """
        make the cooling channels in y-direction. smooth by spline approximation

        :param channels: np.array([[channel_1_x1, ..., channel_n_x1], ...,
                                   [channel_1_xn, ..., channel_n_xn]])
        :param ypos:     np.array([ y1, ...,  yn])
        """
        if self.outlets is None:
            raise Exception(
                "run self.makePrimers first or define parameter outputs!")

        lenPoints = len(channels)
        for Idx in range(len(channels[0])):
            outlet_01 = self.outlets[Idx][3:5]
            outlet_02 = (self.outlets[Idx][0:2]
                         + 0.5 * (self.outlets[Idx][0:2]
                                  - self.outlets[Idx][3:5]))

            channelPath = [outlet_01, outlet_02]

            for pointIdx in range(1, lenPoints):
                channelPath.append([channels[pointIdx][Idx], ypos[pointIdx]])

            channelPath.append(outlet_02 * [1, -1])
            channelPath.append(outlet_01 * [1, -1])

            widthChannel = self.widthChannel + 2 * sizeShape

            if self.SkipPrimerBool:
                channelPath = channelPath  # [1:-1]
                if Idx not in self.skipOutlet:
                    self.insert(pya.Path(arrayToPointList([channelPath[0],
                                                           channelPath[0]]),
                                         widthChannel, 0.5 * widthChannel,
                                         0.5 * widthChannel, True))
                    self.insert(pya.Path(arrayToPointList([channelPath[-1],
                                                           channelPath[-2]]),
                                         widthChannel, 0.5 * widthChannel,
                                         0.5 * widthChannel, True))

            if Idx not in self.skipOutlet:

                Start = channelPath[0:2]
                End = channelPath[-2:]
                factor = 0.5

                End_addPT = (End[1] - End[0]) * factor
                Start_addPT = (Start[0] - Start[1]) * factor

                channelPath.insert(0, channelPath[0] + Start_addPT)
                channelPath.append(channelPath[-1] + End_addPT)

                self.insert(pya.Path(arrayToPointList([channelPath[0],
                                                       channelPath[0]]),
                                     widthChannel, 0.5*widthChannel,
                                     0.5*widthChannel, True))

                self.insert(pya.Path(arrayToPointList([channelPath[-1],
                                                       channelPath[-2]]),
                                     widthChannel, 0.5*widthChannel,
                                     0.5*widthChannel, True))

                self.insert(pathPolyRound(arrayToPointList(channelPath),
                                          widthChannel, r1=10000e3, r2=10000e3,
                                          npoint=256))  # channel_2nd_Part

    def makeSpacers(self, sizeShape=0.0, multi=16):
        """
        inserts spacers into the main channel for lamination reasons
        """
        nInletsPseudo = 14
        points, angles = self.makeXYPointsFromRad(nInletsPseudo * multi + 1)
        separations = list()
        for n in range(multi, nInletsPseudo*(multi), multi):
            separations.append(np.array([points[n], points[n]]))

        widthSpacer = self.widthSpacer - 3 * 10 * 1e3
        for scarray in separations:
            SpacerMain = pya.Path(arrayToPointList(scarray),
                                  widthSpacer, 0.3 * widthSpacer,
                                  0.3*widthSpacer, True)

            # convert to polygon and size inversely
            poly = SpacerMain.polygon()
            poly.size(-1 * sizeShape + self.SpacerInvScale * 30e3)
            self.TargetCell.shapes(self._ly.layer(
                self.LayerInfoSepMerge)).insert(poly)
            polyPlaceMirror = poly.transform(
                pya.Trans(pya.Trans.M0, -0.0, -0.0))

            self.TargetCell.shapes(self._ly.layer(
                self.LayerInfoSepMerge)).insert(polyPlaceMirror)
        self.mergeShapes(self.LayerInfoSepMerge)
        self.shapeBoolean(self.LayerInfoSepMerge,
                          self.LayerInfo, self.LayerInfo)

        LayerIdx = self._ly.layer(self.LayerInfoSepMerge)
        for n in self.TargetCell.shapes(LayerIdx).each():
            self.TargetCell.shapes(LayerIdx).erase(n)

    def skipChannels(self, viaList=[]):
        channelXPos = self.makeChannelsYXpos()
        for n in range(len(channelXPos)):
            for via in viaList:
                minValue = channelXPos[n] < via.min() - self.resistMinWidth
                maxValue = channelXPos[n] > via.max() + self.resistMinWidth
                if (minValue ^ maxValue) is False:
                    self.skipOutlet.append(n)

    def arrangeC(self, start, end, step=None, addsep=0.0):
        """
        arrange Centre
        returns a np.arange array but with the array
        alinged in the centred in respect to start/end.
        """
        if step == None:
            step = (self.widthChannelSpacer + self.widthChannel + addsep)
        arange = np.arange(start, end, step)
        centerSpacer = (end - arange[-1]) * 0.5
        return arange + centerSpacer

    def arrangeL(self, start, end, step=None, addsep=0.0):
        """
        arrange Left
        returns a np.arange array but with the array
        alinged in the centred in respect to start/end.
        """
        if step is None:
            step = (self.widthChannelSpacer + self.widthChannel + addsep)
        arange = np.arange(start, end, step)
        return arange

    def arrangeR(self, start, end, step=None, addsep=0.0):
        """
        arrange Left
        returns a np.arange array but with the array
        alinged in the centred in respect to start/end.
        """
        if step is None:
            step = (self.widthChannelSpacer + self.widthChannel + addsep)
        arange = np.arange(start, end, step)
        LeftSpacer = (end - arange[-1])
        # print( "arangeR", arange)
        return arange + LeftSpacer

    def bboxViaSep(self, gradientCoils, LayerInfo):
        """
        returns bboxes of the shapes present in the Layer of LayerInfo

        :param gradientCoils: list of gradient coil objects with the target
        :param LayerInfo: pya.LayerInfo object
        """
        for gradient in gradientCoils:
            for shape in gradient.TargetCell.shapes(
                    self._ly.layer(LayerInfo)).each():
                if (shape.bbox().right > self.channelLeft) \
                        and (self.channelRight > shape.bbox().left):
                    self.bboxViaSepList.append(shape.bbox())

    def viaYGridClose(self):
        """
        returns the a list of index lists which corresponds to shapes which may
        interfere with the channel routing.

        computes it from self.bboxViaSepList
        """
        viaIdxYClose = []
        yPos = []

        # Start from the Top to Bottom
        idx_sorted = sorted(range(len(self.bboxViaSepList)),
                            key=lambda n: (
                                self.bboxViaSepList[n].center().y * -1))

        # check which idxVias are overlapping for same y
        for idx in idx_sorted:
            # set the width of the via
            idxLocalClose = [idx]
            for idxN in idx_sorted:  # check if other via's interfere!
                # p1 is (top    left ), p2 is (bottom right)
                # check if vias are close in Y-direction!
                if ((idxN != idx) and  # via is other than outer for
                    (self.bboxViaSepList[idxN].p2.y >=
                     self.bboxViaSepList[idx].p1.y - self.viaExtraSep)
                    and (self.bboxViaSepList[idxN].p1.y < \
                         self.bboxViaSepList[idx].p2.y + self.viaExtraSep)):
                    idxLocalClose.append(idxN)
            viaIdxYClose.append(idxLocalClose)
            yPos.append(self.bboxViaSepList[idx].center().y)
        return viaIdxYClose, yPos

    def channelXPosFit(self, idxListVia):
        """
        Returns a list of intervals for via spacings.

        :param viaList: list of pya.box via objects
        :param idxListVia: list of idx which intersect for a given y-position.
        :rtype: list of gap list
        """
        # TODO: make a separate function to compute the number of inlets
        inSectVias = []
        viaItvLst = []

        for n in idxListVia:
            inSectVias.append(self.bboxViaSepList[n])

        # create Via Intervals:
        for n in sorted(range(len(inSectVias)),
                        key=lambda n: (inSectVias[n].p1.x)):
            viaItvLst.append([inSectVias[n].p1.x, inSectVias[n].p2.x])

        # create the gaps
        gapItvLst = []
        viaPre = None

        for Via in viaItvLst:
            # start condition - add start value
            if len(gapItvLst) == 0:
                # Via is larger than channel left
                if (Via[0] > self.channelLeft):
                    gapItvLst.append([self.channelLeft, Via[0]])
                # Via is smaller than channel left and defines start intv
                if (Via[0] <= self.channelLeft) \
                        and (Via[1] > self.channelLeft):
                    gapItvLst.append([Via[1]])
            elif len(gapItvLst) != 0:
                # add new start
                if len(gapItvLst[-1]) == 2:
                    if (Via[0] > gapItvLst[-1][1]):
                        gapItvLst.append([viaPre[1], Via[0]])
                # append end and close intv
                elif len(gapItvLst[-1]) == 1:
                    if (Via[0] > gapItvLst[-1][0]):
                        gapItvLst[-1].append(Via[0])
            viaPre = Via

        # make the last gap
        if len(gapItvLst) == 0:
            gapItvLst.append([self.channelLeft, self.channelRight])
        elif len(gapItvLst[-1]) == 2:
            # and viaPre[0] > gapItvLst[-1][1]):
            if (viaPre[1] < self.channelRight):
                gapItvLst.append([viaPre[1], self.channelRight])
        return gapItvLst

    def fludicConnector(self, sizeShape=0.0):
        """ Creates the Quadro Sided Fluidic connector """
        points = np.array(
            [[self.x1, self.y1], [self.x1 + 162.6, self.y1 - 162.6]]) * 1e3
        connectors = [points, points * [-1, 1],
                      points * [1, -1], points * [-1, -1]]
        for con in connectors:  # make the 4 fluidic connectors
            connectorPath = pya.Path(arrayToPointList(con),
                                     self.widthMain + 2 * sizeShape,
                                     0.5 * (self.widthMain + 2 * sizeShape),
                                     0.5 * (self.widthMain + 2 * sizeShape),
                                     True)
            self.insert(connectorPath)

    def channelPosFromItv(self, gapItvLst, addsep=0.0):
        """ returns the X-positions where to place the fludic channels
        :param : """

        rtvXPos = list()
        for gap in gapItvLst:
            if gap[0] == self.channelLeft:
                if gap[1] == self.channelRight:
                    rtvXPos = [*rtvXPos, *
                               self.arrangeC(gap[0], gap[1], addsep=addsep)]
                else:
                    rtvXPos = [
                        *rtvXPos, *self.arrangeL(gap[0], gap[1]
                                                 - self.resistMinWidth,
                                                 addsep=addsep)]
            elif gap[1] == self.channelRight:
                rtvXPos = [*rtvXPos,
                           *self.arrangeR(gap[0] + self.resistMinWidth,
                                          gap[1], addsep=addsep)]
            else:  # channel in the middle
                rtvXPos = [*rtvXPos, *self.arrangeC(
                    gap[0] + 1.5 * self.resistMinWidth, gap[1]
                    - self.resistMinWidth, addsep=addsep)]
        return rtvXPos

    def getMaxChannels(self, viaIdxYClose):
        lstLen = []
        for viaYClose in viaIdxYClose:  # viaYClose -- if via in channel region
            lstLen.append(len(self.channelPosFromItv(
                self.channelXPosFit(viaYClose))))
        if len(lstLen) == 0:
            return len(self.makeChannelsYXpos())
        return min(lstLen)

    def makeEqualChannelQuant(self, viaIdxYClose):
        maxChannel = self.getMaxChannels(viaIdxYClose)
        addsepLst = []
        repeat = False
        for viaYClose in viaIdxYClose:
            addsep = 0.0
            while len(self.channelPosFromItv(
                    self.channelXPosFit(viaYClose),
                    addsep=addsep)) > maxChannel:
                addsep += 2.0 * 10**3
            addsepLst.append(addsep)
            if len(self.channelPosFromItv(self.channelXPosFit(viaYClose),
                                          addsep=addsep)) < maxChannel:
                repeat = True

        # repeat the separation distribution because of some errors
        if repeat:
            addsepLst = []
            for viaYClose in viaIdxYClose:
                addsep = 0.0
                while len(self.channelPosFromItv(
                        self.channelXPosFit(viaYClose),
                        addsep=addsep)) > maxChannel:
                    addsep += 2e3
                addsepLst.append(addsep)
        return addsepLst

    def makeChannelGrid(self, gradientCoils=[], LayerInfo=None,
                        LayerInfoViaSep=None, sizeShape=0.0):
        """ creates the Channel Grid """

        # get the bounding boxes for of the Vias.
        self.bboxViaSepList = []
        self.bboxViaSep(gradientCoils, LayerInfoViaSep)
        viaIdxYClose, yPos = self.viaYGridClose()

        # get the maximum number of channels
        self.nInlets = self.getMaxChannels(viaIdxYClose)
        addsep = self.makeEqualChannelQuant(viaIdxYClose)

        # append the via positions to the channels list
        channels = []
        for n in range(len(viaIdxYClose)):
            XchannelPos = self.channelPosFromItv(
                self.channelXPosFit(viaIdxYClose[n]),
                addsep=addsep[n])
            channels.append(XchannelPos)

        self.outlets = self.makePrimers(sizeShape=sizeShape)

        # make the top and bottom inlets
        channels.insert(0, self.makeChannelsYXpos())
        channels.insert(0, self.makeChannelsYXpos())
        channels.append(self.makeChannelsYXpos())
        channels.append(self.makeChannelsYXpos())

        if len(yPos) > 1:
            yposVia = min(yPos)
        else:
            yposVia = -950e3

        yPos = [* [self.channelTop, self.channelTop - 100e3],
                * yPos,
                * [self.channelBottom + 100e3, self.channelBottom]]

        if LayerInfo.datatype != 1 or LayerInfo.layer != 4:
            self.makeChannelsYSpline(channels, yPos, sizeShape=sizeShape)


def makeFluidicCooling(cooling, TargetCellName, removeCell=True, gradientCoils=[]):
    """
    creates cooling cannels in the target Cell.

    :param cooling:        dictionary with the parameters from the XML
    :param TargetCellName: Cell name where to place the cell
    :param remove:         if True remove the cell before creation.
    :param gradientCoils:  to get the vias of the gradint coils.
                           The return of the Gradient Coil creator
    """
    fc = FluidicChannels(TargetCellName, cooling, gradientCoils=gradientCoils)
    makeCoolingChannelsRoutine(fc, cooling)

    # calculate skip channels
    if cooling["@type"] == "comb":
        a = fc.makeXYPointsFromRad(fc.nInlets)[0][::, 0]

        b = np.where(np.logical_and(a >= 770e3, a <= 1580e3))
        skip = np.append(
            b, (np.where(np.logical_and(a >= -1580e3, a <= -770e3))))

        fc = FluidicChannels(TargetCellName, cooling,
                             gradientCoils=gradientCoils)
        fc.skipOutlet = skip
        makeCoolingChannelsRoutine(fc, cooling)


def makeCoolingChannelsRoutine(fc, cooling):
    for n in range(len(fc.LayerInfoFCroute)):
        if (fc.LayerInfoFCroute[n].layer in fc.LSkipPrimer):
            fc.SkipPrimerBool = True
        else:
            fc.SkipPrimerBool = False
        multiplier = 0

        # size the shapes individually for the different parts of cooling
        if cooling["@type"] == "comb":
            if fc.LayerInfoFCroute[n].layer == 2:
                multiplier = 1
            if fc.LayerInfoFCroute[n].layer in [3, 4]:
                multiplier = 2
            if fc.LayerInfoFCroute[n].layer >= 6:
                multiplier = 3

            size = fc.baseSizeShapes * (multiplier + 2)
            sizeChannels = fc.baseSizeShapes * (multiplier + 1)

        elif cooling["@type"] == "middle":
            size = fc.baseSizeShapes * (fc.LayerInfoFCroute[n].layer - 2)

        # sets the LayerInfo for drawing the gradient shapes
        fc.LayerInfo = fc.LayerInfoFCroute[n]

        # Bottom channel
        if fc.LayerInfoFCroute[n].layer in fc.LBottomChannel:
            fc.makeMainSuppylChannel(sizeShape=size)
            fc.outlets = fc.makePrimers(sizeShape=size)
            fc.fludicConnector(size)
            fc.mergeShapes()
            fc.makeSpacers(sizeShape=size)

        # Fludic Channel
        if (fc.LayerInfoFCroute[n].layer in fc.LFchannels):
            if ((fc.LayerInfoFCroute[n].layer not in fc.LSkipPrimer)
                    and not (fc.LayerInfoFCroute[n].layer == 4
                             and (fc.LayerInfoFCroute[n].datatype == 0))):
                fc.SkipPrimerBool = False
                fc.makeMainSuppylChannel(sizeShape=size)
            else:
                fc.SkipPrimerBool = True
            fc.LayerInfo = fc.LayerInfoFCroute[n]
            if cooling["@type"] == "comb":  # this is hard coded for simplicity
                fc.makeChannelGrid(fc.gradientCoils,
                                   LayerInfo=fc.LayerInfoFCroute[n],
                                   LayerInfoViaSep=pya.LayerInfo(3, 0),
                                   sizeShape=sizeChannels * 1.00)
            else:
                fc.makeChannelGrid(fc.gradientCoils,
                                   LayerInfo=fc.LayerInfoFCroute[n],
                                   LayerInfoViaSep=fc.LayerInfoFCroute[n],
                                   sizeShape=size)

            if (cooling["@type"] == "comb"):
                fc.mergeShapes()
                fc.makeSpacers(sizeShape=size)
                fc.fludicConnector(size)
            else:
                fc.fludicConnector(size)
                fc.mergeShapes()
                fc.makeSpacers(sizeShape=size)

        # Encapsulation
        if fc.LayerInfoFCroute[n].layer in fc.LEncapsulation:
            if fc.LayerInfoFCroute[n].layer == 6 \
                    or fc.LayerInfoFCroute[n].layer == 7 \
                    or fc.LayerInfoFCroute[n].layer == 8:

                size = fc.baseSizeShapes * (fc.LayerInfoFCroute[n].layer - 3)
            if (cooling["@type"] == "comb"):
                if fc.LayerInfoFCroute[n].layer != 8:
                    fc.fludicConnector(size * 1.0)
                else:
                    fc.fludicConnector(size * 1.8)
            else:
                fc.fludicConnector(size * 2.0)
            if cooling["@type"] == "comb" \
                    and fc.LayerInfoFCroute[n].layer == 3:
                fc.makePrimers(sizeShape=size * 0.8, primerTip=True)
            fc.mergeShapes()
