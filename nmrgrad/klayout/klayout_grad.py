"""

"""
from scipy import (spatial, interpolate)
from lxml import etree
import numpy as np

try:
    import pya
except ImportError:
    # without klayout gui
    import klayout.db as pya


class KlayoutPyFeature():
    TargetCellName = "klayoutPyFeatureCell"
    TargetCell = None

    LayerViaInfos = [pya.LayerInfo(0, 0), pya.LayerInfo(2, 0),
                     pya.LayerInfo(3, 0), pya.LayerInfo(4, 0),
                     pya.LayerInfo(5, 1), pya.LayerInfo(5, 0)]

    LayerSU8TopLitho = pya.LayerInfo(2, 0)
    LayerInfo = pya.LayerInfo(0, 0)

    topChipWidth = 4400e3
    widthConnector = 680e3  # for the gradient coil connectors
    lenConnector = 550e3  # for the gradient coil connectors

    widthVia = 250e3
    lenVia = 250e3
    resistMinWidth = 85e3

    def __init__(self, TopCellName):
        if "Application" in dir(pya):
            app = pya.Application.instance()
            self._mw = app.main_window()
            self._lv = self._mw.current_view()

            if self._lv is None:
                raise Exception("No view selected!")
            self._ly = self._lv.active_cellview().layout()

            if self._ly.has_cell(TopCellName) is False:
                raise Exception('TopCell of name "' + TopCellName
                                + '" to insert the Cell does not exist!')
            else:
                self.TopCell = self._ly.cell(
                    self._ly.cell_by_name(TopCellName))
        else:
            self._ly = pya.Layout()

            for n in self.LayerViaInfos:
                self._ly.insert_layer(n)

            self._ly.create_cell(TopCellName)
            self.TopCell = self._ly.cell(self._ly.cell_by_name(TopCellName))

    def makeCell(self, trans=pya.Trans.new(pya.Point.new(0, 0))):
        """ creates the new cell """
        if self._ly.has_cell(self.TargetCellName) is False:
            self.TargetCell = self._ly.create_cell(self.TargetCellName)
        else:
            self.TargetCell = \
                self._ly.cell(self._ly.cell_by_name(self.TargetCellName))

        self.TopCell.insert(pya.CellInstArray.new(self.TargetCell.cell_index(),
                                                  trans))

    def removeCell(self):
        """
        Delete the scripted cell / remove content of the cell.

        Returns
        -------
        None.

        """
        if self._ly.has_cell(self.TargetCellName):
            self._ly.prune_cell(self._ly.cell_by_name(self.TargetCellName), -1)

    def insert(self, shape):
        """
        Simple abstraction to insert a shape into the target cell.

        Parameters
        ----------
        shape : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.TargetCell.shapes(self._ly.layer(self.LayerInfo)).insert(shape)

    def mirrorArrayXaxis(self, array):
        arraySwap = np.swapaxes(array, 0, 1)
        return np.swapaxes(np.array([arraySwap[0],
                                     arraySwap[1] * -1.0]), 0, 1)

    def mergeShapes(self, LayerInfo=None):
        if str(type(LayerInfo)) != "<class 'pya.LayerInfo'>":
            layer_idx = self._ly.layer(self.LayerInfo)
        else:
            layer_idx = self._ly.layer(LayerInfo)

        merge_processor = pya.ShapeProcessor()
        if self.TargetCell.shapes(layer_idx).is_empty() is False:
            merge_processor.merge(self._ly, self.TargetCell, layer_idx,
                                  self.TargetCell.shapes(layer_idx), True, 0,
                                  True, True)

    def shapeBoolean(self, layer01, layer02, layerBoole,
                     boolType=pya.EdgeProcessor.mode_bnota()):
        """ compute the shape boolean """
        if str(type(layer01)) == "<class 'pya.LayerInfo'>":
            LayerIdx01 = self._ly.layer(layer01)
        elif self._ly.is_valid_layer(layer01.layer):
            LayerIdx01 = layer01
        else:
            return 0

        if str(type(layer02)) == "<class 'pya.LayerInfo'>":
            LayerIdx02 = self._ly.layer(layer02)
        elif self._ly.is_valid_layer(layer02.layer):
            LayerIdx02 = layer02
        else:
            return 0

        if str(type(layerBoole)) == "<class 'pya.LayerInfo'>":
            LayerIdxBoole = self._ly.layer(layerBoole)
        elif self._ly.is_valid_layer(layerBoole.layer):
            LayerIdxBoole = layerBoole
        else:
            return 0

        processor = pya.ShapeProcessor()

        if type(LayerIdx01) == int:  # pya mode!
            processor.boolean(self._ly, self.TargetCell, LayerIdx01,
                              self._ly, self.TargetCell, LayerIdx02,
                              self.TargetCell.shapes(LayerIdxBoole),
                              boolType, True, True, True)
        else:
            processor.boolean(self._ly, self.TargetCell, LayerIdx01.layer,
                              self._ly, self.TargetCell, LayerIdx02.layer,
                              self.TargetCell.shapes(LayerIdxBoole.layer),
                              boolType, True, True, True)

    def removeShapeLayer(self, LayerInfo):
        layer_idx = self._ly.layer(LayerInfo)
        for n in self.TargetCell.shapes(layer_idx).each():
            self.TargetCell.shapes(layer_idx).erase(n)

    def invert_layer(self, target_layer):
        if str(type(target_layer)) == "<class 'pya.LayerInfo'>":
            layer_idx = self._ly.layer(target_layer)
        elif self._ly.is_valid_layer(target_layer):
            layer_idx = target_layer
        else:
            return 0

        self.TopCell.flatten(-1, True)
        processor = pya.ShapeProcessor()
        processor.merge(self._ly, self.TopCell, layer_idx,
                        self.TopCell.shapes(layer_idx), True, 0, True, True)

        self.TargetCell = self.TopCell
        lbool = self._ly.insert_layer(pya.LayerInfo(100, 0, "booleantmp"))
        self.TopCell.shapes(lbool).insert(
            pya.Box.new(-self.invFrameX / 2, -self.invFrameY / 2,
                        self.invFrameX / 2, self.invFrameY / 2))

        self.shapeBoolean(layer_idx,
                          lbool,
                          layer_idx,
                          boolType=pya.EdgeProcessor.mode_xor())

        self.TopCell.clear(lbool)
        self._ly.delete_layer(lbool)

    def makeVia(self, pos, cell=None, vertical=False,
                trans=pya.Trans(pya.Trans.R0, 0.0, 0.0)):
        """
        Create a single vertiacl access interconnect on the subjected cell.

        Parameters
        ----------
        pos : list
            list with the x and y coordinate.
        cell : pya.Cell, optional
            The subjected cell to create the VIA in. The default is None.
        vertical : TYPE, optional
            DESCRIPTION. The default is False.
        trans : TYPE, optional
            DESCRIPTION. The default is pya.Trans(pya.Trans.R0, 0.0, 0.0).

        Returns
        -------
        ViaXDimen : ndarray
            DESCRIPTION.

        """
        flatten = False

        if cell is None:
            cell = self._ly.create_cell("helperCellVia")
            self.TargetCell.insert(pya.CellInstArray.new(cell.cell_index(),
                                                         trans))
            flatten = True

        if vertical is False:
            ViaPos = np.array([[pos[0] - .5 * self.lenVia
                                + .5 * self.widthVia,
                                pos[1]],
                               [pos[0] + .5 * self.lenVia
                                - .5 * self.widthVia,
                                pos[1]]])

            ViaXDimen = np.array([ViaPos[0][0] - .5 * self.widthVia,
                                  ViaPos[1][0] + .5 * self.widthVia])

        elif vertical:
            ViaPos = np.array([[pos[0],
                                pos[1] - .5 * self.lenVia
                                + .5 * self.widthVia],
                               [pos[0],
                                pos[1] + .5 * self.lenVia
                                - .5 * self.widthVia]])

            ViaXDimen = np.array([ViaPos[0][0] - .5 * self.widthVia,
                                  ViaPos[1][0] + .5 * self.widthVia])
        else:
            return

        Via = pya.Path(arrayToPointList(ViaPos), self.widthVia,
                       .5 * self.widthVia, .5 * self.widthVia, True)

        for LayerInfo in self.LayerViaInfos:
            cell.shapes(self._ly.layer(LayerInfo)).insert(Via)

        if flatten:
            self.TargetCell.flatten(-1, True)

        return ViaXDimen

    def RoundPathF(self, pts, radius=500, width=400, LayerInfo=None):
        helperCell = KlayoutPyFeature(self.TargetCellName)
        helperCell.TargetCellName = "helperCell_del_Grad" \
            + str(np.random.rand())
        helperCell.removeCell()
        helperCell.makeCell()

        if LayerInfo is None:
            LayerInfo = self.LayerInfo

        placeRoundPath(helperCell.TargetCell, helperCell._ly,
                       radius=radius, width=width, pts=pts,
                       LayerInfo=LayerInfo)


def arrayToPointList(array):
    ''' convert np.array [[x,y], ... [x_n, y_n]] to point list for klayout'''
    pts = list()

    for n in array:
        pts.append(pya.Point.from_dpoint(pya.DPoint(n[0], n[1])))

    return pts


def rotate2D(pts, cnt, ang=np.pi):
    """
    Rotates points(nx2) about center cnt(2) by angle ang(1) in radian.

    Parameters
    ----------
    pts : ndarray
        The point to rotate.
    cnt : ndarray
        The rotation centre.
    ang : float, optional
        The rotation angle in radian. The default is np.pi.

    Returns
    -------
    ndarray
        The point after rotation.

    """
    return np.dot(pts - cnt,
                  np.array([[np.cos(ang), np.sin(ang)],
                            [-np.sin(ang), np.cos(ang)]])) + cnt


def placeText(cell, ly, text="test", pos=[.0, .0],
              trans=None, mag=1000, LayerInfo=pya.LayerInfo(0, 0),
              mirror=False):
    ''' Place a TEXT pcell on the frame (adapted to pyhton form Klayout forum)
    '''
    lib = pya.Library.library_by_name("Basic")

    if lib is None:
        raise Exception("Unknown lib 'Basic'")

    pcell_decl = lib.layout().pcell_declaration("TEXT")

    if pcell_decl is None:
        raise Exception("Unknown PCell 'TEXT'")

    # parameters
    param = {"text": text, "layer": LayerInfo, "mag": mag * 1.4286}
    # translate to array (to pv)
    pv = []
    for p in pcell_decl.get_parameters():
        if p.name in param:
            pv.append(param[p.name])
        else:
            pv.append(p.default)

    # create the PCell variant cell
    pcell_var = ly.add_pcell_variant(lib, pcell_decl.id(), pv)

    # transformation
    if str(type(trans)) != "<class 'pya.Trans'>":
        trans = pya.Trans(pya.Trans.R0, -1000 + pos[0], -1000 + pos[1])

    # insert into "top_cell" (must be provided by the caller)
    pcell_inst = cell.insert(pya.CellInstArray(pcell_var, trans))


def placeRoundPath(cell, ly, pts=[], trans=None,
                   width=100, radius=100, npoints=64,
                   LayerInfo=pya.LayerInfo(0, 0), RadPath=None):
    ''' Place a Round Path pcell on the frame (adapted to pyhton form Klayout
    forum)
    '''
    lib = pya.Library.library_by_name("Basic")

    if lib is None:
        raise Exception("Unknown lib 'Basic'")

    pcell_decl = lib.layout().pcell_declaration("ROUND_PATH")
    if pcell_decl is None:
        raise Exception("Unknown PCell 'ROUND_PATH'")

    # translating named parameters into an ordered sequence ...
    a1 = []
    for p in pts:
        a1.append(pya.DPoint(p[0], p[1]))
    if RadPath is None:
        wg_path = pya.DPath(a1, width, .5 * width, .5 * width, True)
    else:
        wg_path = pya.DPath(a1, width, .5 * RadPath, .5 * RadPath, True)
    # named parameters
    if type(LayerInfo) == pya.LayerInfo(0, 0).__class__:
        LayerInfo = [LayerInfo]

    for nLayerInfo in LayerInfo:
        param = {"npoints": npoints,
                 "radius": radius,
                 "path": wg_path,
                 "layer": nLayerInfo}
        # translate to array (to pv)
        pv = []
        for p in pcell_decl.get_parameters():
            if p.name in param:
                pv.append(param[p.name])
            else:
                pv.append(p.default)

        # create the PCell variant cell
        pcell_var = ly.add_pcell_variant(lib, pcell_decl.id(), pv)

        # transformation
        if str(type(trans)) != "<class 'pya.Trans'>":
            trans = pya.Trans(0, 0)

        # insert into "top_cell" (must be provided by the caller)
        pcell_inst = cell.insert(pya.CellInstArray(pcell_var, trans))


def svg_pt_to_ndarray(string):
    centreX = 516.685
    centreY = 873.622 - 455.843
    scaleX = 1e6 / (642.491 - centreX)
    scaleY = 1e6 / (873.622 - 581.649 - centreY)
    pointlist = list()

    for i in string.split():
        pointlist.append((np.array(i.split(","), dtype=float)
                          - np.array([centreX, centreY]))
                         * np.array([scaleX, scaleY]))

    return np.asarray(pointlist)


def import_svg_polyline(filename, blueColor="#0000ff", redColor="#ff0000"):
    xmlETree = etree.parse(filename).getroot()
    SVG_NS = "http://www.w3.org/2000/svg"  # SVG Namespace
    blue = list()
    red = list()
    min_value = 20

    for node in xmlETree.findall('.//{%s}polyline' % SVG_NS):
        if "stroke" in node.keys():
            if node.attrib["stroke"] == blueColor \
                    and svg_pt_to_ndarray(node.attrib["points"]).shape[0] >= 20:
                blue.append(svg_pt_to_ndarray(node.attrib["points"]))
            elif node.attrib["stroke"] == redColor\
                    and svg_pt_to_ndarray(node.attrib["points"]).shape[0] >= 20:
                red.append(svg_pt_to_ndarray(node.attrib["points"]))
    return sort_ndarray(blue), sort_ndarray(red)


def sort_ndarray(listArray):
    indexList = []
    sortedList = []

    for n in listArray:
        sumVector = 0.0
        for vector in n:
            sumVector += np.sqrt(vector[0]**2 + vector[1]**2)
        indexList.append(sumVector)

    for new_index in [i[0] for i in sorted(enumerate(indexList),
                                           key=lambda x: x[1])]:
        sortedList.append(listArray[new_index])

    return sortedList


def centre_array_at_pts(array_like, pts):
    """ Centers a given array_like at its coloses point
    of a given point list pts
    array is [[x_0, y_0], [x_1, y_1], ..., [x_n, y_n]] - returns index
    """
    tree = spatial.KDTree(array_like)
    distance, index = tree.query(pts)
    return np.roll(array_like, -index[np.argmin(distance)], axis=0)


def min_dist_array_at_pts(array_like, pts):
    """
    Get the minimum distance from a given array_like at its coloses
    point of a given point list pts
    array is [[x_0, y_0], [x_1, y_1], ..., [x_n, y_n]] - returns index
    """
    tree = spatial.KDTree(array_like)
    distance, index = tree.query(pts)
    return distance.min()


def array_idx_dist(array, distMax=None, start=False, end=False):
    startLen = 0
    endLen = 0
    startIdx = 0
    endIdx = 0

    for n in range(len(array) - 1):
        if startLen < distMax:
            startLen += np.sqrt((array[n+1][0] - array[n][0])**2 \
                                + (array[n+1][1] - array[n][1])**2)
            startIdx = n
        else:
            break
    n = len(array) - 1
    counter = 0
    while True:
        endLen += np.sqrt((array[n-1][0] - array[n][0])**2
                          + (array[n-1][1] - array[n][1])**2)
        endIdx = n
        if endLen > distMax:
            break
        if n <= 0:
            break
        n = n - 1

    if start:
        startIdx = 0

    if end:
        endIdx = -1

    return array[startIdx:endIdx]


def bspline(cv, n=20):
    """ calculates a simple bspline approximation.
        cv : Array ov control vertices np.array([[x_0, y_0], ... [x_n, y_n]]),
        n  : Number of sample points on the spline approximaiton
    """
    pts = np.swapaxes(cv, 0, 1)
    tck, u = interpolate.splprep(pts, s=0.0)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, n), tck)
    return np.swapaxes([x_i, y_i], 0, 1)


def idx_nearest(array, value):
    return (np.absolute(array - value)).argmin()


def placeArc(cell, ly, trans,
             r1=100, r2=200, angle_start=.0, angle_end=90.0,
             LayerInfo=pya.LayerInfo(0, 0), npoints=32):
    """ place an Arc based on the Klayout Pcell """
    lib = pya.Library.library_by_name("Basic")

    if lib is None:
        raise Exception("Unknown lib 'Basic'")

    pcell_decl = lib.layout().pcell_declaration("ARC")

    if pcell_decl is None:
        raise Exception("Unknown PCell 'TEXT'")

    param = {"layer": LayerInfo,
             "actual_radius1": r1,
             "actual_radius2": r2,
             "actual_start_angle": angle_start,
             "actual_end_angle": angle_end,
             "npoints": npoints}

    # translate to array (to pv)
    pv = []
    for p in pcell_decl.get_parameters():
        if p.name in param:
            pv.append(param[p.name])
        else:
            pv.append(p.default)

    # create the PCell variant cell
    pcell_var = ly.add_pcell_variant(lib, pcell_decl.id(), pv)

    # insert into "top_cell" (must be provided by the caller)
    pcell_inst = cell.insert(pya.CellInstArray(pcell_var, trans))

    xc_circle = trans.disp.x * 1e-3
    yc_circle = trans.disp.y * 1e-3

    radius = ((r2 - r1) * .5 + r1)

    start_pt = [xc_circle + radius * np.cos(np.deg2rad(angle_start)),
                yc_circle + radius * np.sin(np.deg2rad(angle_start))]

    end_pt = [xc_circle + radius * np.cos(np.deg2rad(angle_end)),
              yc_circle + radius * np.sin(np.deg2rad(angle_end))]

    arc_T = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_end), npoints)
    arc_pts = np.asarray([xc_circle + radius * np.cos(arc_T),
                          yc_circle + radius * np.sin(arc_T)])
    return arc_pts.T


def containerPos(xIdx, yIdx):
    placeBottomChips = np.array([[0, True, True, 0], [True, 0, 0, True],
                                 [True, 0, 0, True], [0, True, True, 0]])
    shapeX, shapeY = placeBottomChips.shape
    chipSizeX = 16.0e6
    chipSizeY = 16.0e6
    xcoord = np.linspace(-(shapeX - 1) * chipSizeX * .5,
                         (shapeX - 1) * chipSizeX * .5, shapeX)
    ycoord = np.linspace(-(shapeY - 1) * chipSizeY * .5,
                         (shapeY - 1) * chipSizeY * .5, shapeY)
    return xcoord[xIdx], ycoord[yIdx]


def makeChipContainerS(shapeX=4, shapeY=4, reFresh=False):
    for X in range(shapeX):
        for Y in range(shapeY):
            chipContainer = KlayoutPyFeature("ChipFeatures")
            chipContainer.TargetCellName = "Chip_" + str(X) + "_" + str(Y)
            contPos = containerPos(X, Y)
            chipContainer.makeCell(
                trans=pya.Trans.new(pya.Point.new(contPos[0], contPos[1])))


def pathPolyRound(Points, width, r1=200e3, r2=200e3, npoint=32):
    return pya.Path(Points,
                    width,
                    .0, .0,
                    True).polygon().round_corners(r1, r2, npoint)


def layerInfo_eq_list(LayerInfo, CMPLayerInfoLst):
    """
    compares the given layer info with the layer infos
    in given list of LayerInfos

    returns True if in List
    """
    for n in CMPLayerInfoLst:
        if LayerInfo.is_equivalent(n):
            return True
    return False
