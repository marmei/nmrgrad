# $description: CELL to SVG
# $show-in-menu

import pya
import svgwrite
from struct import (pack, unpack)
from svgwrite.extensions import Inkscape
from pathlib import Path


def layout_cell_to_svg(cv, ly, lv, fname, scale_factor=1e3):
    """
    Export ther verticies of all active layers from the current cell.

    This script creates a temporal layout to flatten and merge the
    source cell.

    Todo: make correct layer colors

    Parameters
    ----------
    cv : klayout active cellview
        The active cellview of the klayout object.
    ly : klayout layout
        The current layout.
    lv : klayout view
        The current view.
    fname : pathlib.Path
        Filepath to store the generated SVG.
    scale_factor : float, optional
        The scale factor with respect to the database unit. The default is 1e3.

    Returns
    -------
    None.

    """
    fname_out = Path(fname)
    cell = cv.cell

    merge_processor = pya.ShapeProcessor()
    dbu = ly.dbu / scale_factor
    print(dbu)
    bbox = cell.bbox()
    canvas_tr = pya.CplxTrans(1.0, 0.0, True,
                              pya.DPoint(-bbox.left, bbox.top))

    # copy the cell content to new temporal layout
    layoutTMP = pya.Layout.new()
    cellTMP = layoutTMP.cell(layoutTMP.add_cell("MainSymbol"))
    cellTMP.copy_tree(cell)
    cellTMP.flatten(-1, True)

    for layer_idx in layoutTMP.layer_indices():
        print("layerIDX-copy ", layer_idx)
        if layoutTMP.top_cell().shapes(layer_idx).is_empty() is False:
            merge_processor.merge(layoutTMP,
                                  cellTMP,
                                  layer_idx,
                                  cellTMP.shapes(layer_idx),
                                  True, 0, True, True)

    # create new Inkscape compatible SVG file
    dwg = svgwrite.Drawing(fname_out, profile='full',
                           size=(str(bbox.width() * dbu) + 'mm',
                                 str(bbox.height() * dbu) + 'mm'))
    inkscape = Inkscape(dwg)

    for idx, n_info in enumerate(layoutTMP.layer_infos()):

        # find layer color TODO: this needs some rework
        iter_layers = lv.begin_layers()
        while not(iter_layers.at_end()):
            lp = iter_layers.current()
            if lp.layer_index() == idx \
                    and lp.cellview() == lv.active_cellview_index():
                RGB = unpack("B" * 4, pack("I", lp.fill_color))
                RGB = RGB[:-1]
                break
            iter_layers.next()

        iter_sc = cellTMP.begin_shapes_rec(idx)

        # create SVG layer for each internal layer
        # svg_layer = inkscape.layer(label=n_info.name.replace("*", " "),
        #                            locked=False)

        # dwg.add(svg_layer)

        while not(iter_sc.at_end()):
            s = iter_sc.shape()

            if s.is_polygon() or s.is_box() or s.is_path():
                pp = s.polygon.transformed(canvas_tr)

                tmp_coordinates = []
                for npp in pp.each_point_hull():
                    tmp_coordinates.append([npp.x * dbu * 3.77952,  # px to mm
                                            npp.y * dbu * 3.77952])

                dwg.add(dwg.polyline(points=tmp_coordinates,
                                     fill="rgb{}".format(RGB)))
            iter_sc.next()
    dwg.save()


if __name__ == "__main__":
    app = pya.Application.instance()
    mw = app.main_window()
    lv = mw.current_view()
    ly = lv.active_cellview().layout()

    if lv is None:
        raise Exception("No view selected!")

    cv = pya.CellView.active()
    layout_cell_to_svg(cv, ly, lv, fname='/tmp/test.svg')
