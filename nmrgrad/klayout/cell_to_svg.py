from pathlib import Path
import nmrgrad
import xmltodict
import numpy as np
import svgwrite
import os
import numpy as np

try:
    import pya
except ImportError:
    # without klayout gui
    import klayout.db as pya


lay = pya.Layout()
top = lay.create_cell("TOP")
a = lay.layer(1, 0)
b = lay.layer(2, 0)

para = {"text":  "Hello world", "mag": 100000, "layer": lay.layer(1, 0)}
pcell = lay.create_cell("TEXT", "Basic", para)
trans = pya.DCplxTrans.new(1, 0, False, 0, 0)
mycell = top.insert(pya.DCellInstArray.new(pcell.cell_index(), trans))

lay.write("/tmp/test.gds")
