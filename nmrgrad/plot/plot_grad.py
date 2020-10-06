#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various plotting tools to visualize the gradient field.

"""
import nmrgrad
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

def cm2inch(*tupl):
    """
    Convert cm tuple values to inch.

    Parameters
    ----------
    *tupl : tuple of floats
        The centimetre values.

    Returns
    -------
    tuple
        The input tuple converted to inch.

    """
    inch = 2.54

    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def subplot_adjust_mm(figure, figsize=None, left=8, bottom=10, right=5, top=5):
    if figsize != None:
        figure.set_size_inches( cm2inch(figsize) )
    xsize, ysize = figure.get_size_inches()
    plt.subplots_adjust(left = (left/25.4)/xsize,
                        bottom = (bottom/25.4)/ysize,
                        right = 1 - (right/25.4)/xsize,
                        top = 1 - (top/25.4)/ysize)

def cond_calc_bfield(conductors, axis_x="x", axis_y="y",
                     axis_const="z", ndots=100, roi=None):
    """
    Caclulates the magnetic flux density from a conductor set as the input \
    for the contour plot.

    Parameters
    ----------
    conductors : list
        list of list with element is a list of control points on the conductor \
        and the second is a dict with parameters such es current.
    axis_x : str, optional
        DESCRIPTION. The default is "x".
    axis_y : str, optional
        DESCRIPTION. The default is "y".
    axis_const : str, optional
        DESCRIPTION. The default is "z".
    ndots : TYPE, optional
        DESCRIPTION. The default is 100.
    roi : ndarray, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    x_grid : mgrid
        The coordinates on the with respect to the x-axis of the plotting plane.
    y_grid : mgrid
        The coordinates on the with respect to the y-axis of the plotting plane.
    b_z : mgrid
        The b-bield in the same format as the x_grid.

    """
    if roi is None:
        roi = np.array([[-3.0, -2.0, -1.0],
                        [3.0, 2.0, 1.0]]) * 1e-3

    x_grid, y_grid, r_p = nmrgrad.biot_savart.mgrid_2d_roi(axis_x, axis_y, axis_const,
                                                           roi=roi, ndots=ndots)

    b_field = np.zeros_like(r_p)

    for cond in conductors:
        if "current" in cond[1]: # sets the current for the conductor
            c_current = cond[1]["current"]
        else:
            c_current = 1.0

        b_field += nmrgrad.biot_savart.biot_line_integral(cond[0] * 1e-3,
                                                          r_p, c_current)

    b_z = b_field[:, 2]
    b_z.shape = x_grid.shape

    return x_grid, y_grid, b_z

def b_field_contour_plot(x_grid, y_grid, b_z, fig=None, ax=None,
                         x_label="", y_label="", cmap=None):
    """
    Plots the magnetic flux density of the z component via a contour plot.

    Parameters
    ----------
    x_grid : TYPE
        DESCRIPTION.
    y_grid : TYPE
        DESCRIPTION.
    b_z : TYPE
        DESCRIPTION.
    fig : matplotlib.figure, optional
        Matplotlib top level container for all plot elements. The default is None.
    ax : matplotlib.axes._base._AxesBase, optional
        The the matplotlib Axes object. The default is None.
    x_label : str, optional
        The x axis label text. The default is "".
    y_label : str, optional
        The y axis label text. The default is "".
    cmap : matplotlib.colors.Colormap, optional
        The colormap object to be used. The default is None.

    Returns
    -------
    fig : matplotib figure
        DESCRIPTION.
    ax : matplotlib.axes._base._AxesBase
        The the matplotlib Axes object.

    """
    if fig is None or ax is None:
        plt.clf()
        plt.close()
        fig, ax = plt.subplots(1)
        subplots_adjust(left=0.17, right=0.85, top=.97, bottom=0.19)

    if cmap is None:
        cmap = plt.get_cmap('hot')

    plt.xlabel(x_label + u" (mm)")
    plt.ylabel(y_label + u" (mm)")

    cs = ax.contour(x_grid * 1e3, y_grid * 1e3, b_z,
                    linewidths=0.5, colors='k', antialiased=True)

    cc = ax.contourf(x_grid * 1e3, y_grid * 1e3, b_z,
                     extend="both", cmap=cmap, antialiased=True)

    return fig, ax

def fmt(value, pos=None):
    """
    Format the contour line labels in percent.

    Parameters
    ----------
    xvalue : float
        DESCRIPTION.
    pos : None
        Not used but mandatory for the use with mpl.

    Returns
    -------
    str
        Formated output in percent using latex siunitx formating.

    """
    return r'\SI{'+ '{:.0f}'.format(value) + r'}{\percent}'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n_colors=100):
    """
    Create a new color map by limiting the color range from a input color map.

    See also:
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

    Parameters
    ----------
    cmap : mpl cmap
        matplotlib colormap object.
    minval : float, optional
        The lower value to start with the color map. The default is 0.0.
    maxval : float, optional
        The upper value to end the color map. The default is 1.0.
    n_colors : int, optional
        Number of colour steps generated. The default is 100.

    Returns
    -------
    new_cmap : mpl cmap
        The cmap limited by minval and maxval.

    """
    new_cmap = colors.LinearSegmentedColormap.from_list( \
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n_colors)))

    return new_cmap

def linearity_contour_plot(conductors, grad_direction="y", fig=None, ax=None,
                           only_cbar=False, ndots=200,
                           xlim=[-1.3, 1.3], ylim=[-.43, .43], roi=None,
                           x_axis="x", y_axis="y", axis_const="z"):
    """
    Create the linearity contour plot in the respective axis.

    Parameters
    ----------
    conductors : list
        list of list with element is a list of control points on the conductor \
        and the second is a dict with parameters such es current.
    grad_direction : TYPE, optional
        DESCRIPTION. The default is "y".
    fig : matplotlib.figure
        Matplotlib top level container for all plot elements. The default is None.
    ax : matplotlib.axes._base._AxesBase
        The the matplotlib Axes object. The default is None.
    xlim : list, optional
        The x limits of the plot. The default is [-1.3, 1.3].
    ylim : list, optional
        The y limits of the plot.. The default is [-.43, .43].
    roi : ndarray
        The limits of the cubic region of interest for x, y and z.
    x_axis : str, optional
        The roi-direction which is plotted for the x-axis direction.
        The default is "x".
    y_axis : str, optional
        The roi-direction which is plotted for the x-axis direction.
        The default is "y".
    axis_const : str, optional
        The roi direction which is considered constant. The default is "z".

    Returns
    -------
    fig : matplotlib.figure
        Matplotlib top level container for all plot elements.
    ax : matplotlib.axes._base._AxesBase
        The the matplotlib Axes object.
    g_slope_desired : float
        The desired gradient efficiency in Tesla / (metre x Ampere)

    """
    if roi is None:
        roi = np.array([[-4.0, -3.0, -3.0],
                        [4.0, 3.0, 3.0]]) * 1e-3

    b_z_dot, _ = grad_centre_line(conductors, roi, ndots=50,
                                  axis_x=(grad_direction == "x"),
                                  axis_y=(grad_direction == "y"),
                                  axis_z=(grad_direction == "z"))

    idx_centre = int(0.5 * len(b_z_dot))
    g_slope_desired = b_z_dot[idx_centre-2:idx_centre+2].mean()

    x_grid, y_grid, b_z = cond_calc_bfield(conductors, axis_x=x_axis, axis_y=y_axis,
                                           axis_const=axis_const, ndots=ndots, roi=roi)

    if fig is None:
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        figlength = (15 - 1.7 - 0.25) * 0.5 - 0.25
        subplot_adjust_mm(fig, figsize=(figlength, figlength * 4/5),
                          left=12, right=1, top=1, bottom=9)

    g_desired = np.ones_like(b_z) * g_slope_desired

    if grad_direction == "x":
        dist = (x_grid[:, 0][1:] - x_grid[:, 0][:-1]).mean()
        g_achieved = np.gradient(b_z)[0] / dist

    elif grad_direction == "y":
        dist = (y_grid[0][1:] - y_grid[0][:-1]).mean()
        g_achieved = np.gradient(b_z)[1] / dist

    elif grad_direction == "z":
        dist = (y_grid[0][1:] - y_grid[0][:-1]).mean()
        g_achieved = np.gradient(b_z)[1] / dist

    g_error = np.abs(g_achieved / g_desired -1) * 100

    # create the contour plot
    cmap = plt.get_cmap('gray')
    cmap_gs = truncate_colormap(cmap, .95, 0.4)

    levels = ([2, 3, 5, 10, 20, 50])
    cc = ax.contourf(x_grid * 1e3, y_grid * 1e3, g_error,
                     levels=levels + [100], extend="max", cmap=cmap_gs,
                     antialiased=True)
    cs = ax.contour(x_grid * 1e3, y_grid * 1e3, g_error,
                    levels=levels, linewidths=0.5, colors='k', antialiased=True)
    clim = ax.contour(x_grid * 1e3, y_grid * 1e3, g_achieved,
                      linewidths=2, colors="k", levels=[0.0, np.inf])

    ax.clabel(cs, inline=1, fontsize=4, fmt=ticker.FuncFormatter(fmt))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_ylabel(y_axis + r" (\si{\milli\metre})")
    ax.set_xlabel(x_axis + r" (\si{\milli\metre})")

    if only_cbar is True:
        plt.clf()
        plt.close()
        fig, ax = plt.subplots()
        subplot_adjust_mm(fig, figsize=(1.65, 5), left=-45, right=8, top=1, bottom=9)

        cbar = plt.colorbar(cc, orientation='vertical', shrink=1)
        cbar.add_lines(cs)
        # cbar = ax.colorbar(CS, orientation='vertical', shrink=1)

        cbar.set_label(r"$\left|\dfrac{\boldsymbol{G}_{" + grad_direction \
        + r",\textnormal{achieved}}}{\boldsymbol{G}_{" + grad_direction \
        + r",\textnormal{desired}}}-1 \right|\, \left(\%\right)$")

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

        ax.remove()

    return fig, ax, g_slope_desired

def grad_centre_line(conductors, roi,
                     axis_x=False, axis_y=False, axis_z=False, ndots=100):
    """
    Get the gradient efficiency for a line within the centre of the \
    region of interest.

    Only one of the axis_x/y/z should be set to True.

    Parameters
    ----------
    conductors : list(ndarray, dict)
        The first element is a list of the dicretised points on the conductor.
        The second element is a dict witch parameters for plotting.
    roi : ndarray
        The limits of the cubic region of interest for x, y and z.
    axis_x : bool, optional
        If set to true, the x-axis is used to calculate the filed points.
        The default is False.
    axis_y : bool, optional
        If set to true, the y-axis is used to calculate the filed points.
        The default is False.
    axis_z : bool, optional
        If set to true, the z-axis is used to calculate the filed points.
        The default is False.
    ndots : int, optional
        Number of field points. The default is 100.

    Returns
    -------
    grad_slope : float
        The gradient slope or efficiency in Tesla / (metre Amperes).

    """
    x_pts = np.linspace(roi[0][0], roi[1][0], ndots) * axis_x
    y_pts = np.linspace(roi[0][1], roi[1][1], ndots) * axis_y
    z_pts = np.linspace(roi[0][2], roi[1][2], ndots) * axis_z

    r_p = np.array([x_pts, y_pts, z_pts]).swapaxes(0, 1)

    b_field = np.zeros_like(r_p)

    for cond in conductors:
        # sets the current for the conductor
        if "current" in cond[1]:
            c_current = cond[1]["current"]
        else:
            c_current = 1.0
        b_field += nmrgrad.biot_savart.biot_line_integral(cond[0] * 1e-3,
                                                          r_p, c_current)

    if axis_x:
        d_x = np.gradient(x_pts)
        pos = x_pts
    elif axis_y:
        d_x = np.gradient(y_pts)
        pos = y_pts
    elif axis_z:
        d_x = np.gradient(z_pts)
        pos = z_pts

    # get the slope of the b_z component
    b_z_dot = np.gradient(b_field[:, 2]) / d_x
    return b_z_dot, pos

def cond_plot_3d(conductors, fig=None, ax=None,
                 intercon_cond_segemnts=False,
                 no_connect=[], start_end=False):
    """
    Plot the conductor trajectory in 3D.

    To plot a 3D example to illustrate the conductors.

    current - mainly -1/+1 A
    color = red
    alpha = 1.0
    zorder = 0

    Parameters
    ----------
    conductors : list
        list of list with element is a list of control points on the conductor \
        and the second is a dict with parameters such es current.
    fig : matplotlib.figure
        Matplotlib top level container for all plot elements. The default is None.
    ax : matplotlib.axes._base._AxesBase
        The the matplotlib Axes object. The default is None.
    intercon_cond_segemnts : bool, optional
        If True, connect the individual conductor elements to a single loop.
        The default is False.
    no_connect : list, optional
        Index of conductor elements which must not be conected together.
        The default is [].
    start_end : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : matplotlib.figure
        Matplotlib top level container for all plot elements.
    ax : matplotlib.axes._base._AxesBase
        The the matplotlib Axes object.

    """
    if fig is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    # cond_pts = [i[0] for i in conductors]
    # x_min = min(cond_pts, key=lambda x: min(x[:, 0]))[:, 0].min()
    # x_max = max(cond_pts, key=lambda x: max(x[:, 0]))[:, 0].max()
    # y_min = min(cond_pts, key=lambda y: min(y[:, 1]))[:, 1].min()
    # y_max = max(cond_pts, key=lambda y: max(y[:, 1]))[:, 1].max()
    # z_min = min(cond_pts, key=lambda z: min(z[:, 2]))[:, 2].min()
    # z_max = max(cond_pts, key=lambda z: max(z[:, 2]))[:, 2].max()

    # roi = np.array([[x_min, y_min, z_min],
    #                 [x_max, y_max, z_max]])

    # if contourf is True:
    #     lvs = np.around(np.linspace(self.Bmin, self.Bmax, 30), decimals=5)
    #     X, Y, B_z = self.cond_calc_bfield ( self.conductors, axis_x="x", axis_y="y",
    # axis_const="z", roi=roi)
    #     cset = ax.contourf(X, Y, np.clip(np.nan_to_num(B_z), self.Bmin, self.Bmax),
    #                        levels=lvs,
    #                        zdir='z',      # direction to project the contour
    #                        offset=z_min*1.06,  # how "far" render the contour map
    #                        cmap=self.cmap, zorder=0)
    #     XX, ZZ, B_zYY = self.cond_calc_bfield ( self.conductors, axis_x="x",
    # axis_y="z", axis_const="y", roi=roi)
    #     cset = ax.contourf(XX, np.clip(np.nan_to_num(B_zYY), self.Bmin, self.Bmax), ZZ,
    #                        levels=lvs,
    #                        zdir='y', # direction to project the contour
    #                        offset=y_min*1.06, # how "far" render the contour map
    #                        cmap=self.cmap, zorder=0)
    #     YY, ZZ, B_zXX = self.cond_calc_bfield ( self.conductors, axis_x="y",
    # axis_y="z", axis_const="x", roi=roi)
    #     cset = ax.contourf(np.clip(np.nan_to_num(B_zXX), self.Bmin, self.Bmax), YY, ZZ,
    #                        levels=lvs,
    #                        zdir='x',      # direction to project the contour
    #                        offset=x_max*1.06,  # how "far" render the contour map
    #                        cmap=self.cmap, zorder=0)
    #     plt.clabel(cset, inline=1,  fontsize=5)

    #     cbar = plt.colorbar(cset, orientation='vertical', shrink=1,
    # format=ticker.FuncFormatter(fmt))
    #     cbar.set_label(r"$\frac{B_z}{I} \, \left(\frac{T}{A} \times 10^{-3} \right)$")

    for cond_idx, cond_elem in enumerate(conductors):
        ## iconPrev is a function switch!
        color = "k"
        alpha = 1.0
        zorder = 0

        if "color" in cond_elem[1]:
            color = cond_elem[1]["color"]
        if "alpha" in cond_elem[1]:
            alpha = cond_elem[1]["alpha"]
        if "zorder" in cond_elem[1]:
            zorder = cond_elem[1]["zorder"]

        if (intercon_cond_segemnts is True) \
            and (cond_idx != 0) and (cond_idx not in no_connect):
            # interconnect the conductors - not valid for first conductor
            # and only if a previous conductors is present
            cond_previous = np.array([conductors[cond_idx-1][0][-1]])
            cond = np.append(cond_previous, cond_elem[0], axis=0)
        else:
            cond = cond_elem[0]
        ax.plot(cond[:, 0], cond[:, 1], cond[:, 2], color, marker="*", alpha=alpha, zorder=zorder)

    # if contourf == True and start_end == True:
    #     for nCond in self.conductors:
    #         if "current" in nCond[1]:
    #             current = nCond[1]["current"]
    #         else:
    #             current = 1.0
    #         if current >= 0.0: # positive current
    #             ax.plot([nCond[0][0][0]], [nCond[0][0][1]], [nCond[0][0][2]],
    #                     "o", markersize=5, color='red', zorder=500)
    #             ax.plot([nCond[0][-1][0]], [nCond[0][-1][1]], [nCond[0][-1][2]],
    #                     "*", markersize=5, color='blue', zorder=500)
    #         else:
    #             ax.plot([nCond[0][0][0]], [nCond[0][0][1]], [nCond[0][0][2]], "o",
    # markersize=5, color='blue', zorder=500)
    #             ax.plot([nCond[0][-1][0]], [nCond[0][-1][1]], [nCond[0][-1][2]], "*", markersize=5, color='red', zorder=500)

    plt.subplots_adjust(left=0.0, right=0.95, top=0.95, bottom=0.07)
    ax.view_init(elev=32, azim=135)
    ax.view_init(50, 0)
    ax.locator_params(nbins=5)
    ax.tick_params(pad=-2)
    ax.grid(False)
    ax.set_axisbelow(True)

    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlabel(r'x (\SI{}{\milli\metre})', labelpad=-4)
    ax.set_ylabel(r'y (\SI{}{\milli\metre})', labelpad=-4)

    ax.text(min([i[0] for i in conductors],
                key=lambda x: min(x[:, 0]))[:, 0].min() * 0.97,
            min([i[0] for i in conductors],
                key=lambda y: min(y[:, 1]))[:, 1].min() * 0.97,
            max([i[0] for i in conductors],
                key=lambda z: max(z[:, 2]))[:, 2].max() * 1.4,
            r'z (\SI{}{\milli\metre})', fontsize="x-small")

    inward_adjust = 0.4
    outward_adjust = -0.05

    ax.xaxis._axinfo['tick']['inward_factor'] = inward_adjust
    ax.xaxis._axinfo['tick']['outward_factor'] = outward_adjust
    ax.yaxis._axinfo['tick']['inward_factor'] = inward_adjust
    ax.yaxis._axinfo['tick']['outward_factor'] = outward_adjust
    ax.zaxis._axinfo['tick']['inward_factor'] = inward_adjust
    ax.zaxis._axinfo['tick']['outward_factor'] = outward_adjust

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    return fig, ax
