
"""
    This function is meant to make a colorbar axes attached to an axes without changing
    the size of the axes.
"""
def addAxesNextToAxes(
    fig,
    ax,
    side,
    thickness=0.05,
    spacing=0.05,
    flag_ratio_thickness=True,
    flag_ratio_spacing=True,
):

    figsize = fig.get_size_inches()
    pos = ax.get_position()
    
    if side in ["left", "right"]:
        fig_measure = figsize[0]
        ax_ratio_measure  = pos.width

    elif side in ["top", "bottom"]:
        fig_measure = figsize[1]
        ax_ratio_measure = pos.height

    if flag_ratio_thickness:
        thickness_ratio = ax_ratio_measure * thickness
    else:
        thickness_ratio = thickness / fig_measure

    if flag_ratio_spacing:
        spacing_ratio = ax_ratio_measure * spacing
    else:
        spacing_ratio = spacing / fig_measure


    if side == "right":

        new_pos = (
            pos.x0 + pos.width + spacing_ratio,
            pos.y0,
            thickness_ratio,
            pos.height,
        )
    
    elif side == "left":
 
        new_pos = (
            pos.x0 - thickness_ratio - spacing_ratio,
            pos.y0,
            thickness_ratio,
            pos.height,
        )
 
    elif side == "top":
 
        new_pos = (
            pos.x0,
            pos.y0 + pos.height + spacing_ratio,
            thickness_ratio,
            pos.height,
        )
        
    elif side == "bottom":
 
        new_pos = (
            pos.x0,
            pos.y0 - thickness_ratio - spacing_ratio,
            thickness_ratio,
            pos.height,
        )
 
    return fig.add_axes(new_pos)
    


def calFigParams(
    w, h, wspace, hspace,
    w_left, w_right,
    h_bottom, h_top,
    ncol, nrow,
):


    # test if w is a scalar
    if isinstance(w, (int, float, complex)):
        w = [ w, ] * ncol

    if isinstance(h, (int, float, complex)):
        h = [ h, ] * nrow


    w_avg = sum(w) / ncol
    h_avg = sum(h) / nrow

    _wspace = wspace / w_avg
    _hspace = hspace / h_avg

    W = w_left + w_right + sum(w) + (ncol - 1) * wspace
    H = h_bottom + h_top + sum(h) + (nrow - 1) * hspace

    _left   = w_left / W
    _right  = 1.0 - w_right / W
    _bottom = h_bottom / H
    _top    = 1.0 - h_top / H


    return (W, H), dict(
        left    = _left,
        right   = _right,
        bottom  = _bottom,
        top     = _top,
        wspace  = _wspace,
        hspace  = _hspace,
        width_ratios = w,
        height_ratios = h,
    )



