import numpy as np
from matplotlib import pyplot as plt
from bokeh.plotting import figure
from bokeh.palettes import Sunset11
from bokeh.models import ColumnDataSource
import cv2

#   _____       _____  _       _   
#  |  __ \     |  __ \| |     | |  
#  | |__) |   _| |__) | | ___ | |_ 
#  |  ___/ | | |  ___/| |/ _ \| __|
#  | |   | |_| | |    | | (_) | |_ 
#  |_|    \__, |_|    |_|\___/ \__|
#          __/ |                   
#         |___/                    

##################################################################
#### Sliding Similarity Matrix Figure: do and update

def doMtrxFig(axes, odom_in):
    plt.sca(axes)
    mtrx_image = np.zeros((len(odom_in['position']['x']), len(odom_in['position']['x'])))
    mtrx_handle = axes.imshow(mtrx_image)
    axes.set(xlabel='Query Frame', ylabel='Reference Frame')

    return {'img': mtrx_image, 'handle': mtrx_handle}

def updateMtrxFig(mInd, tInd, dvc, odom_in, fig_handles):
    img_new = np.delete(fig_handles['img'], 0, 1) # delete first column (oldest query)
    fig_handles['img'] = np.concatenate((img_new, np.array(dvc)), 1)
    fig_handles['handle'].set_data(fig_handles['img']) # TODO: access directly.
    fig_handles['handle'].autoscale() # https://stackoverflow.com/questions/10970492/matplotlib-no-effect-of-set-data-in-imshow-for-the-plot

##################################################################
#### Distance Vector Figure: do and update

def doDVecFig(axes, odom_in):
# Set up distance vector figure
# https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.sca(axes) # distance vector
    dist_vector = plt.plot([], [], 'k-')[0] # distance vector
    lowest_dist = plt.plot([], [], 'ro', markersize=7)[0] # matched image (lowest distance)
    actual_dist = plt.plot([], [], 'mo', markersize=7)[0] # true image (correct match)
    axes.set(xlabel='Index', ylabel='Distance')
    axes.legend(["Image Distances", "Selected", "True"])
    axes.set_xlim(0, len(odom_in['position']['x']))
    axes.set_ylim(0, 1.2)

    return {'axes': axes, 'dis': dist_vector, 'low': lowest_dist, 'act': actual_dist}

def updateDVecFig(mInd, tInd, dvc, odom_in, fig_handles):
# Update DVec figure with new data (match->mInd, true->tInd)
# update (overwrite) visualisation with new data:

    # overwrite with new distance vector / image distance:
    max_val = max(dvc[:])
    fig_handles['dis'].set_xdata(range(len(dvc)))
    fig_handles['dis'].set_ydata(dvc/max_val)
    # overwrite with new lowest match:
    fig_handles['low'].set_xdata(mInd)
    fig_handles['low'].set_ydata(dvc[mInd]/max_val)
    # overwrite with new truth value:
    fig_handles['act'].set_xdata(tInd)
    fig_handles['act'].set_ydata(dvc[tInd]/max_val)

##################################################################
#### Odometry Figure: do and update

def doOdomFig(axes, odom_in):
# Set up odometry figure

    plt.sca(axes)
    ref_plotted = plt.plot(odom_in['position']['x'], odom_in['position']['y'], 'b-')[0]
    mat_plotted = plt.plot([], [], 'r+', markersize=6)[0] # Match values: init as empty
    tru_plotted = plt.plot([], [], 'gx', markersize=4)[0] # True value: init as empty

    axes.set(xlabel='X-Axis', ylabel='Y-Axis')
    axes.legend(["Reference", "Match", "True"])
    axes.set_aspect('equal')

    return {'axes': axes, 'ref': ref_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFig(mInd, tInd, dvc, odom_in, fig_handles):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Only display last 'queries_keep' number of points
    num_queries = len(list(fig_handles['tru'].get_xdata()))
    queries_keep = 10
    start_ind = num_queries - queries_keep + 1
    if num_queries < queries_keep:
        start_ind = 0
        
    ## odometry plot:
    # Append new value for "match" (what it matched the image to)
    fig_handles['mat'].set_xdata(np.append(fig_handles['mat'].get_xdata()[start_ind:num_queries], odom_in['position']['x'][mInd]))
    fig_handles['mat'].set_ydata(np.append(fig_handles['mat'].get_ydata()[start_ind:num_queries], odom_in['position']['y'][mInd]))
    # Append new value for "true" (what it should be from the robot odom)
    fig_handles['tru'].set_xdata(np.append(fig_handles['tru'].get_xdata()[start_ind:num_queries], odom_in['position']['x'][tInd]))
    fig_handles['tru'].set_ydata(np.append(fig_handles['tru'].get_ydata()[start_ind:num_queries], odom_in['position']['y'][tInd]))

#   ____        _        _     
#  |  _ \      | |      | |    
#  | |_) | ___ | | _____| |__  
#  |  _ < / _ \| |/ / _ \ '_ \ 
#  | |_) | (_) |   <  __/ | | |
#  |____/ \___/|_|\_\___|_| |_|
    
def disable_toolbar(fig, interact=False):
    # hide toolbar and disable plot interaction
    fig.toolbar_location = None
    if not interact:
        fig.toolbar.active_drag = None
        fig.toolbar.active_scroll = None
        fig.toolbar.active_tap = None
    return fig

def get_contour_data(X, Y, Z, levels):
    cs = plt.contour(X, Y, Z, levels)
    xs = []
    ys = []
    xt = []
    yt = []
    col = []
    text = []
    isolevelid = 0
    for isolevel in cs.collections:
        isocol = isolevel.get_color()[0]
        thecol = 3 * [None]
        theiso = str(cs.get_array()[isolevelid])
        isolevelid += 1
        for i in range(3):
            thecol[i] = int(255 * isocol[i])
        thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])

        for path in isolevel.get_paths():
            v = path.vertices
            x = v[:, 0]
            y = v[:, 1]
            xs.append(x.tolist())
            ys.append(y.tolist())
            try:
                xt.append(x[len(x) / 2])
                yt.append(y[len(y) / 2])
            except:
                xt.append(5)
                yt.append(5)
            text.append(theiso)
            col.append(thecol)

    source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': col,'xt':xt,'yt':yt,'text':text})
    return source

##################################################################
#### Contour Figure: do and update

def doCntrFigBokeh(nmrc, odom_in, img_mat=None):
# Set up contour figure
    # TODO: add lines https://stackoverflow.com/questions/33533047/how-to-make-a-contour-plot-in-python-using-bokeh-or-other-libs
    # fig_cntr    = figure(title="SVM Contour", width=250, height=250, \
    #                         x_axis_label = 'VA Factor', y_axis_label = 'Grad Factor', \
    #                         x_range = (0, 500), y_range = (0, 500))
    # fig_cntr    = disable_toolbar(fig_cntr)

    # if (img_mat is None):
    #     img_mat     = np.random.rand(500,500)
    # x, y        = np.meshgrid(np.arange(img_mat.shape[1]), np.arange(img_mat.shape[0]))
    # levels      = np.linspace(0,1,11) # num levels = 11 (between 0 and 1)
    # source      = get_contour_data(x, y, img_mat, levels)

    # img_plotted = fig_cntr.image([img_mat])
    # mln_plotted = fig_cntr.multi_line(xs='xs', ys='ys', line_color='line_color', source=source)
    # txt_plotted = fig_cntr.text(x='xt',y='yt',text='text',source=source,text_baseline='middle',text_align='center')

    # # https://docs.bokeh.org/en/dev-3.0/docs/user_guide/specialized/contour.html
    # #img_plotted = fig_cntr.contour(x, y, img_mat, levels=levels, fill_color=Sunset11, line_color="black")
    # #colorbar = cntr_plot.construct_color_bar()
    # #fig_cntr.add_layout(colorbar, "right")
    # fig_cntr.axis.visible = False

    # return {'fig': fig_cntr, 'img': img_plotted, 'mln': mln_plotted, 'txt': txt_plotted, 'mat': img_mat, 'ud': True, 'x': x, 'y': y, 'levels': levels}
    pass

def updateCntrFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
    pass

##################################################################
#### Distance Vector Figure: do and update

def doDVecFigBokeh(nmrc, odom_in):
# Set up distance vector figure

    fig_dvec    = figure(title="Distance Vector", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Distance', \
                            x_range = (0, len(odom_in['position']['x'])), y_range = (0, 1.2))
    fig_dvec    = disable_toolbar(fig_dvec)
    dvc_plotted = fig_dvec.line([], [], color="black", legend_label="Distances") # distance vector
    mat_plotted = fig_dvec.circle([], [], color="red", size=7, legend_label="Selected") # matched image (lowest distance)
    tru_plotted = fig_dvec.circle([], [], color="magenta", size=7, legend_label="True") # true image (correct match)

    fig_dvec.legend.location=(100, 140)
    fig_dvec.legend.orientation='horizontal'
    fig_dvec.legend.border_line_alpha=0
    fig_dvec.legend.background_fill_alpha=0

    return {'fig': fig_dvec, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateDVecFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    max_val = max(dvc[:])
    nmrc.fig_dvec_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': dvc/max_val}
    nmrc.fig_dvec_handles['mat'].data_source.data = {'x': [mInd], 'y': [dvc[mInd]/max_val]}
    nmrc.fig_dvec_handles['tru'].data_source.data = {'x': [tInd], 'y': [dvc[tInd]/max_val]}

##################################################################
#### Odometry Figure: do and update

def doOdomFigBokeh(nmrc, odom_in):
# Set up odometry figure
    fig_odom    = figure(title="Odometries", width=500, height=250, \
                            x_axis_label = 'X-Axis', y_axis_label = 'Y-Axis', \
                            match_aspect = True, aspect_ratio = "auto")
    fig_odom    = disable_toolbar(fig_odom)
    
    ref_plotted = fig_odom.line(   x=odom_in['position']['x'], y=odom_in['position']['y'], color="blue",   legend_label="Reference")
    mat_plotted = fig_odom.cross(  x=[], y=[], color="red",    legend_label="Match", size=12)
    tru_plotted = fig_odom.x(      x=[], y=[], color="green",  legend_label="True", size=8)

    fig_odom.legend.location= (100, 70)
    fig_odom.legend.orientation='horizontal'
    fig_odom.legend.border_line_alpha=0
    fig_odom.legend.background_fill_alpha=0

    return {'fig': fig_odom, 'ref': ref_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Stream/append new value for "match" (estimate) and "true" (correct) odometry:
    new_mat_data = dict()
    new_tru_data = dict()
    new_mat_data['x'] = nmrc.fig_odom_handles['mat'].data_source.data['x'] + [odom_in['position']['x'][mInd]]
    new_mat_data['y'] = nmrc.fig_odom_handles['mat'].data_source.data['y'] + [odom_in['position']['y'][mInd]]
    new_tru_data['x'] = nmrc.fig_odom_handles['tru'].data_source.data['x'] + [odom_in['position']['x'][tInd]]
    new_tru_data['y'] = nmrc.fig_odom_handles['tru'].data_source.data['y'] + [odom_in['position']['y'][tInd]]
    #nmrc.fig_odom_handles['mat'].data_source.data = new_mat_data
    #nmrc.fig_odom_handles['tru'].data_source.data = new_tru_data
    nmrc.fig_odom_handles['mat'].data_source.stream(new_mat_data, rollover=10)
    nmrc.fig_odom_handles['tru'].data_source.stream(new_tru_data, rollover=10)