import numpy as np
from matplotlib import pyplot as plt
from bokeh.models import Range1d, ColumnDataSource
from bokeh.plotting import figure, show

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

##################################################################
#### Sliding Similarity Matrix Figure: do and update

def doMtrxFigBokeh(nmrc, odom_in):
# https://docs.bokeh.org/en/2.4.3/docs/gallery/image.html
    fig_mtrx = figure(title="Similarity Matrix", width=500, height=500, \
                      x_axis_label='Query Frame', y_axis_label='Reference Frame')
    
    img_mat = np.zeros((len(odom_in['position']['x']), len(odom_in['position']['x'])))

    fig_mtrx.x_range.range_padding = 0
    fig_mtrx.y_range.range_padding = 0
    fig_mtrx.grid.grid_line_width = 0.5

    # must give a vector of image data for image parameter
    img_plotted = fig_mtrx.image(image=[img_mat], x=0, y=0, dw=10, dh=10, palette="Viridis256")

    return {'fig': fig_mtrx, 'mtrx': img_mat, 'handle': img_plotted, 'size': len(odom_in['position']['x'])**2}

def updateMtrxFigBokeh(nmrc, matchInd, trueInd, dvc, odom_in):
    
    nmrc.fig_mtrx_handles['mtrx'] = np.delete(nmrc.fig_mtrx_handles['mtrx'], 0, 1) # delete first column (oldest query)
    nmrc.fig_mtrx_handles['mtrx'] = np.concatenate((nmrc.fig_mtrx_handles['mtrx'], np.array(np.flipud(dvc))), 1)
    nmrc.fig_mtrx_handles['handle'].data_source.data = {'image': [nmrc.fig_mtrx_handles['mtrx']]}

    #print(nmrc.fig_mtrx_handles['handle'].data_source.data['image'][0].shape)
    
##################################################################
#### Distance Vector Figure: do and update

def doDVecFigBokeh(nmrc, odom_in):
# Set up distance vector figure

    fig_dvec    = figure(title="Distance Vector", width=500, height=500, \
                         x_axis_label = 'Index', y_axis_label = 'Distance', \
                         x_range = (0, len(odom_in['position']['x'])), y_range = (0, 10000))
    
    dvc_plotted = fig_dvec.line([], [], color="black", legend_label="Image Distances") # distance vector
    mat_plotted = fig_dvec.circle([], [], color="red", size=7, legend_label="Selected") # matched image (lowest distance)
    tru_plotted = fig_dvec.circle([], [], color="magenta", size=7, legend_label="True") # true image (correct match)

    return {'fig': fig_dvec, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateDVecFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)

    nmrc.fig_dvec_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': dvc}
    nmrc.fig_dvec_handles['mat'].data_source.data = {'x': [mInd], 'y': [dvc[mInd][0]]}
    nmrc.fig_dvec_handles['tru'].data_source.data = {'x': [tInd], 'y': [dvc[tInd][0]]}

##################################################################
#### Odometry Figure: do and update

def doOdomFigBokeh(nmrc, odom_in):
# Set up odometry figure
    fig_odom            = figure(title="Odometries", width=500, height=500, \
                                 x_axis_label = 'X-Axis', y_axis_label = 'Y-Axis', \
                                 match_aspect = True, aspect_ratio = "auto")
    
    ref_plotted    = fig_odom.line(   x=odom_in['position']['x'], y=odom_in['position']['y'], color="blue",   legend_label="Reference")
    mat_plotted    = fig_odom.cross(  x=[], y=[], color="red",    legend_label="Match", size=6)
    tru_plotted    = fig_odom.x(      x=[], y=[], color="green",  legend_label="True", size=4)


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