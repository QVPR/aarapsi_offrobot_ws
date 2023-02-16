import numpy as np
from matplotlib import pyplot as plt

##################################################################
#### Similarity Matrix Figure: do and update

def doMtrxFig(axes, odom_in):
    plt.sca(axes)
    mtrx_image = np.zeros((len(odom_in['x']), len(odom_in['x'])))
    mtrx_handle = axes.imshow(mtrx_image)
    axes.set(xlabel='Query Frame', ylabel='Reference Frame')

    return {'img': mtrx_image, 'handle': mtrx_handle}

def updateMtrxFig(mInd, tInd, dvc, odom_in, fig_handles):
    img_new = np.delete(fig_handles['img'], 0, 1) # delete first column (oldest query)
    fig_handles['img'] = np.concatenate((img_new, np.array(dvc)), 1)
    fig_handles['handle'].set_data(fig_handles['img'])
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
    axes.set_xlim(0, len(odom_in['x']))
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
    ref_plotted = plt.plot(odom_in['x'], odom_in['y'], 'b-')[0]
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
    fig_handles['mat'].set_xdata(np.append(fig_handles['mat'].get_xdata()[start_ind:num_queries], odom_in['x'][mInd]))
    fig_handles['mat'].set_ydata(np.append(fig_handles['mat'].get_ydata()[start_ind:num_queries], odom_in['y'][mInd]))
    # Append new value for "true" (what it should be from the robot odom)
    fig_handles['tru'].set_xdata(np.append(fig_handles['tru'].get_xdata()[start_ind:num_queries], odom_in['x'][tInd]))
    fig_handles['tru'].set_ydata(np.append(fig_handles['tru'].get_ydata()[start_ind:num_queries], odom_in['y'][tInd]))