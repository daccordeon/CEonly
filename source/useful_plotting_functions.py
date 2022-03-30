"""James Gardner, March 2022"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from constants import *

def force_log_grid(ax, log_axis='both', color='gainsboro', preserve_tick_labels=True, **kwargs):
    """attempts to force matplotlib to display all major and minor grid lines on both axes, if the plot would be too dense, then it will fail to do so
    for preserve_tick_labels to work, the y/xlim must be set beforehand
    to-do: fix preserve_tick_labels, just calling fig.canvas.draw() before doesn't fix it"""
#     if preserve_tick_labels:
#         # this could be re-written with a decorator
#         ax.set(ylim=ax.get_ylim(), xlim=ax.get_xlim())
#         yticklabels, xticklabels = ax.yaxis.get_ticklabels(), ax.xaxis.get_ticklabels()
#         #print(ax.get_ylim(), ax.get_xlim(), yticklabels, xticklabels)
#         force_log_grid(ax, log_axis=log_axis, preserve_tick_labels=False, color=color, **kwargs)
#         ax.yaxis.set_ticklabels(yticklabels)
#         ax.xaxis.set_ticklabels(xticklabels)
#     else:
    ax.grid(which='both', axis='both', color=color, **kwargs)
    # minorticks_on must be called before the locators by experimentation
    ax.minorticks_on() 
    # to-do: fix this, current doesn't force loglog grid
#     if log_axis in ('both', 'x'):
#         ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
#         ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
#     if log_axis in ('both', 'y'):
#         ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
#         ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
#     # to-do: stop overcrowding of ticklabels, currently not working
#     print(len(ax.yaxis.get_ticklabels()), ax.get_yticks()[::2], len(ax.xaxis.get_ticklabels()), ax.get_xticks()[::2])
#     if len(ax.yaxis.get_ticklabels()) > 5:
#         ax.set(yticks=ax.get_yticks()[::2], ylim=ax.get_ylim())
#     if len(ax.xaxis.get_ticklabels()) > 5:
#         ax.set(xticks=ax.get_xticks()[::2], xlim=ax.get_xlim())

def add_SNR_contour_legend(ax): 
    """adds a contour legend a la Kuns+ 2020 to the given axes"""
    rect_pos = (0.05, 0.8)
    rect_width = 0.06
    rect_height = 0.1
    rect_top = (rect_pos[0], rect_pos[1] + rect_height)

    ax.add_patch(Rectangle((rect_pos[0] - 0.005, rect_pos[1]), rect_width + 2*0.005, 0.01, color='k', transform=ax.transAxes, zorder=10))
    ax.add_patch(Rectangle(rect_pos, rect_width, rect_height, color='darkgrey', transform=ax.transAxes, zorder=9))
    ax.add_patch(Rectangle(rect_top, rect_width, 0.001, color='k', transform=ax.transAxes, zorder=10))
    ax.text(rect_pos[0] + rect_width + 0.015, rect_pos[1], f'SNR>{SNR_THRESHOLD_HI}', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
    ax.text(rect_top[0] + rect_width + 0.015, rect_top[1], f'SNR>{SNR_THRESHOLD_LO}', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')            
        