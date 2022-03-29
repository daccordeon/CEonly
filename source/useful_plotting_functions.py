"""James Gardner, March 2022"""
import matplotlib.pyplot as plt

def force_log_grid(ax, log_axis='both', color='gainsboro', preserve_tick_labels=True, **kwargs):
    """attempts to force matplotlib to display all major and minor grid lines on both axes, if the plot would be too dense, then it will fail to do so
    for preserve_tick_labels to work, the y/xlim must be set beforehand
    to-do: fix preserve_tick_labels, just calling fig.canvas.draw() before doesn't fix it"""
#     if preserve_tick_labels:
#         # this could be re-written with a decorator
#         ax.set(ylim=ax.get_ylim(), xlim=ax.get_xlim())
#         yticklabels, xticklabels = ax.yaxis.get_ticklabels(), ax.xaxis.get_ticklabels()
#         #print(ax.get_ylim(), ax.get_xlim(), yticklabels, xticklabels)
#         force_log_grid(ax, preserve_tick_labels=False, color=color, **kwargs)
#         ax.yaxis.set_ticklabels(yticklabels)
#         ax.xaxis.set_ticklabels(xticklabels)
#     else:
    ax.grid(which='both', axis='both', color=color, **kwargs)
    # minorticks_on must be called before the locators by experimentation
    ax.minorticks_on()     
    if log_axis in ('both', 'x'):
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
    if log_axis in ('both', 'y'):
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
