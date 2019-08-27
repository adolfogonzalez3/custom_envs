'''Utilities for plotting.'''
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def plot_sequence(axes, sequence, **kwargs):
    '''
    Plot a sequence on a matplotlib axis object.

    :param axis: (matplotlib.Axes) An Axes object used to plot the sequence.
    :param sequence: (Sequence) A sequence of numbers to plot.
    :param **kwargs: Keywords passed to the Axes.plot method.
    '''
    sequence = np.asarray(sequence)
    assert sequence.ndim == 1
    axes.plot(range(sequence.size), sequence, **kwargs)


def fill_between(axes, means, stds, **kwargs):
    '''
    Create a shaded region on a plot using mean and std.

    :param axes: (matplotlib.Axes) An Axes object used to plot the sequence.
    :param mean: (Sequence) A sequence of means to use to produce the shaded
                            region. The ith mean in means corresponds to the
                            ith standard deviation in stds.
    :param std: (Sequence) A sequence of standard deviations to use to produce
                           the shaded region. The ith standard deviation in
                           stds corresponds to the The ith mean in means.
    :param **kwargs: Keywords passed to the Axes.fill_between method.
    '''
    means = np.asarray(means)
    stds = np.asarray(stds)
    assert means.ndim == 1
    assert stds.ndim == 1
    axes.fill_between(range(means.size), means + stds, means - stds, **kwargs)


def add_legend(axes):
    '''
    Add a legend to a figure.

    :param axes: (matplotlib.Axes) An Axes object that will have a legend
                                   added.
    '''
    chart_box = axes.get_position()
    axes.set_position([chart_box.x0, chart_box.y0, chart_box.width*0.8,
                       chart_box.height])
    axes.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True,
                ncol=1)


def set_attributes(axes, attr):
    '''
    Set matplotlib plot attributes.

    :param axes: (matplotlib.Axes) An Axes object that will have a legend
                                   added.
    :param attr: (dict) A dictionary containing the pyplot attributes to set.
    '''
    axes.set_xlabel(attr.get('xlabel'))
    axes.set_ylabel(attr.get('ylabel'))
    axes.set_title(attr.get('title'))
    try:
        axes.set_zlabel(attr.get('zlabel'))
        axes.set_zlim(attr.get('zmin'), attr.get('zmax'))
    except AttributeError:
        pass


def plot_results(axes, dataframe, groupby, label=None):
    '''
    Plot results on multiple axes given a dataframe.


    '''
    figures = defaultdict(plt.figure())
    axes = {}
    groupby = {groupby} if isinstance(groupby, str) else set(groupby)
    grouped = dataframe.groupby(groupby)
    mean_df = grouped.mean()
    std_df = grouped.std()
    columns = set(mean_df.columns) & set(axes.keys()) - groupby
    for name in columns:
        axes[name] = figures[name].add_subplot(111)
        plot_sequence(axes[name], mean_df[name], label=label)
        fill_between(axes[name], mean_df[name], std_df[name],
                     alpha=0.1, label=label)
