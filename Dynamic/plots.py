"""
Plot results
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
import multiprocessing
import itertools as it
import calculations as c
from scipy.stats import gaussian_kde


def from_file_to_data(inputfile):
    """
    Take a file and return an array of plottable data
    """
    with open(inputfile, 'r') as data:
        results = data.readlines()
        numresults = np.array([[eval(number) for number in result.split()[1:]] for
                               result in results])
    return numresults

def plot_data(numresults, infousename, infovaluename):
    """
    Take an array of data and return a couple of plots
    """
    plt.close()
    c = np.linspace(0, 1, num=10)
    avg_info_use = numresults[:, 0]
    max_info_use = numresults[:, 1]
    avg_info_value =  numresults[:, 3]
    max_info_value =  numresults[:, 4]
    plt.plot(c, avg_info_use, label='average')
#    plt.plot(c, max_info_use, marker='o', linestyle='--', color='r',
             #label='maximum')
    plt.xlabel('C')
    plt.ylabel('%')
    plt.title('percentage of info use')
    plt.legend()
    plt.savefig(infousename)
    plt.close()
    plt.plot(c, avg_info_value, label='average')
#    plt.plot(c, max_info_value, marker='o', linestyle='--', color='r',
#             label='maximum')
    plt.xlabel('C')
    plt.ylabel('bits')
    plt.title('maximum info use')
    plt.legend()
    plt.savefig(infovaluename)

def plot_data2(numresults, numresults2, infousename, infovaluename):
    """
    Take two arrays of data and return a couple of plots
    """
    plt.close()
    c = np.linspace(0, 1, num=10)
    avg_info_use = numresults[:, 0]
    max_info_use = numresults[:, 1]
    avg_info_value =  numresults[:, 3]
    max_info_value =  numresults[:, 4]
    avg_info_use2 = numresults2[:, 0]
    max_info_use2 = numresults2[:, 1]
    avg_info_value2 =  numresults2[:, 3]
    max_info_value2 =  numresults2[:, 4]
    plt.plot(c, avg_info_use, label='replicator')
    plt.plot(c, avg_info_use2, label='replicator-mutator')
#    plt.plot(c, max_info_use, marker='o', linestyle='--', color='r',
             #label='maximum')
    plt.xlabel('C')
    plt.ylabel('%')
    plt.title('percentage of info use')
    plt.legend()
    plt.savefig(infousename)
    plt.close()
    plt.plot(c, avg_info_value, label='replicator')
    plt.plot(c, avg_info_value2, label='replicator-mutator')
#    plt.plot(c, max_info_value, marker='o', linestyle='--', color='r',
#             label='maximum')
    plt.xlabel('C')
    plt.ylabel('bits')
    plt.title('maximum info use')
    plt.legend()
    plt.savefig(infovaluename)

def from_stats_file_to_plot(inputfile):
    """
    Take a stats file, of the sort created by stats.compress_stats_directory(),
    and return a couple of plots
    """
    infousename = ''.join([inputfile, '_infouse.pdf'])
    infovaluename = ''.join([inputfile, '_infovalue.pdf'])
    plot_data(from_file_to_data(inputfile), infousename, infovaluename)

def from_two_stats_file_to_plot(inputfile, inputfile2):
    """
    Take a stats file, of the sort created by stats.compress_stats_directory(),
    and return a couple of plots
    """
    infousename = ''.join([inputfile, '_infouse.pdf'])
    infovaluename = ''.join([inputfile, '_infovalue.pdf'])
    plot_data2(from_file_to_data(inputfile), from_file_to_data(inputfile2), infousename, infovaluename)

def from_evol_to_mixed_strats(evol, strats):
    """
    Take an array with an evolution of populations, and return an array with an
    evolution of mixed strategies.
    """
    return np.array([point_to_mixed_strat(point, strats) for point in evol])

def point_to_mixed_strat(point, strats):
    """
    Take a vector of population weights, and return a mixed strategy
    """
    info = c.s.Information(strats, point)
    return info.population_to_mixed_strat()

def sender_mixed(mixed):
    sender = mixed[:, 0, :, :]
    return sender.swapaxes(1, 2).T

def receiver_mixed(mixed):
    receiver = mixed[:, 1, :, :]
    return receiver.swapaxes(1, 2).T

def plot_mixed(mixed):
    """
    Take an evolution of mixed strats as calculated with
    from_evol_to_mixed_strats() and plot the six subplots for sender and
    receiver
    """
    sendermix = sender_mixed(mixed)
    receivermix = receiver_mixed(mixed)
    fig = plt.figure()
    s1_plot = fig.add_subplot(331)
    state1 = sendermix[0].T
    s1_plot.plot(state1)
    s1_plot.legend(("Message 1", "Message 2", "Message 3"))
    s1_plot.set_title("State 1")
    s2_plot = fig.add_subplot(332)
    state2 = sendermix[1].T
    s2_plot.plot(state2)
    s2_plot.legend(("Message 1", "Message 2", "Message 3"))
    s2_plot.set_title("State 2")
    s3_plot = fig.add_subplot(333)
    state3 = sendermix[2].T
    s3_plot.plot(state3)
    s3_plot.legend(("Message 1", "Message 2", "Message 3"))
    s3_plot.set_title("State 3")
    r1_plot = fig.add_subplot(334)
    message1 = receivermix[0].T
    r1_plot.plot(message1)
    r1_plot.legend(("Act 1", "Act 2", "Act 3"))
    r1_plot.set_title("Message 1")
    r2_plot = fig.add_subplot(335)
    message2 = receivermix[1].T
    r2_plot.plot(message2)
    r2_plot.legend(("Act 1", "Act 2", "Act 3"))
    r2_plot.set_title("Message 2")
    r3_plot = fig.add_subplot(336)
    message3 = receivermix[2].T
    r3_plot.plot(message3)
    r3_plot.legend(("Act 1", "Act 2", "Act 3"))
    r3_plot.set_title("Message 3")
    for ax in [s1_plot, s2_plot, s3_plot, r1_plot, r2_plot, r3_plot]:
        ax.set_frame_on(False)
        ax.set_xlim(0,1000)
        ax.set_ylim(0,1)
    plt.show()


def coord_conversion(triple): # Takes a vector to a point in the simplex
    a = triple[0]
    b = triple[1]
    c = triple[2]
    return (1/2*(2*b+c)/(a+b+c), np.sqrt(3)/2*c/(a+b+c))

def mixed_to_simplex(mixed):
    """
    Take one of the arrays created by sender_mixed() and turn every triple into
    a barycentric coordinates
    """
    triangle1 = np.array([coord_conversion(point) for point in mixed[0].T])
    triangle2 = np.array([coord_conversion(point) for point in mixed[1].T])
    triangle3 = np.array([coord_conversion(point) for point in mixed[2].T])
    return triangle1, triangle2, triangle3


def plot_simplex(simplexdata):
    """
    Take one of the three arrays generated by mixed_to_simplex() and plot its
    simplex. Doesn't quite work yet.
    """
    x = simplexdata[:,0]
    y = simplexdata[:,1]
    xy = simplexdata.T
    z = gaussian_kde(xy)(xy)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y, c=z, s=10, edgecolor='')
    ax.add_artist(plt.Polygon([coord_conversion(n) for n in
                            [[1,0,0],[0,1,0],[0,0,1]]], closed=True,
                              fill=False))
    ax.set_frame_on(False)
    ax.set_axis_off()
    ax.set_xlim(0,1)
    ax.set_ylim(0,np.sqrt(3)/2)
    plt.show()


