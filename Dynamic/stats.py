"""
The script that extracts info from the data
"""

import numpy as np
import os
import pickle as p
import json as j
import calculations as c

strats = c.s.Strategies(3, 3, 3)

def pickle_to_json():
    """
    Take two parallel hierarchies of pickled files to an analogous hierarchy of
    json files
    """
    basedir = os.getcwd()
    datadirs = os.listdir('ResultsJacobian/')
    newdir = "Resultsjson"
    os.mkdir(os.path.join(basedir, newdir))
    for ddir in datadirs:
        print(ddir, '...')
        newddir = os.path.join(basedir, newdir, ddir)
        os.mkdir(newddir)
        dirpath1 = os.path.join(basedir, 'ResultsJacobian/', ddir)
        dirpath2 = os.path.join(basedir, 'ResultsJacobianbis/', ddir)
        files1 = [filename for filename in os.listdir('dirpath1') if 'data' in
                  filename]
        for fname in files1:
            print(fname)
            with open(os.path.join(dirpath1, fname), 'rb') as pickle1, open(
                    os.path.join(dirpath2, fname), 'rb') as pickle2:
                gamedict1 = p.load(pickle1)
                gamedict2 = p.load(pickle2)
                gamedict = np.concatenate((gamedict1, gamedict2), axis=0)
            with open(os.path.join(newddir, fname), 'w') as jsonfile:
                jsonfile.write(j.dumps(gamedict))
        print('Done')

def mutual_info_batch(inputdirs, newdir):
    """
    Take any number of  parallel hierarchies of pickled files (including one;
    in the <inputdirs> list) to an analogous hierarchy of files (in
    <newdir>) with mutual info per end point per game
    """
    basedir = os.getcwd()
    if type(inputdirs) == str:
        inputdirs = [inputdirs]
    datadirs = os.listdir(inputdirs[0]) # Any of them will do (they are
    # parallel)
    print(datadirs)
    os.mkdir(os.path.join(basedir, newdir))
    for ddir in datadirs:
        print(ddir, '...')
        newddir = os.path.join(basedir, newdir, ddir)
        os.mkdir(newddir)
        dirpaths = [os.path.join(basedir, inputdir, ddir) for inputdir in
                   inputdirs]
        files = [filename for filename in os.listdir(dirpaths[0]) if 'data' in
                  filename]
        for fname in files:
            gamedict = {}
            print(fname)
            for dirpath in dirpaths:
                with open(os.path.join(dirpath, fname), 'rb') as pickled:
                    newgamedict = p.load(pickled)
                    print(type(newgamedict))
                    if type(newgamedict) != str:
                        try:
                           gamedict = np.concatenate((gamedict, newgamedict), axis=0)
                        except ValueError:
                            gamedict = newgamedict
                    else:
                        gamedict = {}
            print(len(gamedict))
            with open(os.path.join(newddir, fname), 'w') as output:
                for evol in gamedict:
                    if is_a_prob_vector(evol[-1]):
                        output.write("{}\n".format(info_use_per_evol(evol)))
        print('Done')


def info_use_per_point(point):
    """
    Take one state and return mutual info between states and acts in the
    state.
    """
    info = c.s.Information(strats, point)
    return info.mutual_info_states_acts()

def mutual_info_states_messages_per_point(point):
    """
    Take one state and return mutual info between states and messages in the
    state.
    """
    info = c.s.Information(strats, point)
    return c.s.mutual_info_from_joint(info.joint_states_messages())

def mutual_info_acts_messages_per_point(point):
    """
    Take one state and return mutual info between acts and messages in the
    state.
    """
    info = c.s.Information(strats, point)
    return c.s.mutual_info_from_joint(info.joint_messages_acts())

def info_use_per_evol(evol):
    """
    Take one evolution and return mutual info between states and acts in the
    end state.
    """
    endstate = evol[-1]
    info = c.s.Information(strats, endstate)
    return info.mutual_info_states_acts()

def stats_per_C(datadir, newdir):
    """
    Calculate the average, max and min percentage of info-using end points.
    Also the average, max and min info-use
    """
    basedir = os.getcwd()
    datadirs = os.listdir(datadir)
    os.mkdir(os.path.join(basedir, newdir))
    for ddir in datadirs:
        print(ddir, '...')
        newfile = os.path.join(basedir, newdir, ddir)
        dirpath = os.path.join(basedir, datadir, ddir)
        files = [os.path.join(basedir, datadir, ddir, filename) for filename in os.listdir(dirpath)]
        with open(newfile, 'w') as statsperc:
            for fname in files:
                print(fname)
                try:
                    statsperc.write("{}\t{}\t{}\t{}\n".format(*mutual_info_avgs_per_file(fname)))
                except:
                    pass
    print('Done')

def mutual_info_avgs_per_file(gamefile):
    """
    Take a file with evolutions, and calculate the percentage of info-using end
    points, and the avg, max and min info uses.
    """
    with open(gamefile, 'r') as evols:
        results = evols.readlines()
        evolslist = np.array([eval(evol) for evol in results])
        print(len(evolslist))
        percentage_mi = len(evolslist[evolslist > 1e-03])/len(evolslist)
        avg_mi = np.average(evolslist)
    return percentage_mi, avg_mi, max(evolslist), min(evolslist)

def compress_stats_per_C(filename):
    """
    Take a file of stats produced by stats_per_c() and calculate 
    * an average of percentages of info-use per starting point
    * the maximum percentage
    * the min percentage
    * average of maximums of info uses per starting point
    * total max of info uses 
    * total min of info uses
    """
    with open(filename, 'r') as statsperc:
        results = statsperc.readlines()
        numresults = np.array([[eval(number) for number in result.split()] for
                               result in results])
    averages = np.average(numresults, axis = 0)
    maxs = np.amax(numresults, axis = 0)
    mins = np.amin(numresults, axis = 0)
    return averages[0], maxs[0], mins[0], averages[1], maxs[2], mins[3]

def compress_stats_directory(directory, outputfile):
    """
    Take a directory with files of stats produced by stats_per_c() and create a
    file with the results of compress_stats_per_C() per file
    """
    basedir = os.getcwd()
    datadir = os.path.join(basedir, directory)
    output_fullpath = os.path.join(basedir, outputfile)
    datafiles = os.listdir(datadir)
    with open(output_fullpath, 'w') as output:
        for datafile in datafiles:
            output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(datafile,
                                                       *compress_stats_per_C(os.path.join(basedir,
                                                                                          datadir,
                                                                                          datafile))))

def is_a_prob_vector(vector):
    greater_than_zero = np.all(vector > -1e-5)
    add_up_to_one = abs(sum(vector) - 2) < 1e-5
    return greater_than_zero and add_up_to_one
