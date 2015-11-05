"""
Solve large batches of games. There are a bunch of idyosincratic functions
here. This module is mostly for illustration of use cases.
"""
import json
import os.path
import pickle
import multiprocessing
import itertools as it
import numpy as np

import skyrms.game

def one_basin_mixed(game, trials, times):
    """
    Calculate evolutions for <trials> starting points of <game> (which is an
    instance of game.Evolve), in <times> (an instance of game.Times)
    """
    pool = multiprocessing.Pool(None)
    remain = trials
    # nash = s.Nash(game)
    newsols = pool.imap_unordered(one_basin_aux_mixed, zip(range(remain),
                                  it.repeat(game), it.repeat(times)))
    data = np.array([sol for sol in newsols])
    pool.close()
    pool.join()
    return data


def one_basin_aux_mixed(triple):
    """
    Calculate the one_basin loop. First odeint, then ode if error
    """
    print("trial {} -- odeint".format(triple[0]))
    np.random.seed()
    game = triple[1]
    times = triple[2]
    data = game.replicator_odeint(game.random_sender(), game.random_receiver(),
                                  times)
    if test_failure(data[1]):
        print("trial {} -- ode".format(triple[0]))
        sols = game.replicator_ode(game.random_sender(),
                                   game.random_receiver(), times)
    else:
        sols = data[0]
    tofile = [sols[0]] + [sols[-1]]
    return tofile


def one_basin_aux(triple):
    """
    Calculate the one_basin loop using replicator_odeint
    """
    np.random.seed()
    print("trial {}".format(triple[0]))
    game = triple[1]
    times = triple[2]
    sols = game.replicator_odeint(game.random_sender(), game.random_receiver(),
                                  times)
    return sols


def one_basin_ode_aux(triple):
    """
    Calculate the one_basin loop using one_run_ode
    """
    np.random.seed()
    print("trial {}".format(triple[0]))
    game = triple[1]
    times = triple[2]
    sols = game.replicator_ode(game.random_sender(), game.random_receiver(),
                               times)
    return sols


def one_batch(fileinput, directory, alreadydone=""):
    """
    Take all games in <fileinput> and calculate one_basin on each. Save in
    <directory>
    """
    strat = s.Strategies(3, 3, 3)
    with open(fileinput, 'r') as inputgames:
        gamesdict = json.load(inputgames)
    remaining_games = gamesdict
    if alreadydone != "":
        with open(alreadydone, 'r') as donegames:
            games_done = json.load(donegames)
    else:
        games_done = []
    for key in remaining_games:
        if key not in games_done and eval(key) not in games_done:
            game = s.Game(eval(key), 0, strat)
            outputname = ''.join(["data_", key])
            print(eval(key))
            data = one_basin_mixed(game, 1000)
            with open(os.path.join(directory, outputname), 'wb') as datafile:
                pickle.dump(data, datafile)
            # with open(os.path.join(directory, errorname), 'wb') as errorfile:
            #    pickle.dump(errors, errorfile)
            games_done.append(key)
            with open(
                    os.path.join(directory, "alreadydone"), 'w') as donegames:
                json.dump(games_done, donegames)


def prob_vector(vector):
    """
    Test if <vector> is a probability vector
    """
    return abs(np.sum(vector) - 1) < 1e-5 and np.all([0 <= elem for elem in
                                                      vector])


def test_success(element):
    return "successful" in element['message']


def test_failure(element):
    return "successful" not in element['message']


vtest_success = np.vectorize(test_success)
vtest_failure = np.vectorize(test_failure)


def test_endstate(array):
    """
    Test if <array> is composed by two concatenated probability vectors
    """
    vectors = np.split(np.around(array, decimals=10), 2)
    return prob_vector(vectors[0]) and prob_vector(vectors[1])
