"""
Actually solve the differentials/difference equations
"""
import setup as s
from scipy.integrate import ode
import json
import os.path
import pickle
import multiprocessing
import itertools as it

# t = s.np.arange(1001)
# strat = s.Strategies(2,2,2)
# game = s.Game(
#    [63, 67, 10, 19, 8, 4, 4, 31, 40, 35, 92,
#     36, 47, 59, 62, 77, 24, 34], 0.01, strat)


def one_basin_mixed(game, trials):
    """
    Calculate evolutions for <trials> starting points
    """
    pool = multiprocessing.Pool(None)
    remain = trials
    # nash = s.Nash(game)
    newsols = pool.imap_unordered(one_basin_aux_mixed, zip(range(remain),
                                  it.repeat(game)))
    data = s.np.array([sol for sol in newsols])
    pool.close()
    pool.join()
    return data


def one_basin_aux_mixed(pair):
    """
    Calculate the one_basin loop. First odeint, then ode if error
    """
    s.np.random.seed()
    print("trial {} -- odeint".format(pair[0]))
    data = one_run_odeint(pair[1], pair[1].strats.random_sender(),
                          pair[1].strats.random_receiver())
    if test_failure(data[1]):
        print("trial {} -- ode".format(pair[0]))
        sols = one_run_ode(pair[1], pair[1].strats.random_sender(),
                           pair[1].strats.random_receiver())
    else:
        sols = data[0]
    tofile = [sols[0]] + [sols[-1]]
    return tofile


def one_basin_aux(pair):
    """
    Calculate the one_basin loop using one_run_odeint
    """
    s.np.random.seed()
    print("trial {}".format(pair[0]))
    sols = one_run_odeint(pair[1], pair[1].strats.random_sender(),
                          pair[1].strats.random_receiver())
    return sols


def one_basin_ode_aux(pair):
    """
    Calculate the one_basin loop using one_run_ode
    """
    s.np.random.seed()
    print("trial {}".format(pair[0]))
    sols = one_run_ode(pair[1], pair[1].strats.random_sender(),
                       pair[1].strats.random_receiver())
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
    return abs(s.np.sum(vector) - 1) < 1e-5 and s.np.all([0 <= elem for elem
                                                          in vector])


def test_success(element):
    return "successful" in element['message']


def test_failure(element):
    return "successful" not in element['message']


vtest_success = s.np.vectorize(test_success)
vtest_failure = s.np.vectorize(test_failure)


def test_endstate(array):
    """
    Test if <array> is composed by two concatenated probability vectors
    """
    vectors = s.np.split(s.np.around(array, decimals=10), 2)
    return prob_vector(vectors[0]) and prob_vector(vectors[1])
