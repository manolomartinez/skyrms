# coding: utf-8
import os
import pickle as p
import setup as s
import calculations as c
import itertools as it

def mainscript(gamesfile, outputdir):
    """
    Calculate a batch of bad marriages from a list
    """
    with open(gamesfile, "r") as bm:
        games = bm.readlines()
    gameslist = [eval(game) for game in games]
    os.mkdir(outputdir)
    os.chdir(outputdir)
    for game in gameslist:
        print(game)
        gameobj = s.Game(game, 0, s.Strategies(3, 3, 3))
        data, _ = c.one_basin_mixed(gameobj, 1000)
        with open(''.join(["data_", str(game)]), 'wb') as datapickle:
            p.dump(data, datapickle)
            
def aretheyNash(pickledfile):
    """
    Take a pickled file of starting and end points of evolutions and calculate
    whether the end point is a Nash equilibrium
    """
    strats = s.Strategies(3, 3, 3)
    payoffs = eval(pickledfile[5:])
    game = s.Game(payoffs, 0, strats)
    nashclass = s.Nash(game)
    with open(pickledfile, 'rb') as pevols:
        evols = p.load(pevols)
    for evol, counter in zip(evols, it.count()):
        endstate = evol[-1]
        senderpops = endstate[:game.strats.lss]
        receiverpops = endstate[game.strats.lss:]
        if nashclass.is_Nash(senderpops, receiverpops):
            info = s.Information(strats, endstate)
            if info.mutual_info_states_acts() > 1e-2:
                print(counter)

