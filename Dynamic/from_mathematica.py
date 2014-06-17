import itertools as it
import json
import imp
import importlib
import os
import re

import commoninterest as ci

# Some useful constants

# Constructing Strategies

states = range(3)
signals = range(3)
acts = range(3)

senderstrategies = [i for i in it.product(signals, repeat=len(states))]
receiverstrategies = [i for i in it.product(acts, repeat=len(signals))]

lrs = len(receiverstrategies)
lss = len(senderstrategies)


# Any game will do. This is bad programming in commoninterest.py. Need to change
# it.

game = ci.Game(
    [31, 7, 5, 71, 17, 66, 0, 95, 99, 1, 62, 23, 57, 26, 15, 62, 28, 48])

def tuple_to_equilibrium(sender, receiver):
    stratsender = senderstrategies[sender]
    stratreceiver = receiverstrategies[receiver]
    eqb = []
    for state in stratsender:
        messages = [0, 0, 0]
        messages[state] = 1
        eqb += messages
    for message in stratreceiver:
        receiveracts = [0, 0, 0]
        receiveracts[message] = 1
        eqb += receiveracts
    return eqb

def MI(sender, receiver):
    eqb = tuple_to_equilibrium(sender, receiver)
    return game.conditional_probabilities(eqb)[2]

MI_table = [[MI(i,j) for i in range(27)] for j in range(27)]

class Basin:
    """
    A class with data common to all populations in a basin
    """
    def __init__(self, game):
        self.game = game

    def parse_entry(self, runresults):
        infochange = []
        for trial in runresults:
            dictionary = runresults[trial]
            postsender = dictionary['SenderFinal']
            postreceiver = dictionary['ReceiverFinal']
            infochange.append(self.mean_MI(postsender, postreceiver))
        return infochange

    def mean_MI(self, senderprobs, receiverprobs):
        coeffs = [senderprobs[i] * receiverprobs[j] * MI_table[i][j]
                  for i, j in it.product(range(27), repeat=2)]
        return sum(coeffs)


def results(mathematicafile):
    """
    Takes the result from a Basin[] calculation, and outputs a list with the
    final info content for each trial.
    """
    currentgame = Basin(ci.Game(filename_to_game(mathematicafile)))
    with open(mathematicafile, 'r') as inputfile:
        runresults = eval(inputfile.read())
        infochange = currentgame.parse_entry(runresults)
    return infochange

def result_stats(infochange):
    """
    Take the result of a game and yield percentage of basin with info use,
    and max info use
    """
    return(sum(1 for i in infochange if i>1e-4)/len(
        infochange), max(infochange))

def result_file_per_C(directory, outputfile):
    """
    Take every game in a directory, run result_stats on it, and write
    a csv file with every value
    """
    games = [os.path.join(directory, file) for file in os.listdir(directory)]
    with open(outputfile, 'w') as outputfobj:
        for onegame in games:
            print('Calculating {}'.format(onegame))
            gameresults = result_stats(results(onegame))
            outputfobj.write('{}\t{}\n'.format(gameresults[0], gameresults[1]))

def filename_to_game(filename):
    """
    Take a filename and output a game list
    """
    gamestring = os.path.basename(os.path.splitext(filename)[0])
    gamelist = [eval(number) for number in gamestring.split("-")]
    return gamelist



def game_data_to_payoff_list(filename):
    """
    Take a file with games found with manygames.py, and output a list of
    payoffs (from its name)
    """
    gamelist = []
    with open(filename, 'r') as allgames:
        dictlist = eval(allgames.readline())
        for dictionary in dictlist:
            for key in dictionary:
                gamelist.append(eval(key))
    return gamelist

def sanitize_keys(filename):
    """
    Take a file, and changes every key to avoid duplicates
    """
    with open(filename, 'r') as fileobject:
        filestring = fileobject.read()
    keys = re.findall(r"(?:{|\n)[0-9]+ :", filestring)
    for i in zip(keys, it.count(1000)):
        if i[0][0] == '{':
            replacement = ''.join(['{', str(i[1]), " :"])
        else:
            replacement = ''.join(['\n', str(i[1]), " :"])
        filestring = filestring.replace(i[0], replacement, 1)
    with open(filename, 'w') as fileobject:
        fileobject.write(filestring)

