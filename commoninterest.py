#!/usr/bin/python3

import io, itertools, math, pickle, random, subprocess, sys
from contextlib import closing

import scipy.stats

class Game:
    def __init__(self):
        self.dimension = 3 # The dimension of the (square) game -- this is
        # hardcoded here and there; I need to change that.
        self.chances = chance(self.dimension)
        self.payoffs = payoffs()
        self.sender, self.receiver = fromlisttomatrix(self.payoffs)
        self.kendallmod = self.aggregate_kendall_mod()
        print("Modified Kendall tau distance: {}".format(self.kendallmod))


    def kendall_tau_distance(self, state):
        kendall =  sum([abs(math.floor(preferable(self.sender[state], pair[0], pair[1]) -
            preferable(self.receiver[state], pair[0], pair[1]))) for pair in
            itertools.combinations_with_replacement(range(self.dimension), 2)])
        normalization_coeff = self.dimension * (self.dimension - 1) / 2
        return kendall/normalization_coeff

    def same_best(self):
        bestactsforsender = [setofindexes(acts, max(acts)) for acts in
                self.sender]
        bestactsforreceiver = [setofindexes(acts, max(acts)) for acts in
                self.receiver]
        samebest = [sender & receiver for sender, receiver in
                zip(bestactsforsender, bestactsforreceiver)]
        return samebest

    def kendall_mod(self, state): # The kendall tau distance, trumped by equality of
        # best acts
        if self.same_best()[state]:
            return 0
        else:
            return self.kendall_tau_distance(state)

    def aggregate_kendall_mod(self):
        return sum([self.chances[state] * self.kendall_mod(state) for state in
            range(self.dimension)])

    def info_in_equilibria(self):
        gambitgame = bytes(self.write_efg(), "utf-8")
        calc_eqs = subprocess.Popen(['gambit-lcp'], stdin = subprocess.PIPE,
                stdout = subprocess.PIPE)
        result = calc_eqs.communicate(input = gambitgame)[0]
        equilibria = str(result, "utf-8").split("\n")[:-1]
        infos = self.calculate_info_content(equilibria)
        return max(infos)
        
    def write_efg(self): # This writes the game in the form Gambit
        # expects. 'output' is a file object.
        chance = self.chances
        V = self.payoffs
        filelist = []
        filelist.append(r'EFG 2 R "Untitled Extensive Game" { "Player 1" "Player 2" }')
        filelist.append("\n")
        filelist.append(r'""')
        filelist.append("\n")
        filelist.append('')
        filelist.append("\n")
        filelist.append(r'c "" 1 "" {{ "1" {0} "2" {1} "3" {2} }} 0'.format(chance[0], chance[1], chance[2]))
        filelist.append("\n")
        filelist.append(r'p "" 1 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r'p "" 2 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 1 "" {{ {} , {} }}'.format(V[0],V[1]))
        filelist.append("\n")
        filelist.append(r't "" 2 "" {{ {}, {} }}'.format(V[2], V[3]))
        filelist.append("\n")
        filelist.append(r't "" 3 "" {{ {}, {} }}'.format(V[4], V[5]))
        filelist.append("\n")
        filelist.append(r'p "" 2 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 4 "" {{ {}, {} }}'.format(V[0],V[1]))
        filelist.append("\n")
        filelist.append(r't "" 5 "" {{ {}, {} }}'.format(V[2], V[3]))
        filelist.append("\n")
        filelist.append(r't "" 6 "" {{ {}, {} }}'.format(V[4], V[5]))
        filelist.append("\n")
        filelist.append(r'p "" 2 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 7 "" {{ {}, {} }}'.format(V[0],V[1]))
        filelist.append("\n")
        filelist.append(r't "" 8 "" {{ {}, {} }}'.format(V[2], V[3]))
        filelist.append("\n")
        filelist.append(r't "" 9 "" {{ {}, {} }}'.format(V[4], V[5]))
        filelist.append("\n")
        filelist.append(r'p "" 1 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r'p "" 2 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 10 "" {{ {}, {} }}'.format(V[6], V[7]))
        filelist.append("\n")
        filelist.append(r't "" 11 "" {{ {}, {} }}'.format(V[8], V[9]))
        filelist.append("\n")
        filelist.append(r't "" 12 "" {{ {}, {} }}'.format(V[10], V[11]))
        filelist.append("\n")
        filelist.append(r'p "" 2 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 13 "" {{ {}, {} }}'.format(V[6], V[7]))
        filelist.append("\n")
        filelist.append(r't "" 14 "" {{ {}, {} }}'.format(V[8], V[9]))
        filelist.append("\n")
        filelist.append(r't "" 15 "" {{ {}, {} }}'.format(V[10], V[11]))
        filelist.append("\n")
        filelist.append(r'p "" 2 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 16 "" {{ {}, {} }}'.format(V[6], V[7]))
        filelist.append("\n")
        filelist.append(r't "" 17 "" {{ {}, {} }}'.format(V[8], V[9]))
        filelist.append("\n")
        filelist.append(r't "" 18 "" {{ {}, {} }}'.format(V[10], V[11]))
        filelist.append("\n")
        filelist.append(r'p "" 1 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r'p "" 2 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 19 "" {{ {}, {} }}'.format(V[12], V[13]))
        filelist.append("\n")
        filelist.append(r't "" 20 "" {{ {}, {} }}'.format(V[14], V[15]))
        filelist.append("\n")
        filelist.append(r't "" 21 "" {{ {}, {} }}'.format(V[16], V[17]))
        filelist.append("\n")
        filelist.append(r'p "" 2 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 22 "" {{ {}, {} }}'.format(V[12], V[13]))
        filelist.append("\n")
        filelist.append(r't "" 23 "" {{ {}, {} }}'.format(V[14], V[15]))
        filelist.append("\n")
        filelist.append(r't "" 24 "" {{ {}, {} }}'.format(V[16], V[17]))
        filelist.append("\n")
        filelist.append(r'p "" 2 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 25 "" {{ {}, {} }}'.format(V[12], V[13]))
        filelist.append("\n")
        filelist.append(r't "" 26 "" {{ {}, {} }}'.format(V[14], V[15]))
        filelist.append("\n")
        filelist.append(r't "" 27 "" {{ {}, {} }}'.format(V[16], V[17]))
        filelist.append("\n")
        filelist.append(r'</efgfile>')
        filelist.append("\n")
        stringinput = ''.join(filelist)
        return stringinput

    def calculate_Nash_eqs(self, inputfile, outputfile): # calls Gambit and
        #stores the resulting equilibria
        proc = subprocess.Popen(["gambit-lcp"], stdin = inputfile, stdout =
                outputfile)
        return proc

    def calculate_info_content(self, equilibria): # Given Gambit results, calculate in which equilibria do signals carry information
        chance = self.chances
        infos = []
        print(equilibria)
        for line in equilibria:
            print("Equilibrium", line, end =":\n")
            # The following takes a line such as "NE, 0, 0, 1, 0, 0, 1..." to a list [0, 0, 1, 0, 0, 1...]
            equilibrium = list(map(eval, line.split(sep =",")[1:]))
            averagekbd = self.conditional_probabilities(equilibrium)
            infos.append(averagekbd)
        return infos

    def conditional_probabilities(self, equilibrium): # Calculates the conditional probabilities of states on signals
        # Note the resulting matrix has the form: [[P(S1|M1), P(S2|M1), P(S3|M1)], [P(S1|M2), P(S2|M2), P(S3|M3)]...]
        chance = self.chances
        conditional_probability_matrix = []
        unconditionals = []
        kullbackleibler = []
        for message in range(self.dimension):
            unconditional = sum([chance[i] * equilibrium[self.dimension * i + message] for i in range(self.dimension)]) # The unconditional probability of message
            unconditionals.append(unconditional)
            conditionals4message = []
            if unconditional != 0:
                for state in range(self.dimension):
                    conditional = chance[state] * equilibrium[self.dimension * state + message] / unconditional
                    conditionals4message.append(conditional)
            else:
                conditionals4message=[0,0,0]

            def safe_kld_coefficient(conditional, unconditional):
                if conditional == 0:
                    return 0
                else:
                    return  conditional * math.log2(conditional/unconditional)

            kld = sum([safe_kld_coefficient(conditional, unconditional) for
                conditional, unconditional in zip(conditionals4message, chance)])
            kullbackleibler.append(kld)
            conditional_probability_matrix.append(conditionals4message)
        averagekbd = sum([prob * kbd for prob, kbd in zip(kullbackleibler,
            unconditionals)])
        print('Average KL distance: {}'.format(averagekbd))
        return(averagekbd)

def order_indexes(preferences):
    return [i[0] for i in sorted(enumerate(preferences), key=lambda x:x[1])]

def chance(dimension): # State probabilities
    randomlist = [random.random() for i in range(dimension)]
    total = sum(randomlist)
    return [element/total for element in randomlist]

def payoffs(): # The payoff matrix, as a list
    return [random.randrange(0,100) for x in range(18)]

def fromlisttomatrix(payoff): # Takes a list of intertwined sender and receiver
    # payoffs (what payoffs() outputs) and outputs two lists of lists.
    sender = [payoff[i] for i in range(0,18,2)]
    sendermatrix = [sender[0:3],sender[3:6],sender[6:9]]
    receiver = [payoff[i] for i in range(1,18,2)]
    receivermatrix = [receiver[0:3],receiver[3:6],receiver[6:9]]
    return sendermatrix, receivermatrix

def preferable(ranking, element1, element2): # returns 0 if element1 is
    # preferable; 0.5 if both equally preferable; 1 if element2 is preferable
    index1 = ranking[element1]
    index2 = ranking[element2]
    if index2 > index1:
        return 0
    if index2 == index1:
        return 0.5
    if index2 < index1:
        return 1

def setofindexes(originallist, element):
    return set([i for i in range(len(originallist)) if originallist[i] ==
        element]) # We return sets -- later we are doing intersections


if __name__ == "__main__":
    games = []
    chances = [1/3, 1/3, 1/3]
    with open("datapoints", 'w') as datapoints:
        for i in range(50):
            print("EXPERIMENT", i)
            print()
            games.append(Game())
            games[i].maxinfo = games[i].info_in_equilibria() 
            datapoints.write('{} {}\n'.format(games[i].kendallmod,
                games[i].maxinfo))
            print()
    with open("games", 'wb') as gamefile:
        pickle.dump(games, gamefile)
