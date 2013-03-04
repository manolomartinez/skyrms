#!/usr/bin/python3

import itertools, math, pickle, random, subprocess, sys

import scipy.stats

class Game:
    def __init__(self):
        self.dimension = 3 # The dimension of the (square) game -- this is
        # hardcoded here and there; I need to change that.
        self.chances = chance(self.dimension)
        self.payoffs = payoffs()
        self.sender, self.receiver = fromlisttomatrix(self.payoffs)
        self.kendallmod = aggregate_kendall_mod(self)
        print("Modified Kendall tau distance: {}".format(self.kendallmod))

    def kendall_tau_distance(self, state):
        kendall =  sum([abs(floor(preferable(self.sender[state], pair[0], pair[1]) -
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
        if self.same_best[state]:
            return 0
        else:
            return self.kendall_tau_distance(state)

    def aggregate_kendall_mod(self):
        return sum([chances[state] * kendall_mod(state) for state in
            range(self.dimension)])
        
    def write_efg(self, outputfile): # This writes the game in the form Gambit expects
        chance = self.chances
        V = self.payoffs
        with open(outputfile, 'a') as output:
            print(chance)
            output.write(r'EFG 2 R "Untitled Extensive Game" { "Player 1" "Player 2" }')
            output.write("\n")
            output.write(r'""')
            output.write("\n")
            output.write('')
            output.write("\n")
            output.write(r'c "" 1 "" {{ "1" {0} "2" {1} "3" {2} }} 0'.format(chance[0], chance[1], chance[2]))
            output.write("\n")
            output.write(r'p "" 1 1 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r'p "" 2 1 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 1 "" {{ {} , {} }}'.format(V[0],V[1]))
            output.write("\n")
            output.write(r't "" 2 "" {{ {}, {} }}'.format(V[2], V[3]))
            output.write("\n")
            output.write(r't "" 3 "" {{ {}, {} }}'.format(V[4], V[5]))
            output.write("\n")
            output.write(r'p "" 2 2 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 4 "" {{ {}, {} }}'.format(V[0],V[1]))
            output.write("\n")
            output.write(r't "" 5 "" {{ {}, {} }}'.format(V[2], V[3]))
            output.write("\n")
            output.write(r't "" 6 "" {{ {}, {} }}'.format(V[4], V[5]))
            output.write("\n")
            output.write(r'p "" 2 3 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 7 "" {{ {}, {} }}'.format(V[0],V[1]))
            output.write("\n")
            output.write(r't "" 8 "" {{ {}, {} }}'.format(V[2], V[3]))
            output.write("\n")
            output.write(r't "" 9 "" {{ {}, {} }}'.format(V[4], V[5]))
            output.write("\n")
            output.write(r'p "" 1 2 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r'p "" 2 1 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 10 "" {{ {}, {} }}'.format(V[6], V[7]))
            output.write("\n")
            output.write(r't "" 11 "" {{ {}, {} }}'.format(V[8], V[9]))
            output.write("\n")
            output.write(r't "" 12 "" {{ {}, {} }}'.format(V[10], V[11]))
            output.write("\n")
            output.write(r'p "" 2 2 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 13 "" {{ {}, {} }}'.format(V[6], V[7]))
            output.write("\n")
            output.write(r't "" 14 "" {{ {}, {} }}'.format(V[8], V[9]))
            output.write("\n")
            output.write(r't "" 15 "" {{ {}, {} }}'.format(V[10], V[11]))
            output.write("\n")
            output.write(r'p "" 2 3 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 16 "" {{ {}, {} }}'.format(V[6], V[7]))
            output.write("\n")
            output.write(r't "" 17 "" {{ {}, {} }}'.format(V[8], V[9]))
            output.write("\n")
            output.write(r't "" 18 "" {{ {}, {} }}'.format(V[10], V[11]))
            output.write("\n")
            output.write(r'p "" 1 3 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r'p "" 2 1 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 19 "" {{ {}, {} }}'.format(V[12], V[13]))
            output.write("\n")
            output.write(r't "" 20 "" {{ {}, {} }}'.format(V[14], V[15]))
            output.write("\n")
            output.write(r't "" 21 "" {{ {}, {} }}'.format(V[16], V[17]))
            output.write("\n")
            output.write(r'p "" 2 2 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 22 "" {{ {}, {} }}'.format(V[12], V[13]))
            output.write("\n")
            output.write(r't "" 23 "" {{ {}, {} }}'.format(V[14], V[15]))
            output.write("\n")
            output.write(r't "" 24 "" {{ {}, {} }}'.format(V[16], V[17]))
            output.write("\n")
            output.write(r'p "" 2 3 "" { "1" "2" "3" } 0')
            output.write("\n")
            output.write(r't "" 25 "" {{ {}, {} }}'.format(V[12], V[13]))
            output.write("\n")
            output.write(r't "" 26 "" {{ {}, {} }}'.format(V[14], V[15]))
            output.write("\n")
            output.write(r't "" 27 "" {{ {}, {} }}'.format(V[16], V[17]))
            output.write("\n")
            output.write(r'</efgfile>')
            output.write("\n")

    def calculate_Nash_eqs(inputfile, outputfile): # calls Gambit and stores the resulting equilibria
        with open(inputfile) as gambitinput, open(outputfile, 'w') as gambitoutput:
            subprocess.call("gambit-lcp", stdin = gambitinput, stdout = gambitoutput)

    def calculate_info_content(self, inputfile): # Given Gambit results, calculate in which equilibria do signals carry information
        chance = self.chances
        with open(inputfile) as gambitresults:
            equilibria = gambitresults.readlines()
            for line in equilibria:
                print("Equilibrium", line, end =":\n")
                # The following takes a line such as "NE, 0, 0, 1, 0, 0, 1..." to a list [0, 0, 1, 0, 0, 1...]
                equilibrium = list(map(eval, line.split(sep =",")[1:]))
                cond_prob_matrix = conditional_probabilities(self, equilibrium)
                info = carries_info(self, cond_prob_matrix)
        return info

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
        return(conditional_probability_matrix)

    def carries_info(self, conditional_probability_matrix):
        chances = self.chances
        info = []
        for message in range(self.dimension):
            for state in range(self.dimension):
                if conditional_probability_matrix[message][state] != 0 and conditional_probability_matrix[message][state] != chances[state]:
                    print(message, state, conditional_probability_matrix[message][state])
                    print("Message", message + 1, "carries information about state", state + 1)
                    info.append([message, state])
        return(info)

def payoff_type(payoff): # What type is the payoff matrix
    sendermatrix, receivermatrix = fromlisttomatrix(payoff)
    bestactsforsender = [setofindexes(acts, max(acts)) for acts in
            sendermatrix]
    bestactsforreceiver = [setofindexes(acts, max(acts)) for acts in
            receivermatrix]
    worstactsforsender = [setofindexes(acts, min(acts)) for acts in
            sendermatrix]
    worstactsforreceiver = [setofindexes(acts, min(acts)) for acts in
            receivermatrix]
    iterator = range(len(worstactsforsender))
    bestacts = [len(bestactsforsender[i] & bestactsforreceiver[i]) > 0 for i in
            iterator] # the intersection of the set of best acts for
    # sender and receiver is nonzero
    worstacts = [len(worstactsforsender[i] & worstactsforreceiver[i]) > 0 for i in
            iterator]

    sameordering = [bestacts[i] and worstacts[i] for i in iterator]
    onlybestacts = [bestacts[i] and not worstacts[i] for i in iterator]
    onlyworstacts = [worstacts[i] and not bestacts[i] for i in iterator]
    return [sum(sameordering), sum(onlybestacts), sum(onlyworstacts)]

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










            
def kendall_tau(payoffs): # Calculates the three Kendall tau distances between
    # payoffs -- I'm not using it at the moment
    senderpayoffs = payoffs[::2]
    senderpayoffperstate = [senderpayoffs[i:i+3] for i in [0, 3, 6]]
    senderpreferences = [order_indexes(senderpayoffperstate[i]) for i in [0,1,2]]
    receiverpayoffs = payoffs[1::2]
    receiverpayoffperstate = [receiverpayoffs[i:i+3] for i in [0, 3, 6]]
    receiverpreferences = [order_indexes(receiverpayoffperstate[i]) for i in [0, 1, 2]]
    print (senderpayoffperstate, receiverpayoffperstate)
    print(senderpreferences, receiverpreferences)

    kendalltau = [scipy.stats.kendalltau(senderpayoffperstate[i], receiverpayoffperstate[i]) for i in [0,1,2]]
    spearmanr = [scipy.stats.spearmanr(senderpayoffperstate[i], receiverpayoffperstate[i]) for i in [0,1,2]]

    return kendalltau, spearmanr
            


if __name__ == "__main__":
    outcome = {}
    chances = [1/3, 1/3, 1/3]
    for i in range(10):
        print("EXPERIMENT", i)
        print()
        entry = {}
        #chances = chance()
        payoff = payoffs()
        payofftype = payoff_type(payoff)
        while payofftype != [0,0,0]:
            payoff = payoffs()
            payofftype = payoff_type(payoff)
        entry["type"] = payofftype
        sendermatrix, receivermatrix = fromlisttomatrix(payoff)
        entry["senderpayoff"] = sendermatrix
        entry["receiverpayoff"] = receivermatrix
        print("sender ", sendermatrix)
        print("receiver", receivermatrix)
        print("type", payofftype)
        inputname = "input{}.efg".format(i)
        outputname = "output{}".format(i)
        write_efg(inputname, chances,  payoff)
        calculate_Nash_eqs(inputname, outputname)
        entry["info"]=calculate_info_content(outputname, chances)
        kendall = kendall_tau(payoff)
        entry["kendall"] = kendall
        print("Kendall tau", kendall)
        outcome[i] = entry
    with open("outcome", 'wb') as resultsfile:
        pickle.dump(outcome, resultsfile)


