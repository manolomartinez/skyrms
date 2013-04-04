#!/usr/bin/python3

import io, itertools, json, math, random, subprocess, sys, time

import scipy.stats

class Game:
    def __init__(self, payoffs):
        self.dimension = 3 # The dimension of the (square) game -- this is
        # hardcoded here and there; I need to change that.
        self.chances = [1/3, 1/3, 1/3]
        self.payoffs = payoffs
        self.sender, self.receiver = fromlisttomatrix(self.payoffs)
        self.payofftype = self.payoff_type()
        self.kendallmod = self.aggregate_kendall_mod()
        self.kendalldistance = round(self. aggregate_kendall_distance(),2)
        self.minkendallmod = self.min_kendall_mod()
        self.kendallmoddistance = self.aggregate_kendall_mod_distance()
        self.minkendallmoddistance = self.min_kendall_mod_distance()
        self.actdev = self.act_deviation3()
        self.kendallsender, self.kendallreceiver = self.intrakendall()
#        print("Act deviation: {}".format(self.actdev))
#        print("Modified Kendall tau distance: {}".format(self.kendallmoddistance))
#        print("Modified Kendall tau distance: {}".format(self.kendallmod))
    def intrakendall(self):
        def points(state1, state2, element1, element2):
            pairwise = abs(math.floor(preferable(state1, element1, element2) -
            preferable(state2, element1, element2)))
            return pairwise 
        def kendall(state1, state2):
            return sum([points(state1, state2, pair[0], pair[1]) for pair in 
                itertools.combinations(range(self.dimension), 2)])
        skendalls = [kendall(self.sender[pair[0]], self.sender[pair[1]]) for
            pair in itertools.combinations(range(self.dimension), 2)]
        rkendalls = [kendall(self.receiver[pair[0]], self.receiver[pair[1]]) for
            pair in itertools.combinations(range(self.dimension), 2)]
        return sum(skendalls)/len(skendalls), sum(rkendalls)/len(rkendalls)

        

    def aggregate_kendall_distance(self):
        return sum([self.chances[state] * self.kendall_tau_distance(state) for state in
            range(self.dimension)])

    def kendall_tau_distance(self, state):
        def points(sender, receiver, element1, element2):
            pairwise = abs(math.floor(preferable(sender, element1, element2) -
            preferable(receiver, element1, element2)))
            if sender[element1] == max(sender) or sender[element2] == max(sender):
                weight = 1
            elif sender[element1] == min(sender) or sender[element2] == min(sender):
                weight = 1
            return pairwise * weight
        kendall =  sum([points(self.sender[state], self.receiver[state],
            pair[0], pair[1]) for pair in
            itertools.combinations(range(self.dimension), 2)])
        return kendall

    def kendall_tau_distance_weighted(self, state):
        sender = order_indexes(self.sender[state])
        receiver = order_indexes(self.receiver[state])
        sender = [boost(sender, element) for element in sender]
        receiver = [boost(receiver, element) for element in receiver]
        kendall =  sum([abs((sender[pair[0]]-sender[pair[1]]) - 
            (receiver[pair[0]]-sender[pair[1]])) 
            for pair in itertools.combinations(range(self.dimension), 2)])
        #normalization_coeff = self.dimension * (self.dimension - 1) / 2
        return kendall


    def kendall_tau_distance_distances(self, state):
        normsender = normalize_matrix(self.sender)
        normreceiver = normalize_matrix(self.receiver)
        kendall = sum([abs((normsender[state][pair[0]]-normsender[state][pair[1]]) -
            (normreceiver[state][pair[0]]-normreceiver[state][pair[1]])) for pair in
            itertools.combinations_with_replacement(range(self.dimension), 2)])
        #print(kendall)
        return kendall

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
        
    def min_kendall_mod(self):
        return min([self.kendall_mod(state) for state in
            range(self.dimension)])

    def aggregate_kendall_mod_distance(self):
        return sum([self.chances[state] * self.kendall_tau_distance_distances(state) for state in
            range(self.dimension)])

    def min_kendall_mod_distance(self):
        return min([self.kendall_tau_distance_distances(state) for state in
            range(self.dimension)])

    def info_in_equilibria(self):
        gambitgame = bytes(self.write_efg(), "utf-8")
        calc_eqs = subprocess.Popen(['gambit-lcp'], stdin = subprocess.PIPE,
                stdout = subprocess.PIPE)
        result = calc_eqs.communicate(input = gambitgame)[0]
        equilibria = str(result, "utf-8").split("\n")[:-1]
        infos = self.calculate_info_content(equilibria)
        return equilibria, max(infos)
        
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
        #print(equilibria)
        for line in equilibria:
            #print("Equilibrium", line, end =":\n")
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
        #print('Average KL distance: {}'.format(averagekbd))
        return(averagekbd)

    def payoff_type(self): # What type is the payoff matrix
        sendermatrix, receivermatrix = self.sender, self.receiver
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

    def act_deviation(self):
        normsender = normalize_matrix(self.sender)
        normreceiver = normalize_matrix(self.receiver)
        ad = []
        mean = []
        for act in range(self.dimension):
            pairs = [[statesender[act],
                statereceiver[act]] for statesender, statereceiver in
                zip(normsender, normreceiver)]
            ad.append([avg([avg_abs_dev(pair) for pair in pairs]),
                avg_abs_dev([avg(pair) for pair in pairs])])
        #return avg([row[0] for row in ad])-avg([row[1] for row in ad])
        return ad

    def act_deviation2(self): # A simpler version
        osender = [order_list(row) for row in self.sender]
        oreceiver =[order_list(row) for row in self.receiver] 
        #print(osender, oreceiver)
        ad = []
        mean = []
        for act in range(self.dimension):
            pairs = [[statesender[act],
                statereceiver[act]] for statesender, statereceiver in
                zip(osender, oreceiver)]
            ad.append([avg([abs(pair[0]-pair[1]) for pair in pairs]),
                sum([avg(pair) for pair in pairs])])
        #return avg([row[0] for row in ad])-avg([row[1] for row in ad])
        return ad

    def act_deviation3(self): # A simpler version
        osender = normalize_matrix(self.sender)
        oreceiver = normalize_matrix(self.receiver)
        #print(osender, oreceiver)
        ad = []
        mean = []
        for act in range(self.dimension):
            pairs = [[statesender[act],
                statereceiver[act]] for statesender, statereceiver in
                zip(osender, oreceiver)]
            ad.append([avg([abs(pair[0]-pair[1]) for pair in pairs]),
                sum([avg(pair) for pair in pairs])])
        #return avg([row[0] for row in ad])-avg([row[1] for row in ad])
        return ad
    
def boost(list, element):
    if element == max(list):
        return element * 2
    else:
        return element


def order_indexes(preferences):
    return [i[0] for i in sorted(enumerate(preferences), key=lambda x:x[1])]

def chance(dimension): # State probabilities
    randomlist = [random.random() for i in range(dimension)]
    total = sum(randomlist)
    return [element/total for element in randomlist]

def avg(alist):
    return round(sum(alist)/len(alist),2)

def avg_abs_dev(alist):
    return sum([abs(element - avg(alist)) for element in alist])/len(alist)

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

def normalize_matrix(matrix):
    flatmatrix = [i for i in itertools.chain.from_iterable(matrix)] # what's
    # the right way to do this?
    bottom = min(flatmatrix)
    top = max(flatmatrix)
    return [[(element - bottom)/(top - bottom) for element in row] for row in matrix]

def type_code(payofftype):
    return payofftype[0]*100 + payofftype[1]*10 + payofftype[2]

def order_list(alist):
    olist = sorted(alist, reverse=True)
    return [olist.index(element) for element in alist]

def main():
    games = {}
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    with open(datapointsname, 'w') as datapoints:
        for i in range(40):
            print("EXPERIMENT", i)
            #print()
            entry = {}
            game = Game(payoffs())
            #while game.payofftype != [0,3,0]: 
            #    game = Game(payoffs())
            game.maxinfo = game.info_in_equilibria() 
            datapoints.write('{} {}\n'.format(game.actdev,
                game.maxinfo))
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["payoffs"] = game.payoffs
            entry["kendallmod"] = game.kendallmod
            entry["maxinfo"] = game.maxinfo
            games[i] = entry
            #print()
    #print(games)
    gamesname = ''.join(["games", timestr])
    with open(gamesname, 'w') as gamefile:
        json.dump(games, gamefile)

def main2():
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    with open(datapointsname, 'w') as datapoints:
        for payofftype in types:
            print(payofftype)
            typecode = payofftype[0]*100 + payofftype[1]*10 + payofftype[2]
            for i in range(40):
                print("EXPERIMENT", i)
                #print()
                entry = {}
                game = Game(payoffs())
                while game.payofftype != payofftype: 
                    game = Game(payoffs())
                datapoints.write('{} {}\n'.format(typecode, game.kendallmod))
                
def main3():
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    games = {}
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    with open(datapointsname, 'w') as datapoints:
        for i in range(40):
            print("EXPERIMENT", i)
            print()
            entry = {}
            game = Game(payoffs())
            #while game.payofftype != [0,0,1]: 
            #    game = Game(payoffs())
            game.maxinfo = game.info_in_equilibria() 
            datapoints.write('{} {}\n'.format(type_code(game.payofftype),
                game.maxinfo))
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["payoffs"] = game.payoffs
            entry["kendallmod"] = game.kendallmod
            entry["maxinfo"] = game.maxinfo
            entry["type"] = game.payofftype
            games[i] = entry
            print()
    print(games)
    gamesname = ''.join(["games", timestr])
    with open(gamesname, 'w') as gamefile:
        json.dump(games, gamefile)

def manygames():
    games = {}
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    for gametype in types:
        print("Type: {}".format(gametype))
        for i in range(100):
            print("EXPERIMENT", i)
            #print()
            entry = {}
            game = Game(payoffs())
            while game.payofftype != gametype: 
                game = Game(payoffs())
            game.equilibria, game.maxinfo = game.info_in_equilibria() 
            entry["equilibria"] = str(game.equilibria)
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["kendallmod"] = game.kendallmod
            entry["maxinfo"] = game.maxinfo
            games[str(game.payoffs)] = entry
        gamesname = ''.join(["type", ''.join([str(number) for number in
            gametype]), timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(games, gamefile)


def manymanygames():
    games = {}
    chances = [1/3, 1/3, 1/3]
    for j in range(100):
        timestr = time.strftime("%d%b%H-%M")
        for i in range(200):
            print("EXPERIMENT", j,i)
            entry = {}
            game = Game(payoffs())
            game.equilibria, game.maxinfo = game.info_in_equilibria() 
            entry["equilibria"] = str(game.equilibria)
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["kendallmod"] = game.kendallmod
            entry["maxinfo"] = game.maxinfo
            games[str(game.payoffs)] = entry
        gamesname = ''.join(["manygames",str(j),str(i),"_",timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(games, gamefile)
