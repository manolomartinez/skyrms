#!/usr/bin/python3

import io, itertools, json, math, random, subprocess, sys, time

import scipy.stats


class Game:
    def __init__(self, payoffs):
        self.dimension = int(math.sqrt(len(payoffs)/2))
        # The dimension of the (square) game
        self.chances = [1/self.dimension for i in range(self.dimension)]
        self.payoffs = payoffs
        self.sender, self.receiver = fromlisttomatrix(
            self.payoffs, self.dimension)
        self.cipeter = self.aggregate_ci_peter()
        self.kendalldistance = round(self. aggregate_kendall_distance(), 2)
        self.kendalldistances = self.aggregate_kendall_distance_distances()
        self.kendallsender, self.kendallreceiver = self.intrakendall()
        self.petersender, self.peterreceiver = self.intrakendallpeter()

    def same_best(self):
        bestactsforsender = [
            setofindexes(acts, max(acts)) for acts in self.sender]
        bestactsforreceiver = [
            setofindexes(acts, max(acts)) for acts in self.receiver]
        samebest = [
            sender & receiver for sender, receiver in
            zip(bestactsforsender, bestactsforreceiver)]
        return samebest

    def intrakendallpeter(self):
        def points(state1, state2, element1, element2):
            pairwise = math.floor(
                abs(preferable(state1, element1, element2) -
                preferable(state2, element1, element2)))
            return pairwise

        def kendall(state1, state2):
            state1plusmean = state1 + [sum(state1)/len(state1)]
            state2plusmean = state2 + [sum(state2)/len(state2)]
            return sum(
                [points(
                    state1plusmean, state2plusmean, pair[0], pair[1])
                    for pair in itertools.combinations(
                        range(self.dimension + 1), 2)])
        skendalls = [kendall(self.sender[pair[0]], self.sender[pair[1]]) for
            pair in itertools.combinations(range(self.dimension), 2)]
        rkendalls = [kendall(self.receiver[pair[0]], self.receiver[pair[1]]) for
            pair in itertools.combinations(range(self.dimension), 2)]
        return sum(skendalls)/len(skendalls), sum(rkendalls)/len(rkendalls)

    def intrakendall(self):
        def points(state1, state2, element1, element2):
            pairwise = math.floor(abs(preferable(state1, element1, element2) -
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
            pairwise = math.floor(abs(preferable(sender, element1, element2) -
            preferable(receiver, element1, element2)))
            return pairwise 
        kendall =  sum([points(self.sender[state], self.receiver[state],
            pair[0], pair[1]) for pair in
            itertools.combinations(range(self.dimension), 2)])
        return kendall

    def aggregate_ci_peter(self):
        return sum([self.chances[state] * self.common_interest_peter(state) for
            state in range(self.dimension)])
        #return [self.common_interest_peter(state) for state in
                #range(self.dimension)]

    def common_interest_peter(self, state):
        senderplusmean = self.sender[state] + [sum(self.sender[state])/len(self.sender[state])]
        receiverplusmean = self.receiver[state] + [sum(self.receiver[state])/len(self.receiver[state])]
        #print(senderplusmean, receiverplusmean)
        def points(sender, receiver, element1, element2):
            pairwise = math.floor(abs(preferable(sender, element1, element2) -
            preferable(receiver, element1, element2)))
            return pairwise 
        kendall =  sum([points(senderplusmean, receiverplusmean,
            pair[0], pair[1]) for pair in
            itertools.combinations(range(self.dimension + 1), 2)])
        return kendall


    def aggregate_kendall_distance_distances(self):
        return sum([self.chances[state] * self.kendall_tau_distance_distances(state) for state in
            range(self.dimension)])

    def kendall_tau_distance_distances(self, state):
        normsender = normalize_vector(self.sender[state])
        normreceiver = normalize_vector(self.receiver[state])
        avg = [(normsender[state] + normreceiver[state])/2 for state in
                range(self.dimension)]
        distance = sum([abs(avg[state] - 0.5) for state in
            range(self.dimension)])
        #print(kendall)
        if distance > 1:
            return 1
        else:
            return distance


    def info_in_equilibria(self):
        gambitgame = bytes(self.write_efg(), "utf-8")
        calc_eqs = subprocess.Popen(['gambit-lcp', '-d', '3'], stdin = subprocess.PIPE,
        #calc_eqs = subprocess.Popen(['gambit-lcp'], stdin = subprocess.PIPE,
                stdout = subprocess.PIPE)
        result = calc_eqs.communicate(input = gambitgame)[0]
        equilibria = str(result, "utf-8").split("\n")[:-1]
        sinfos, rinfos, jinfos = self.calculate_info_content(equilibria)
        return equilibria, max(sinfos), max(rinfos), max(jinfos)

    def payoffs_for_efg(self):
        payoffs = [[','.join([str(sact), str(ract)]) for sact, ract in
            zip(sstate, rstate)] for sstate, rstate in zip(self.sender,
                self.receiver)]
        return payoffs

    def actually_write_efg(self, filename):
        with open(filename, 'w') as output:
            output.write(self.write_efg())
        
    def write_efg(self):
        dimension = self.dimension
        chance = [str(i) for i in self.chances]
        players = self.payoffs_for_efg()
        filelist = []
        filelist.append(r'EFG 2 R "Untitled Extensive Game" { "Player 1" "Player 2" }')
        filelist.append("\n")
        filelist.append(r'""')
        filelist.append("\n")
        filelist.append('')
        filelist.append("\n")

        # The Chance Player line
        line = "c \"\" 1 \"\" {"
        for element in range(len(chance)):
            line = line + " \"" + str(element + 1) + "\" " + str(chance[element])
        line = line + " } 0\n"
        filelist.append(line)

        # A Couple of Useful Strings
        statesstr = "{ "
        for states in range(dimension):
            statesstr = statesstr + "\"" + str(states + 1) + "\" "
        statesstr = statesstr + "}"
        actsstr = "{ "
        for acts in range(len(players[0])):
            actsstr = actsstr + "\"" + str(acts + 1) + "\" "
        actsstr = actsstr + "}"
        messagesstr = "{ "
        for i in range(dimension):
            messagesstr = messagesstr + "\"" + str(i + 1) + "\" "
        messagesstr = messagesstr + "}"

        # The Players lines
        index = 1
        for states in range(dimension):
            line = "p \"\" 1 " + str(states + 1) + " \"\" " + messagesstr + " 0\n"
            filelist.append(line)
            for i in range(dimension):
                line = "p \"\" 2 " + str(i + 1) + " \"\" " + messagesstr + " 0\n"
                filelist.append(line)
                for acts in range(len(players[states])):
                    line = "t \"\" " + str(index) + " \"\" { " + players[states][acts] + " }\n"
                    filelist.append(line)
                    index = index + 1
        filelist.append("</efgfile>\n</game>\n</gambit:document>")
        stringinput = ''.join(filelist)
        return stringinput

    def calculate_Nash_eqs(self, inputfile, outputfile): # calls Gambit and
        #stores the resulting equilibria
        proc = subprocess.Popen(["gambit-lcp", "-d", "3"], stdin = inputfile, stdout =
                outputfile)
        return proc

    def calculate_info_content(self, equilibria): # Given Gambit results, calculate in which equilibria do signals carry information
        chance = self.chances
        sinfos = []
        rinfos = []
        jinfos = []
        #print(equilibria)
        for line in equilibria:
            #print("Equilibrium", line, end =":\n")
            # The following takes a line such as "NE, 0, 0, 1, 0, 0, 1..." to a list [0, 0, 1, 0, 0, 1...]
            equilibrium = list(map(eval, line.split(sep =",")[1:]))
            mutualinfoSM, mutualinfoAM, mutualinfoSA = self.conditional_probabilities(equilibrium)
            #print(mutualinfoSA)
            sinfos.append(mutualinfoSM)
            rinfos.append(mutualinfoAM)
            jinfos.append(mutualinfoSA)
        return sinfos, rinfos, jinfos

    def conditional_probabilities(self, equilibrium): # Calculates the
        #conditional probabilities of states on signals, acts on signals, and
        #states of acts
        # Note the resulting matrices have the form: [[P(S1|M1), P(S2|M1), P(S3|M1)], [P(S1|M2), P(S2|M2), P(S3|M2)]...]
        chance = self.chances
        half = int(len(self.payoffs)/2)
        equilibriumsender = equilibrium[:half]
        equilibriumreceiver = equilibrium[half:]
        #print()
        #print("*******************************")

        ### First, the information that messages carry about states ###

        conditional_probability_matrixsender = []
        unconditionalsmessages = []
        kullbackleibler = []
        for message in range(self.dimension):
            unconditional = sum([chance[i] * equilibriumsender[self.dimension * i + message] for i in range(self.dimension)]) # The unconditional probability of message
            unconditionalsmessages.append(unconditional)
            statesconditionalonmsg = []
            for state in range(self.dimension):
                conditional = chance[state] * safe_div(
                        equilibriumsender[self.dimension * state + message] , unconditional)
                statesconditionalonmsg.append(conditional)

            kld = sum([safe_kld_coefficient(conditional, unconditional) for
                conditional, unconditional in zip(statesconditionalonmsg, chance)])
            kullbackleibler.append(kld)
            #print("KL", kullbackleibler)
            conditional_probability_matrixsender.append(statesconditionalonmsg)
        averagekldsender = sum([prob * kbd for prob, kbd in zip(kullbackleibler,
            unconditionalsmessages)])

        jointprobSM = [[conditional_probability_matrixsender[message][state] *
                unconditionalsmessages[message] for state in
                range(self.dimension)] for message in range(self.dimension)]

        #print("eqbsender", equilibriumsender)
        #print("eqbreceiver", equilibriumreceiver)

        #print("jointprobSM",jointprobSM)

        mutualinfoSM = sum([jointprobSM[message][state] *
                safe_log(jointprobSM[message][state], self.chances[state] *
                    unconditionalsmessages[message]) for state in
                range(self.dimension) for message in range(self.dimension)])

        #print("MutualInfo SM", mutualinfoSM)

        #print('Average KL distance: {}'.format(averagekldsender))
        #print("Uncondmessages", unconditionalsmessages)

        ### Then, the information that messages carry about acts ###

        #print("eq sender {}".format(equilibriumsender))
        #print("eq receiver {}".format(equilibriumreceiver))
        conditional_probability_matrixreceiver= []
        unconditionalsacts = []
        kullbackleibler = []
        # We first calculate the unconditional probability of acts
        for act in range(self.dimension):
            unconditional = sum([unconditionalsmessages[i] *
                equilibriumreceiver[self.dimension * i + act] for i in
                range(self.dimension)]) 
            unconditionalsacts.append(unconditional)
        # Then their probability conditional on a message
        for message in range(self.dimension):
            conditionals4act = []
            if unconditionalsmessages[message] != 0:
                for act in range(self.dimension):
                    conditional = unconditionalsmessages[message] * equilibriumreceiver[self.dimension * message + act] / unconditionalsmessages[message]
                    conditionals4act.append(conditional)
                    #print("act: {}, message: {}, conditional: {}".format(act,
                        #message, conditional))
            else:
                conditionals4act=[0 for i in range(self.dimension)]
            #print("Uncondacts", unconditionalsacts)
            ##print("Cond4acts", conditional)

            kld = sum([safe_kld_coefficient(conditional, unconditional) for
                conditional, unconditional in zip(conditionals4act,
                    unconditionalsacts)])
            kullbackleibler.append(kld)
            #print("KLD: {}".format(kullbackleibler))
            conditional_probability_matrixreceiver.append(conditionals4act)
        averagekldreceiver = sum([prob * kld for prob, kld in zip(
            unconditionalsmessages, kullbackleibler)])

        jointprobAM = [[conditional_probability_matrixreceiver[message][act] *
                unconditionalsmessages[message] for act in
                range(self.dimension)] for message in range(self.dimension)]

        #print("eqbsender", equilibriumsender)
        #print("eqbreceiver", equilibriumreceiver)

        #print("jointprobAM",jointprobAM)

        mutualinfoAM = sum([jointprobAM[message][act] *
                safe_log(jointprobAM[message][act], unconditionalsacts[act] *
                    unconditionalsmessages[message]) for act in
                range(self.dimension) for message in range(self.dimension)])

        #print("MutualInfo AM", mutualinfoAM)

        ### Finally, the info that acts carry about states

        stateconditionalonact = [[safe_div(sum([equilibriumsender[self.dimension * state + message] *
                equilibriumreceiver[self.dimension * message + act] *
                self.chances[state] for message in
                    range(self.dimension)]) , unconditionalsacts[act])
                        for state in range(self.dimension)] for act in
                            range(self.dimension)]

        #print("conditional prob:", stateconditionalonact)
                            
        avgkldjoint = sum([sum([safe_kld_coefficient(stateconditionalonact[act][state], 
            self.chances[state]) for state in
                range(self.dimension)]) * unconditionalsacts[act] for act in
                    range(self.dimension)])

        jointprobSA = [[stateconditionalonact[act][state] *
                unconditionalsacts[act] for state in
                range(self.dimension)] for act in range(self.dimension)]

        #print("eqbsender", equilibriumsender)
        #print("eqbreceiver", equilibriumreceiver)

        #print("jointprobSA",jointprobSA)
        #print("unconditionalsacts",unconditionalsacts)
        #print("unconditionalsacts", unconditionalsacts)
        #print("chances", self.chances)

        mutualinfoSA = sum([jointprobSA[act][state] *
            safe_log(jointprobSA[act][state], unconditionalsacts[act] *
                self.chances[state]) for act in range(self.dimension) for state in range(self.dimension)])

        #print("MutualInfo SA", mutualinfoSA)


        return(mutualinfoSM, mutualinfoAM, mutualinfoSA)

def safe_kld_coefficient(conditional, unconditional):
                if conditional == 0 or unconditional == 0:
                    return 0
                else:
                    return  conditional * math.log2(conditional/unconditional)

def safe_log(a, b):
    try:
        return math.log2(safe_div(a,b))
    except ValueError:
        return 0

def safe_div(a, b):
    try:
        return a/b
    except ZeroDivisionError:
        return 0

def entropy(unconditional_probs):
    return sum([element * math.log2(1/element) for element in
        unconditional_probs])

#def conditional_entropy(unconditional_probs, conditional_probs):
    #return -1 * sum([ unconditional_probs[unconditional] *
        #sum(conditional_probs[conditional][unconditional] *
            #math.log2(1/conditional_probs[conditional][unconditional])))

def order_indexes(preferences):
    return [i[0] for i in sorted(enumerate(preferences), key=lambda x:x[1])]

def avg(alist):
    return round(sum(alist)/len(alist),2)

def avg_abs_dev(alist):
    return sum([abs(element - avg(alist)) for element in alist])/len(alist)

def payoffs(dimension): # The payoff matrix, as a list
    return [random.randrange(100) for x in range(dimension*dimension*2)]

def fromlisttomatrix(payoff, dimension): # Takes a list of intertwined sender and receiver
    # payoffs (what payoffs() outputs) and outputs two lists of lists.
    sender = [payoff[i] for i in range(0,len(payoff),2)]
    sendermatrix = [sender[i:i + dimension] for i in range(0, len(sender),
        dimension)]
    receiver = [payoff[i] for i in range(1,len(payoff),2)]
    receivermatrix = [receiver[i:i+dimension] for i in range(0, len(receiver),
        dimension)]
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

def normalize_vector(vector):
    bottom = min(vector)
    top = max(vector)
    if top != bottom:
        normalized = [(element - bottom)/(top - bottom) for element in vector]
    else:
        normalized  = [0.5 for i in range(len(vector))]
    return normalized

def normalize_matrix(matrix):
    flatmatrix = [i for i in itertools.chain.from_iterable(matrix)] # what's
    # the right way to do this?
    bottom = min(flatmatrix)
    top = max(flatmatrix)
    return [[(element - bottom)/(top - bottom) for element in row] for row in matrix]

def order_list(alist):
    olist = sorted(alist, reverse=True)
    return [olist.index(element) for element in alist]

def generate_30game():
    sender = []
    for j in range(3):
        senderstate = [random.randrange(100) for i in range(3)]
        while len(set(senderstate)) != len(senderstate):
            senderstate = [random.randrange(100) for i in range(3)]
        sender.append(senderstate)
    osender = [order_list(state) for state in sender]
    oreceiver = osender
    receiverlong = []
    senderlong = []
    for i in range(3):
        rmax = max(oreceiver[i])
        rmin = min(oreceiver[i])
        oreceiver[i][oreceiver[i].index(rmax)] = 3
        oreceiver[i][oreceiver[i].index(rmin)]= 2
        oreceiver[i][oreceiver[i].index(3)] = 0
        state = [random.randrange(100) for j in range(3)]
        while order_list(state) != oreceiver[i]:
            state = [random.randrange(100) for i in range(3)]
        receiverlong = receiverlong + state
        senderlong = senderlong + sender[i]
    return [item for pair in zip(senderlong, receiverlong) for item in pair]


def how_many_kendalls(dimension):
    kendallsender = []
    for i in range(1000000):
        payoff = payoffs(dimension)
        game = Game(payoff)
        pair = [game.cipeter, game.petersender]
        if pair not in kendallsender:
            kendallsender.append(pair)
    #with open("sender", "w") as examples:
    #    if sorted(kendallssender) == sorted(kendallsreceiver):
    #        print("yes")
    #        examples.write(str(sorted(kendallssender)))
    #        return sorted(kendallssender), []
    #    else:
    #        print("no")
    return kendallsender
