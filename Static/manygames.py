import commoninterest as ci
import create_games as cg
import json
import os
import pickle
import time


def manygames(dimension):
    possible_pairs = [
        [2.33, 1.3333333333333333], [2.33, 2.0], [2.33, 2.6666666666666665],
        [2.33, 3.3333333333333335], [2.33, 4.0], [2.6666666666666666, 0.0],
        [2.6666666666666666, 0.6666666666666666], [2.6666666666666666, 1.3333333333333333], [2.6666666666666666, 2.0],
        [2.6666666666666666, 2.6666666666666665], [2.6666666666666666, 3.3333333333333335], [2.6666666666666666, 4.0],
        [3.0, 0.0], [3.0, 0.6666666666666666], [3.0, 1.3333333333333333],
        [3.0, 2.0], [3.0, 2.6666666666666665], [3.0, 3.3333333333333335],
        [3.0, 4.0], [3.33, 0.0], [3.33, 0.6666666666666666],
        [3.33, 1.3333333333333333], [3.33, 2.0], [3.33, 2.6666666666666665],
        [3.33, 3.3333333333333335], [3.33, 4.0], [3.6666666666666666, 0.0],
        [3.6666666666666666, 0.6666666666666666], [3.6666666666666666, 1.3333333333333333], [3.6666666666666666, 2.0],
        [3.6666666666666666, 2.6666666666666665], [3.6666666666666666, 3.3333333333333335], [3.6666666666666666, 4.0],
        [4.0, 0.0], [4.0, 0.6666666666666666], [4.0, 1.3333333333333333],
        [4.0, 2.0], [4.0, 2.6666666666666665], [4.0, 3.3333333333333335],
        [4.0, 4.0], [4.33, 0.0], [4.33, 0.6666666666666666],
        [4.33, 1.3333333333333333], [4.33, 2.0], [4.33, 2.6666666666666665],
        [4.33, 3.3333333333333335], [4.33, 4.0], [4.67, 0.0],
        [4.67, 0.6666666666666666], [4.67, 1.3333333333333333], [4.67, 2.0],
        [4.67, 2.6666666666666665], [4.67, 3.3333333333333335], [4.67, 4.0],
        [5.0, 0.0], [5.0, 0.6666666666666666], [5.0, 1.3333333333333333],
        [5.0, 2.0], [5.0, 2.6666666666666665], [5.0, 3.3333333333333335],
        [5.0, 4.0], [5.33, 0.0], [5.33, 0.6666666666666666],
        [5.33, 1.3333333333333333], [5.33, 2.0], [5.33, 2.6666666666666665],
        [5.33, 3.3333333333333335], [5.33, 4.0], [5.67, 0.0],
        [5.67, 0.6666666666666666], [5.67, 1.3333333333333333], [5.67, 2.0],
        [5.67, 2.6666666666666665], [5.67, 3.3333333333333335], [5.67, 4.0],
        [6.0, 0.0], [6.0, 1.3333333333333333], [6.0, 2.0],
        [6.0, 2.6666666666666665], [6.0, 3.3333333333333335], [6.0, 4.0]]
    #possible_intrakendalls = [1.3333333333333333, 0.6666666666666666, 2.0,
            #1.6666666666666667, 0.0, 0.3333333333333333, 1.0]
    #possible_kendalls = sorted([2.0, 1.33, 1.6666666666666666, 0.6666666666666666, 1.0, 2.6666666666666666, 0.33, 2.33, 0.0,
            #3.0])
    #possible_intrakendalls = [3.3333333333333335, 2.0, 2.6666666666666665,
    #    4.0, 2.3333333333333335, 0.6666666666666666, 1.3333333333333333,
    #    3.6666666666666665, 3.0, 0.0, 1.6666666666666667, 0.3333333333333333,
    #    1.0]
    #possible_kendalls = [3.33, 1.0, 4.33, 2.0, 3.0, 2.33, 2.6666666666666666, 0.33, 3.6666666666666666,
    #        1.33, 1.6666666666666666, 4.0, 5.0, 4.67, 5.67, 0.6666666666666666, 5.33, 6.0, 0.0]

    gamessender = {}
    gamesreceiver = {}
    timestr = time.strftime("%d%b%H-%M")
    #for duo in itertools.product(possible_kendalls, possible_intrakendalls):
    for pair in possible_pairs:
        print("Type: {}".format(pair))
        for i in range(2000):
            print("EXPERIMENT", i)
            #print()
            gamesender = ci.Game(cg.create_game(pair, "sender"))
            gamereceiver = ci.Game(cg.create_game(pair, "receiver"))
            print(
                gamesender.payoffs, gamesender.cistar, gamesender.starsender)
            print(
                gamereceiver.payoffs,
                gamereceiver.cistar, gamereceiver.starreceiver)
            gamesender.equilibria, b, c, d = gamesender.info_in_equilibria()
            gamereceiver.equilibria, b, c, d = gamereceiver.info_in_equilibria()
            gamessender[str(gamesender.payoffs)] = gamesender.equilibria
            gamesreceiver[str(gamereceiver.payoffs)] = gamereceiver.equilibria
        gamesname = ''.join(["boostintrasender", str(pair), timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(gamessender, gamefile)
        gamesname = ''.join(["boostintrareceiver", str(pair), timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(gamesreceiver, gamefile)


def manymanygames(dimension):
    possible_kendalls = sorted([2.0, 1.33, 1.6666666666666666, 0.6666666666666666, 1.0, 2.6666666666666666, 0.33, 2.33, 0.0,
            3.0])
    games = {}
    for j in possible_kendalls:
        timestr = time.strftime("%d%b%H-%M")
        for i in range(2000):
            print("EXPERIMENT", j,i)
            entry = {}
            game = Game(payoffs(dimension))
            while game.kendalldistance != j:
                game = Game(payoffs(dimension))
            game.equilibria, b, c, d = game.info_in_equilibria() 
            games[str(game.payoffs)] = game.equilibria
        gamesname = ''.join(["manygames",str(j),str(i),"_",timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(games, gamefile)

def test(dimension):
    maxinfo = 0
    counter = 0
    while maxinfo < 10e-4:
        print(counter)
        counter += 1
        game = ci.Game(ci.cs_payoffs(dimension))
        a, b, c, maxinfo = game.info_in_equilibria() 
    print(game.payoffs, maxinfo)

def create_pickles(possible_pairs_file):
    with open(possible_pairs_file, 'r') as ppf:
        possible_pairs = eval(ppf.readline())
    gamessender = {}
    gamesreceiver = {}
    for pair in possible_pairs:
        gamessender[str(pair)] = []
        gamesreceiver[str(pair)] = []
    with open("pickledsender", "wb") as psender:
        pickle.dump(gamessender, psender)
    with open("pickledreceiver", "wb") as preceiver:
        pickle.dump(gamesreceiver, preceiver)

def findgames(dimension):
    with open("pickledsender", "rb") as psender:
        gamessender = pickle.load(psender)
    with open("pickledreceiver", "rb") as preceiver:
        gamesreceiver = pickle.load(preceiver)
    with open("pickledpairsender", "rb") as psender:
        pairsender = pickle.load(psender)
    with open("pickledpairreceiver", "rb") as preceiver:
        pairreceiver = pickle.load(preceiver)
    with open("pickledpossiblepairs", "rb") as pp:
        possible_pairs = pickle.load(pp)
    try:
        possible_cstars = []
        
        #possible_pairs = []
        #pairsender = []
        #pairreceiver = []
        #gamessender = {}
        #gamesreceiver = {}
        timestr = time.strftime("%d%b%H-%M")
        #for pair in pairsender:
            #gamessender[str(pair)] = []
            #gamesreceiver[str(pair)] = []
        #game = ci.Game(ci.payoffs(dimension))
        #gamepairsender = [game.cistar, game.starsender]
        #gamepairreceiver = [game.cistar, game.starreceiver]
        #if gamepairsender not in possible_pairs or gamepairreceiver not in possible_pairs:
            #equilibria = game.just_equilibria()
            #if gamepairsender not in possible_pairs:
                #ppold = possible_pairs
                #possible_pairs.append(gamepairsender)
                #pairsender.append(gamepairsender)
                #pairreceiver.append(gamepairsender)
                #gamessender[str(gamepairsender)] = []
                #gamesreceiver[str(gamepairsender)] = []
                #gameentry = {}
                #gameentry[str(game.payoffs)] = equilibria
                #gamessender[str(gamepairsender)].append(gameentry)
                #print("sender", gamepairsender,
                      #len(gamessender[str(gamepairsender)]))
            #if gamepairreceiver not in ppold:
                #if gamepairreceiver not in possible_pairs:
                    #possible_pairs.append(gamepairreceiver)
                #pairreceiver.append(gamepairreceiver)
                #pairsender.append(gamepairreceiver)
                #gamesreceiver[str(gamepairreceiver)] = []
                #gamessender[str(gamepairreceiver)] = []
                #gameentry = {}
                #gameentry[str(game.payoffs)] = equilibria
                #gamesreceiver[str(gamepairreceiver)].append(gameentry)
                #print("receiver", gamepairreceiver,
                      #len(gamesreceiver[str(gamepairreceiver)]))

        while pairsender != [] or pairreceiver != []:
            game = ci.Game(ci.payoffs(dimension))
            gamepairsender = [game.cistar, game.starsender]
            gamepairreceiver = [game.cistar, game.starreceiver]
            if gamepairsender not in possible_pairs or gamepairreceiver not in possible_pairs or gamepairsender in pairsender or gamepairreceiver in pairreceiver:
                equilibria = game.just_equilibria()
                if gamepairsender not in possible_pairs:
                    possible_pairs.append(gamepairsender)
                    pairsender.append(gamepairsender)
                    pairreceiver.append(gamepairsender)
                    gamessender[str(gamepairsender)] = []
                    gamesreceiver[str(gamepairsender)] = []
                if gamepairreceiver not in possible_pairs:
                    possible_pairs.append(gamepairreceiver)
                    pairsender.append(gamepairreceiver)
                    gamessender[str(gamepairreceiver)] = []
                    pairreceiver.append(gamepairreceiver)
                    gamesreceiver[str(gamepairreceiver)] = []
                for pair in pairsender:
                    if gamepairsender == pair:
                        gameentry = {}
                        gameentry[str(game.payoffs)] = equilibria
                        gamessender[str(pair)].append(gameentry)
                        print("sender", pair, len(gamessender[str(pair)]))
                        if len(gamessender[str(pair)]) > 1499:
                            pairsender.remove(pair)
                            print("sender", pair, "done.")
                            filename = ''.join(["sender", str(pair), timestr])
                            with open(filename, 'w') as senderpair:
                                json.dump(gamessender[str(pair)], senderpair)
                for pair in pairreceiver:
                    if gamepairreceiver == pair:
                        gameentry = {}
                        gameentry[str(game.payoffs)] = equilibria
                        gamesreceiver[str(pair)].append(gameentry)
                        print("receiver", pair, len(gamesreceiver[str(pair)]))
                        if len(gamesreceiver[str(pair)]) > 1499:
                            pairreceiver.remove(pair)
                            print("receiver", pair, "done.")
                            filename = ''.join(["receiver", str(pair), timestr])
                            with open(filename, 'w') as receiverpair:
                                json.dump(gamesreceiver[str(pair)], receiverpair)
        with open("gamessender", "w") as senderfile:
            json.dump(gamessender, senderfile)
        with open("gamesreceiver", "w") as receiverfile:
            json.dump(gamesreceiver, receiverfile)
        with open("pickledsender", "wb") as psender:
            pickle.dump(gamessender, psender)
        with open("pickledreceiver", "wb") as preceiver:
            pickle.dump(gamesreceiver, preceiver)
        with open("pickledpairsender", "wb") as psender:
            pickle.dump(pairsender, psender)
        with open("pickledpairreceiver", "wb") as preceiver:
            pickle.dump(pairreceiver, preceiver)
        with open("pickledpossiblepairs", "wb") as pp:
            pickle.dump(possible_pairs, pp)

    except KeyboardInterrupt:
        with open("pickledsender", "wb") as psender:
            pickle.dump(gamessender, psender)
        with open("pickledreceiver", "wb") as preceiver:
            pickle.dump(gamesreceiver, preceiver)
        with open("pickledpairsender", "wb") as psender:
            pickle.dump(pairsender, psender)
        with open("pickledpairreceiver", "wb") as preceiver:
            pickle.dump(pairreceiver, preceiver)
        with open("pickledpossiblepairs", "wb") as pp:
            pickle.dump(possible_pairs, pp)



def calculate_eqbs(filelist):
    for payoffsfile in filelist:
        print('Calculating eqb for {}.'.format(payoffsfile))
        games = {}
        resultsname = ''.join(['results', payoffsfile])
        with open(payoffsfile, 'r') as pfile, open(
                os.path.join("Equilibria", resultsname), 'w') as resultsfile:
            payofflist = json.loads(pfile.readline())
            for payoff in payofflist:
                print(payoff, type(payoff))
                game = ci.Game(payoff)
                #game.actually_write_efg("pruebicaatope.efg")
                game.equilibria = game.info_in_equilibria()[0]
                games[str(payoff)] = game.equilibria
            json.dump(games, resultsfile)

def findgames_CKnostar(dimension):
    with open("pickledsender", "rb") as psender:
        gamessender = pickle.load(psender)
    with open("pickledreceiver", "rb") as preceiver:
        gamesreceiver = pickle.load(preceiver)
    with open("pickledpairsender", "rb") as psender:
        pairsender = pickle.load(psender)
    with open("pickledpairreceiver", "rb") as preceiver:
        pairreceiver = pickle.load(preceiver)
    with open("pickledpossiblepairs", "rb") as pp:
        possible_pairs = pickle.load(pp)
    with open("pickledcs", "rb") as pp:
        possible_cs = pickle.load(pp)
    with open("pickledgames", "rb") as pp:
        games = pickle.load(pp)
    try:
        #possible_cs = []
        #possible_pairs = []
        #pairsender = []
        #pairreceiver = []
        #games = {}
        #gamessender = {}
        #gamesreceiver = {}
        timestr = time.strftime("%d%b%H-%M")
        #for pair in pairsender:
            #gamessender[str(pair)] = []
            #gamesreceiver[str(pair)] = []
        #game = ci.Game(ci.payoffs(dimension))
        #gamec = game.kendalldistance
        #gamepairsender = [game.kendalldistance, game.kendallsender]
        #gamepairreceiver = [game.kendalldistance, game.kendallreceiver]
        #if gamepairsender not in possible_pairs or gamepairreceiver not in possible_pairs:
            #equilibria = game.just_equilibria()
            #if gamepairsender not in possible_pairs:
                #ppold = possible_pairs
                #possible_cs.append(gamec)
                #possible_pairs.append(gamepairsender)
                #pairsender.append(gamepairsender)
                #pairreceiver.append(gamepairsender)
                #gamessender[str(gamepairsender)] = []
                #gamesreceiver[str(gamepairsender)] = []
                #games[str(gamec)] = []
                #gameentry = {}
                #gameentry[str(game.payoffs)] = equilibria
                #gamessender[str(gamepairsender)].append(gameentry)
                #games[str(gamec)].append(gameentry)
                #print("sender", gamepairsender,
                      #len(gamessender[str(gamepairsender)]))
            #if gamepairreceiver not in ppold:
                #if gamepairreceiver not in possible_pairs:
                    #possible_pairs.append(gamepairreceiver)
                #pairreceiver.append(gamepairreceiver)
                #pairsender.append(gamepairreceiver)
                #gamesreceiver[str(gamepairreceiver)] = []
                #gamessender[str(gamepairreceiver)] = []
                #gameentry = {}
                #gameentry[str(game.payoffs)] = equilibria
                #gamesreceiver[str(gamepairreceiver)].append(gameentry)
                #print("receiver", gamepairreceiver,
                      #len(gamesreceiver[str(gamepairreceiver)]))

        while pairsender != [] or pairreceiver != []:
            game = ci.Game(ci.payoffs(dimension))
            gamec = game.kendalldistance
            gamepairsender = [game.kendalldistance, game.kendallsender]
            gamepairreceiver = [game.kendalldistance, game.kendallreceiver]
            if gamepairsender not in possible_pairs or gamepairreceiver not in possible_pairs or gamepairsender in pairsender or gamepairreceiver in pairreceiver:
                equilibria = game.just_equilibria()
                #if gamec not in possible_cs:
                    #possible_cs.append(gamec)
                    #games[str(gamec)] = []
                if gamepairsender not in possible_pairs:
                    possible_pairs.append(gamepairsender)
                    pairsender.append(gamepairsender)
                    pairreceiver.append(gamepairsender)
                    gamessender[str(gamepairsender)] = []
                    gamesreceiver[str(gamepairsender)] = []
                if gamepairreceiver not in possible_pairs:
                    possible_pairs.append(gamepairreceiver)
                    pairsender.append(gamepairreceiver)
                    gamessender[str(gamepairreceiver)] = []
                    pairreceiver.append(gamepairreceiver)
                    gamesreceiver[str(gamepairreceiver)] = []
                for pair in pairsender:
                    if gamepairsender == pair:
                        gameentry = {}
                        gameentry[str(game.payoffs)] = equilibria
                        gamessender[str(pair)].append(gameentry)
                        print("sender", pair, len(gamessender[str(pair)]))
                        if len(gamessender[str(pair)]) > 1499:
                            pairsender.remove(pair)
                            print("sender", pair, "done.")
                            filename = ''.join(["sender", str(pair), timestr])
                            with open(filename, 'w') as senderpair:
                                json.dump(gamessender[str(pair)], senderpair)
                for pair in pairreceiver:
                    if gamepairreceiver == pair:
                        gameentry = {}
                        gameentry[str(game.payoffs)] = equilibria
                        gamesreceiver[str(pair)].append(gameentry)
                        print("receiver", pair, len(gamesreceiver[str(pair)]))
                        if len(gamesreceiver[str(pair)]) > 1499:
                            pairreceiver.remove(pair)
                            print("receiver", pair, "done.")
                            filename = ''.join(["receiver", str(pair), timestr])
                            with open(filename, 'w') as receiverpair:
                                json.dump(gamesreceiver[str(pair)], receiverpair)
                #for c in possible_cs:
                    #if gamec == c:
                        #gameentry = {}
                        #gameentry[str(game.payoffs)] = equilibria
                        #games[str(c)].append(gameentry)
                        #print("c", c, len(games[str(c)]))
                        #if len(games[str(c)]) > 1499:
                            #possible_cs.remove(c)
                            #print("c", c, "done.")
                            #filename = ''.join(["c", str(c), timestr])
                            #with open(filename, 'w') as cs:
                                #json.dump(games[str(c)], cs)


        with open("gamessender", "w") as senderfile:
            json.dump(gamessender, senderfile)
        with open("gamesreceiver", "w") as receiverfile:
            json.dump(gamesreceiver, receiverfile)
        with open("pickledsender", "wb") as psender:
            pickle.dump(gamessender, psender)
        with open("pickledreceiver", "wb") as preceiver:
            pickle.dump(gamesreceiver, preceiver)
        with open("pickledpairsender", "wb") as psender:
            pickle.dump(pairsender, psender)
        with open("pickledpairreceiver", "wb") as preceiver:
            pickle.dump(pairreceiver, preceiver)
        with open("pickledpossiblepairs", "wb") as pp:
            pickle.dump(possible_pairs, pp)
        with open("pickledcs", "wb") as pp:
            pickle.dump(possible_cs, pp)
        with open("pickledgames", "wb") as pp:
            pickle.dump(games, pp)

    except KeyboardInterrupt:
        with open("pickledsender", "wb") as psender:
            pickle.dump(gamessender, psender)
        with open("pickledreceiver", "wb") as preceiver:
            pickle.dump(gamesreceiver, preceiver)
        with open("pickledpairsender", "wb") as psender:
            pickle.dump(pairsender, psender)
        with open("pickledpairreceiver", "wb") as preceiver:
            pickle.dump(pairreceiver, preceiver)
        with open("pickledpossiblepairs", "wb") as pp:
            pickle.dump(possible_pairs, pp)
        with open("pickledcs", "wb") as pp:
            pickle.dump(possible_cs, pp)
        with open("pickledgames", "wb") as pp:
            pickle.dump(games, pp)

def findgames_Cnostar(dimension, nofgames):
    """
    Find <nofgames> square games of <dimension>
    with each value of C
    """
    try:
        with open("pickledoutcs", "rb") as pp:
            outstanding_cs = pickle.load(pp)
        with open("pickledcs", "rb") as pp:
            possible_cs = pickle.load(pp)
        with open("pickledgames", "rb") as pp:
            games = pickle.load(pp)
    except FileNotFoundError:
        possible_cs = []
        outstanding_cs = []
        games = {}
        game = ci.Game(ci.payoffs(dimension))
        gamec = game.kendalldistance
        if gamec not in possible_cs:
            equilibria = game.just_equilibria()
            possible_cs.append(gamec)
            outstanding_cs.append(gamec)
            games[str(gamec)] = {}
            games[str(gamec)][str(game.payoffs)] = equilibria
    timestr = time.strftime("%d%b%H-%M")
    try:
        while outstanding_cs != []:
            game = ci.Game(ci.payoffs(dimension))
            gamec = game.kendalldistance
            if gamec not in possible_cs or gamec in outstanding_cs:
                equilibria = game.just_equilibria()
                if gamec not in possible_cs:
                    possible_cs.append(gamec)
                    outstanding_cs.append(gamec)
                    games[str(gamec)] = {}
                for c in possible_cs:
                    if gamec == c:
                        games[str(c)][str(game.payoffs)] = equilibria
                        print("c", c, len(games[str(c)]))
                        if len(games[str(c)]) > nofgames:
                            outstanding_cs.remove(c)
                            print("c", c, "done.")
                            filename = ''.join(["c", str(c), timestr])
                            with open(filename, 'w') as cs:
                                json.dump(games[str(c)], cs)
        with open("pickledoutcs", "wb") as pp:
            pickle.dump(outstanding_cs, pp)
        with open("pickledcs", "wb") as pp:
            pickle.dump(possible_cs, pp)
        with open("pickledgames", "wb") as pp:
            pickle.dump(games, pp)

    except KeyboardInterrupt:
        with open("pickledoutcs", "wb") as pp:
            pickle.dump(outstanding_cs, pp)
        with open("pickledcs", "wb") as pp:
            pickle.dump(possible_cs, pp)
        with open("pickledgames", "wb") as pp:
            pickle.dump(games, pp)


