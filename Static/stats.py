import commoninterest as ci, json, time
from scipy.stats import scoreatpercentile

def gamepoints(gametype):
    return 8 * gametype[0] + 4 * gametype[1] + 1 * gametype[2]

def convert_type(typelist):
    return [0, typelist[0] + typelist[1], typelist[2] + typelist[0]]

def main(filenamelist):
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    typedict = {str(payofftype):[0,0, [],[]] for payofftype in types}
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    print(len(games))
    for key in games:
        payoffs = eval(key)
        game = ci.Game(payoffs)
        parameter = game.minkendallmoddistance
        gametype = str(game.payoff_type())
        infos = game.calculate_info_content(eval(games[key]["equilibria"]))
        typedict[gametype][0] += 1
        typedict[gametype][2].append(parameter)

        if max(infos) > 0:
            typedict[gametype][1] += 1
        else:
            typedict[gametype][3].append(parameter)
    for key in typedict:
        print("{}: {}".format(key, typedict[key][0]))
        try:
            print("{}: {}%".format(key, typedict[key][1]/typedict[key][0] * 100))
            print("parameter with info: {}".format
                    #(sum(typedict[key][2])/len(typedict[key][2])))
                    (max(typedict[key][2])))
            print("parameter without info: {}".format
                    #(sum(typedict[key][3])/len(typedict[key][3])))
                    (min(typedict[key][3])))
        except:
            pass

def main2(filenamelist):
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    typedict = {str(payofftype):[0,0, [],[]] for payofftype in types}
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    print(len(games))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    with open(datapointsname, 'w') as datapoints:
        for key in games:
            payoffs = eval(key)
            game = ci.Game(payoffs)
            parameter = game.kendalldistance
            gametype = str(game.payoff_type())
            infos = game.calculate_info_content(eval(games[key]["equilibria"]))
            datapoints.write("{} {}\n".format(parameter, max(infos)))
                    

def main3(filenamelist):
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4)]
    typedict = {str(payofftype):[0,0] for payofftype in types}
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    for key in games:
        payoffs = eval(key)
        game = ci.Game(payoffs)
        gtype = game.payoff_type()
        gtype = convert_type(gtype)
        gametype = str(gtype)
        infos = game.calculate_info_content(eval(games[key]["equilibria"]))
        typedict[gametype][0] += 1
        if max(infos) > 0:
            typedict[gametype][1] += 1
    with open(datapointsname, 'w') as datapoints:
        for key in sorted(typedict):
            if eval(key) == convert_type(eval(key)):
                points = gamepoints(eval(key)) 
                proportion = typedict[key][1]/typedict[key][0]
                datapoints.write("{} {}\n".format(points, proportion))

def proportion(filenamelist): # Proportion of games with informative equilibria
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["proportion", timestr])
    kendalldict = {}
    for key in games:
        payoffs = eval(key)
        game = ci.Game(payoffs)
        kendall = game.kendalldistance
        sinfos, rinfos, jinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
        if not kendall in kendalldict:
            kendalldict[kendall] = [0,0]
        kendalldict[kendall][0] += 1
        if any([jinfo > 10e-8 for
                jinfo in jinfos]):
            kendalldict[kendall][1] += 1
            #if 2 < kendall < 3:
                #print(sinfos, rinfos, payoffs)
    print(len(games))
    with open(datapointsname, 'w') as datapoints:
        for key in sorted(kendalldict):
            proportion = kendalldict[key][1]/kendalldict[key][0]
            datapoints.write("{} {} {}\n".format(key, proportion,
                kendalldict[key][0]))
        

def champions(filenamelist, percentile): # Keep track of champions
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["champions", timestr])
    kendalldict = {}
    for key in games:
        payoffs = eval(key)
        #if len(payoffs) != len(set(payoffs)):
            #pass
        game = ci.Game(payoffs)
        kendall = game.kendalldistance
        sinfos, rinfos, jinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
        if not kendall in kendalldict:
            kendalldict[kendall] = []
        #infos = [min(sinfos[element], rinfos[element]) for element in range(len(rinfos)) if
                #rinfos[element] > 10e-8 and sinfos[element] > 10e-8]
        infos = max(jinfos)
        try:
            kendalldict[kendall].append(infos)
            if kendall == 1  and infos > 1.5:
                print(game.sender, game.receiver,infos, kendall,
                        game.same_best())
        except ValueError:
            pass
    print(len(games))
    with open(datapointsname, 'w') as datapoints:
        for key in sorted(kendalldict):
            try:
                champion = scoreatpercentile(kendalldict[key], percentile)
                champion95 = scoreatpercentile(kendalldict[key], 95)
                champion50 = scoreatpercentile(kendalldict[key], 50)
                datapoints.write("{} {} {} {} {}\n".format(key, champion,
                    champion95, champion50, len(kendalldict[key])))
            except IndexError:
                pass

def withintra(filenamelist): # Taking into account intra-Kendalls
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsnamesender = ''.join(["withintrasender", timestr])
    datapointsnamereceiver = ''.join(["withintrareceiver", timestr])
    kendalldictsender = {}
    kendalldictreceiver = {}
    with open(datapointsnamesender, 'w') as datapointssender, open(
            datapointsnamereceiver, 'w') as datapointsreceiver:
        for key in games:
            payoffs = eval(key)
            game = ci.Game(payoffs)
            kendall = game.kendalldistance
            kendallsender = game.kendallsender
            kendallreceiver = game.kendallreceiver
            # Grouping
            if kendallsender < 1:
                kendallsender = 0.5
            elif kendallsender == 1:
                kendallsender = 1
            elif kendallsender < 2:
                kendallsender = 1.5
            elif kendallsender == 2:
                kendallsender = 2
            else:
                kendallsender = 2.5
            if kendallreceiver < 1:
                kendallreceiver = 0.5
            elif kendallreceiver == 1:
                kendallreceiver = 1
            elif kendallreceiver < 2:
                kendallreceiver = 1.5
            elif kendallreceiver == 2:
                kendallreceiver = 2
            else:
                kendallreceiver = 2.5
            
            sinfos, rinfos, jinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
            duosender = str([kendall, kendallsender])
            if not duosender in kendalldictsender:
                kendalldictsender[duosender] = [0,0,[]]
            kendalldictsender[duosender][0] += 1
            try:
                if max(jinfos)>0:
                    kendalldictsender[duosender][1] += 1
                kendalldictsender[duosender][2].append(max(jinfos))
            except ValueError:
                kendalldictsender[duosender][2].append(0)
            duoreceiver = str([kendall, kendallreceiver])
            if not duoreceiver in kendalldictreceiver:
                kendalldictreceiver[duoreceiver] = [0,0,[]]
            kendalldictreceiver[duoreceiver][0] += 1
            try:
                if max(jinfos)>0:
                    kendalldictreceiver[duoreceiver][1] += 1
                kendalldictreceiver[duoreceiver][2].append(max(jinfos))
            except ValueError:
                kendalldictreceiver[duoreceiver][2].append(0)
        totalgames = []
        for key in sorted(kendalldictsender):
            proportion = kendalldictsender[key][1]/kendalldictsender[key][0]
            values  = eval(key)
            totalgames.append(kendalldictsender[key][0])
            datapointssender.write("{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictsender[key][1], kendalldictsender[key][0],
                proportion, max(kendalldictsender[key][2])))
            print(sum(totalgames))
        for key in sorted(kendalldictreceiver):
            proportion = kendalldictreceiver[key][1]/kendalldictreceiver[key][0]
            values  = eval(key)
            totalgames.append(kendalldictreceiver[key][0])
            datapointsreceiver.write("{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictreceiver[key][1], kendalldictreceiver[key][0],
                proportion, max(kendalldictreceiver[key][2])))
            print(sum(totalgames))

def cloud(filenamelist): # Every game and its best equilibrium
    games = {}
    for filename in filenamelist:
        with open(filename, 'r') as fileobject:
            games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["cloudinfo", timestr])
    kendalldict = {}
    with open(datapointsname, 'w') as datapoints:
        for key in games:
            payoffs = eval(key)
            game = ci.Game(payoffs)
            kendall = game.kendallwithdistances
            sinfos, rinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
            datapoints.write("{} {}\n".format(kendall, max(sinfos)))

