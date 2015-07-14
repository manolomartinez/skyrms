import commoninterest as ci
import json
import pickle
import time


def gamepoints(gametype):
    return 8 * gametype[0] + 4 * gametype[1] + 1 * gametype[2]

def convert_type(typelist):
    return [0, typelist[0] + typelist[1], typelist[2] + typelist[0]]

def files_to_json(filenamelist, outputjson):
    games = {}
    for filename in filenamelist:
        print(filename)
        with open(filename, 'r') as fileobject:
            jsfl = json.load(fileobject)
            games.update(jsfl)
    with open(outputjson, 'w') as outputfile:
        json.dump(games, outputfile)
        
def dedupe(jsonfile, outputjson):
    newjson = {}
    with open(jsonfile, 'r') as fileobject:
        ds = json.load(fileobject) #this contains the json
        for key in ds:
            if key not in newjson:
                newjson[key] = ds[key]
    with open(outputjson, 'w') as outputfile:
        json.dump(newjson, outputfile)


def proportion(jsonfile): # Proportion of games with informative equilibria
    games = {}
    with open(jsonfile, 'r') as fileobject:
        games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["proportion", timestr, ".csv"])
    #dictio = {list(item)[0]:item[list(item)[0]] for item in games}
    kendalldict = {}
    for key in games:
        payoffs = eval(key)
        game = ci.Game(payoffs)
        kendall = game.cistar # the value of kendall will be different if we
        # are preparing the data for Figure 1 or for Figure ...
        #sinfos, rinfos, jinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
        sinfos, rinfos, jinfos = game.calculate_info_content(games[key])
        if not kendall in kendalldict:
            kendalldict[kendall] = [0,0,[]]
        kendalldict[kendall][0] += 1
        if any([jinfo > 10e-4 for
                jinfo in jinfos]):
            kendalldict[kendall][1] += 1
        try:
            kendalldict[kendall][2].append(max(jinfos))
        except ValueError:
            pass
            #if 2 < kendall < 3:
                #print(sinfos, rinfos, payoffs)
    print(len(games))
    with open(datapointsname, 'w') as datapoints:
        for key in sorted(kendalldict):
            proportion = kendalldict[key][1]/kendalldict[key][0]
            champion  = max(kendalldict[key][2])
            datapoints.write("{} {} {} {}\n".format(key, proportion, champion,
                kendalldict[key][0]))
        


def champions(jsonfile): # Keep track of champions
    games = {}
    with open(jsonfile, 'r') as fileobject:
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
        #sinfos, rinfos, jinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
        sinfos, rinfos, jinfos = game.calculate_info_content(games[key])
        if not kendall in kendalldict:
            kendalldict[kendall] = []
        #infos = [min(sinfos[element], rinfos[element]) for element in range(len(rinfos)) if
                #rinfos[element] > 10e-8 and sinfos[element] > 10e-8]
        infos = max(jinfos)
        try:
            kendalldict[kendall].append(infos)
            #if kendall == 3 and infos > 10e-8:
                #print(game.sender, game.receiver,infos, kendall,
                        #game.same_best())
        except ValueError:
            pass
    print(len(games))
    with open(datapointsname, 'w') as datapoints:
        for key in sorted(kendalldict):
            try:
                champion = max(kendalldict[key])
                datapoints.write("{} {} {}\n".format(key, champion, len(kendalldict[key])))
            except IndexError:
                pass


def query(jsonfile):
    #games = {}
    #with open(picklefile, 'rb') as fileobject:
    #    games = pickle.load(fileobject)
        #dictio = {list(item)[0]:item[list(item)[0]] for item in jsfl}
        #games.update(dictio)
        #games.update(jsfl)
    with open(jsonfile, 'r') as fileobject:
        #line = eval(fileobject.readlines()[0])
        #print(line[0])
        #games = {list(item)[0]:item[list(item)[0]] for item in line}
        games = json.load(fileobject)
        #games.update(json.load(dictio))
#    receivers = {}
#    for pair in games:
#            print(pair)
#            print(receivers)
#            for gamedata in games[pair]:
#                key = list(gamedata.keys())[0]
#                payoffs = eval(key)
#                game = ci.Game(payoffs)
#                c = game.kendalldistance
#                sender = game.kendallsender
#                if c < 10e-4 and sender < 10e-4:
#                    receiver = game.kendallreceiver
#                    if str([c, receiver]) not in receivers:
#                        receivers[str([c, receiver])] = 0
#                    receivers[str([c, receiver])] += 1
#    totalrec = sum([receivers[rec] for rec in receivers])
#    print(len(receivers), totalrec)
#    for receiver in receivers:
#        try:
#            print("{}: {}%".format(receiver, receivers[receiver]/totalrec*100))
#        except ZeroDivisionError:
#            pass
#
#    return 0
    # for cs in games:
    for key in games:
        # payoffskey = list(key.keys())[0]
        payoffs = eval(key)
        game = ci.Game(payoffs)
        if game.kendalldistance > 2.8:
            #sinfos, rinfos, jinfos = game.calculate_info_content(games[key])
            #if max(jinfos) == 0:
            #    print(payoffs)
            _, _, jinfos = game.calculate_info_content(games[key])
            ##if 0 < max(jinfos) < 10e-2:
            #    if 10e-6 < max(jinfos) < 10e-3:
            #        print("theoddone", max(sinfos), max(rinfos), max(jinfos))
            #        print(game.payoffs)
                #print(counter)
                #print(counter2)
                #counter += 1
                #if 0 < max(jinfos) < 10e-4:
                    ##print("residue: ", max(sinfos), max(rinfos), max(jinfos))
                    #print(oldpair)
                #elif max(jinfos) == 0:
                    #print("correct: ", max(sinfos), max(rinfos), max(jinfos))
                    #print(oldpair)
                #else:
            if max(jinfos) > 10e-4:
                print("c: {}".format(game.kendalldistance))
                print("payoffs: {}".format(payoffs))
                print("info: {}".format(max(jinfos)))

                
        #    pair = [game.kendalldistance, game.kendallsender]
        #    newpairs[str(pair)][0] += 1
        #    sinfos, rinfos, jinfos = game.calculate_info_content(games[key])
        #    if max(jinfos) > 0:
        #        newpairs[str(pair)][1] += 1
    #print(newpairs)
    #allgames = 0
    #infogames = 0
    #for pair in sorted(newpairs):
    #    print("pair {}".format(pair))
    #    print("games: {}".format(newpairs[pair][0]))
    #    print("proportion: {}".format(newpairs[pair][1]/newpairs[pair][0]))
    #    allgames += newpairs[pair][0]
    #    infogames += newpairs[pair][1]
    #print(allgames, infogames)
             

def withintra(jsonfile):  # Taking into account intra-Kendalls
    games = {}
    with open(jsonfile, 'r') as fileobject:
        games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsnamereceiver = ''.join(["withintrasender", timestr, ".csv"])
    kendalldictreceiver = {}
    intrakendallsreceiver = []
    kendalls = []
    with open(datapointsnamereceiver, 'w') as datapointsreceiver:
        for key in games:
            payoffs = eval(key)
            game = ci.Game(payoffs)
            kendall = game.cistar
            #kendallsender = round(game.petersender, 2)
            kendallreceiver = game.starsender
            if kendall not in kendalls:
                kendalls.append(kendall)
            #if kendallsender not in intrakendallssender:
                #intrakendallssender.append(kendallsender)
            if kendallreceiver not in intrakendallsreceiver:
                intrakendallsreceiver.append(kendallreceiver)
            #sinfos, rinfos, jinfos = game.calculate_info_content(eval(games[key]["equilibria"]))
            sinfos, rinfos, jinfos = game.calculate_info_content(games[key])
            #duosender = str([kendall, kendallsender])
            #if not duosender in kendalldictsender:
                #kendalldictsender[duosender] = [0,0,[]]
            #kendalldictsender[duosender][0] += 1
            #try:
                #if max(jinfos) > 10e-6:
                    #kendalldictsender[duosender][1] += 1
                    #if eval(duosender)[0] > 5.9:
                        #print(game.payoffs)
                #kendalldictsender[duosender][2].append(max(jinfos))
            #except ValueError:
                #kendalldictsender[duosender][2].append(0)
            duoreceiver = str([kendall, kendallreceiver])
            if not duoreceiver in kendalldictreceiver:
                kendalldictreceiver[duoreceiver] = [0, 0, []]
            kendalldictreceiver[duoreceiver][0] += 1
            try:
                if max(jinfos) > 10e-4:
                    kendalldictreceiver[duoreceiver][1] += 1
                kendalldictreceiver[duoreceiver][2].append(max(jinfos))
            except ValueError:
                kendalldictreceiver[duoreceiver][2].append(0)
        totalgames = []
        #for key in sorted(kendalldictsender):
            #proportion = kendalldictsender[key][1]/kendalldictsender[key][0]
            #values  = eval(key)
            #totalgames.append(kendalldictsender[key][0])
            #datapointssender.write("{} {} {} {} {} {}\n".format(values[0], values[1],
                #kendalldictsender[key][1], kendalldictsender[key][0],
                #proportion, max(kendalldictsender[key][2])))
            #print(sum(totalgames))
        for key in sorted(kendalldictreceiver):
            proportion = kendalldictreceiver[
                key][1]/kendalldictreceiver[key][0]
            values = eval(key)
            #totalgames.append(kendalldictreceiver[key][0])
            datapointsreceiver.write(
                "{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictreceiver[key][1], kendalldictreceiver[key][0],
                proportion, max(kendalldictreceiver[key][2])))
            #print(sum(totalgames))
            #print(intrakendallssender)
            #print(intrakendallsreceiver)
            #print(kendalls)


def cloud(jsonfile): # Every game and its best equilibrium
    games = {}
    with open(jsonfile, 'r') as fileobject:
        games.update(json.load(fileobject))
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["cloudinfo", timestr])
    kendalldict = {}
    with open(datapointsname, 'w') as datapoints:
        for key in games:
            payoffs = eval(key)
            game = ci.Game(payoffs)
            kendall = game.kendalldistances
            sinfos, rinfos, jinfos = game.calculate_info_content(games[key])
            datapoints.write("{} {}\n".format(kendall, max(jinfos)))

def withintraties(pickledsender, pickledreceiver):  # Taking into account intra-Kendalls
    print("tehstuff")
    with open(pickledsender, "rb") as psender:
        gamessender = pickle.load(psender)
    with open(pickledreceiver, "rb") as preceiver:
        gamesreceiver = pickle.load(preceiver)
    timestr = time.strftime("%d%b%H-%M")
    datapointsnamereceiver = ''.join(["withintrareceiver", timestr, ".csv"])
    datapointsnamereceiver500 = ''.join(["withintrareceiver500", timestr, ".csv"])
    datapointsnamesender = ''.join(["withintrasender", timestr, ".csv"])
    datapointsnamesender500 = ''.join(["withintrasender500", timestr, ".csv"])
    kendalldictreceiver = {}
    kendalldictreceiver500 = {}
    kendalldictsender = {}
    kendalldictsender500 = {}
    intrakendallsreceiver = []
    kendalls = []
    for key in gamesreceiver:
        print("receiver pair:", key)
        kendalldictreceiver[key] = [0, 0, []]
        kendalldictreceiver500[key] = [0, 0, []]
        print("converting to dict")
        dictio = {list(item)[0]:item[list(item)[0]] for item in gamesreceiver[key]}
        print("done:", len(dictio))
        for gamekey in dictio:
            payoffs = eval(gamekey)
            game = ci.Game(payoffs)
            sinfos, rinfos, jinfos = game.calculate_info_content(
                dictio[gamekey])
            kendalldictreceiver[key][0] += 1
            if kendalldictreceiver500[key][0] <= 500:
                kendalldictreceiver500[key][0] += 1
            try:
                if max(jinfos) > 10e-4:
                    kendalldictreceiver[key][1] += 1
                    if kendalldictreceiver500[key][0] <= 500:
                        kendalldictreceiver500[key][1] += 1
                kendalldictreceiver[key][2].append(max(jinfos))
                if kendalldictreceiver500[key][0] <= 500:
                    kendalldictreceiver500[key][2].append(max(jinfos))
            except ValueError:
                kendalldictreceiver[key][2].append(0)
                if kendalldictreceiver500[key][0] <= 500:
                    kendalldictreceiver500[key][2].append(0)
    with open(datapointsnamereceiver, 'w') as datapointsreceiver:
        for key in sorted(kendalldictreceiver):
            proportion = kendalldictreceiver[
                key][1]/kendalldictreceiver[key][0]
            values = eval(key)
            datapointsreceiver.write(
                "{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictreceiver[key][1], kendalldictreceiver[key][0],
                proportion, max(kendalldictreceiver[key][2])))
    with open(datapointsnamereceiver500, 'w') as datapointsreceiver:
        for key in sorted(kendalldictreceiver):
            proportion = kendalldictreceiver[
                key][1]/kendalldictreceiver[key][0]
            values = eval(key)
            datapointsreceiver.write(
                "{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictreceiver[key][1], kendalldictreceiver[key][0],
                proportion, max(kendalldictreceiver[key][2])))
    for key in gamessender:
        print("sender pair:", key)
        kendalldictsender[key] = [0, 0, []]
        kendalldictsender500[key] = [0, 0, []]
        dictio = {list(item)[0]:item[list(item)[0]] for item in gamessender[key]}
        for gamekey in dictio:
            payoffs = eval(gamekey)
            game = ci.Game(payoffs)
            sinfos, rinfos, jinfos = game.calculate_info_content(
                dictio[gamekey])
            kendalldictsender[key][0] += 1
            if kendalldictsender500[key][0] <= 500:
                kendalldictsender500[key][0] += 1
            try:
                if max(jinfos) > 10e-4:
                    kendalldictsender[key][1] += 1
                    if kendalldictsender500[key][0] <= 500:
                        kendalldictsender500[key][1] += 1
                kendalldictsender[key][2].append(max(jinfos))
                if kendalldictsender500[key][0] <= 500:
                    kendalldictsender500[key][2].append(max(jinfos))
            except ValueError:
                kendalldictsender[key][2].append(0)
                if kendalldictsender500[key][0] <= 500:
                    kendalldictsender500[key][2].append(0)
    with open(datapointsnamesender, 'w') as datapointssender:
        for key in sorted(kendalldictsender):
            proportion = kendalldictsender[
                key][1]/kendalldictsender[key][0]
            values = eval(key)
            datapointssender.write(
                "{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictsender[key][1], kendalldictsender[key][0],
                proportion, max(kendalldictsender[key][2])))
    with open(datapointsnamesender500, 'w') as datapointssender:
        for key in sorted(kendalldictsender):
            proportion = kendalldictsender[
                key][1]/kendalldictsender[key][0]
            values = eval(key)
            datapointssender.write(
                "{} {} {} {} {} {}\n".format(values[0], values[1],
                kendalldictsender[key][1], kendalldictsender[key][0],
                proportion, max(kendalldictsender[key][2])))
    
