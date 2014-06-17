import start
import json
get_ipython().magic('cd Oscillation/')
with open("nocommoninterest", "r") as nci:
    games = json.load(nci)
    
strats = start.c.s.Strategies(3,3,3)
with open("results", "w") as results:
    for game in games:
        gameobj = start.c.s.Game(game, 0, strats)
        data = start.c.one_basin_ode(gameobj, 10)
        infouse = [start.stats.info_use_per_evol(point) for point in data]
        if max(infouse) > 1e-5:
            results.write("{}\t{}\n".format(game, infouse))
