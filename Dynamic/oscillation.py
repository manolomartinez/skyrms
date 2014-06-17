# coding: utf-8
import numpy as np
import pickle as p
import imp
import calculations as c
import stats

get_ipython().magic('cd Oscillation/35_97/')
strats = c.s.Strategies(3,3,3)
game = c.s.Game([35, 97, 75, 36, 40, 85, 10, 73, 32, 34, 61, 15, 91, 0, 78, 34, 53, 70], 0, strats)
c.t1 = 100000
with open("sinitrinit", "rb") as sr:
    sinit, rinit = p.load(sr)
data = c.one_run_ode(game, sinit, rinit)
