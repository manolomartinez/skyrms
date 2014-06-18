"""
Library for the paper "Signalling games and modality"
"""

import nim
import setup as s

import numpy as np


class MainGame(s.Strategies):
    """
    Calculate the main game
    """

    def __init__(self):
        investments = nim.coarse_investments(1, 2, [0, 0.25, 0.5, 0.75, 1])
        preparetable = np.insert(investments, 0, 2, axis=1)
        attacks = np.array([[0, 0, 0], [1, 0, 0]])
        acts = np.concatenate((attacks, preparetable))
        s.Strategies.__init__(self, 3, 3, acts)
        self.unpair = 0  # the unprepared air attack is equivalent investing
        # this amount in preparation
        self.unpsea = 0  # ditto for the unprepared sea attack
        self.loseair = 0  # the payoff of losing against the air monster
        self.losesea = 0  # ditto for the sea monster
        self.probs = [0.2, 0.2, 0.3, 0.3]  # The probs of air monster, sea
        self.probair = self.probs[0] / (self.probs[0] + self.probs[1])
        self.probsea = self.probs[1] / (self.probs[0] + self.probs[1])
        self.probtoair = self.probs[2] / (self.probs[2] + self.probs[3])
        self.probtosea = self.probs[3] / (self.probs[2] + self.probs[3])
        # monster, undefined to air monster and to sea monster

    def payoff1stround(self, cost):
        """
        Calculate the payoff matrix for the first round
        """
        return [[pay(self.unpair), self.loseair, self.loseair - cost],
                [self.losesea, pay(self.unpsea), self.losesea - cost],
                [self.probsea * self.losesea + self.probair * pay(self.unpair),
                 self.probair * self.loseair + self.probsea * pay(self.unpsea),
                 0]]

    def payoff2ndround(self, costair, costsea, cost):
        """
        Calculate the payoff matrix for the second round
        """
        gain = np.array([[pay(costair), self.loseair, self.loseair],
                         [self.losesea, pay(costsea), self.losesea]])
        return gain - cost

    def payoff(self, sender, receiver, initialstate, secondstate):
        """
        Calculate the payoff for certain sender and receiver strats, and
        certain first and second states
        """
        firstmessage = int(self.senderstrategies[sender][initialstate])
        firstresponse = np.array(
            self.receiverstrategies[receiver][firstmessage], np.int32)
        firstr = firstresponse[0]
        if firstr == 2:
            costair = firstresponse[1]
            costsea = firstresponse[2]
            cost = costair + costsea
        else:
            cost = 0
        if initialstate != 2 or firstr != 2:
            return self.payoff1stround(cost)[initialstate][firstr]
        else:
            secondmessage = int(self.senderstrategies[sender][secondstate])
            secondresponse = np.array(
                self.receiverstrategies[receiver][secondmessage], np.int32)
            secondr = secondresponse[0]
            return self.payoff2ndround(
                costair, costsea, cost)[secondstate][secondr]

    def avgpayoff(self, sender, receiver):
        """
        Calculate the average payoff for certain sender and receiver strategies
        """
        airmonster = self.payoff(sender, receiver, 0, 1)
        seamonster = self.payoff(sender, receiver, 1, 1)
        toairmonster = self.payoff(sender, receiver, 2, 0)
        toseamonster = self.payoff(sender, receiver, 2, 1)
        avgpoff = (
            1/6 * airmonster + 1/6 * seamonster + 2/3 * toairmonster + 2/3
            * toseamonster)
        return [avgpoff, avgpoff]


def pay(cost):
    """
    A diminishing-returns pay function
    """
    return 5 * (1 + cost)
