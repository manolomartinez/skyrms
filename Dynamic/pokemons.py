"""
Library for the paper "Signalling games and modality"
"""

import setup as s


class MainGame(s.Strategies):
    """
    Calculate the main game
    """

    def __init__(self):
        preparetable = [[2, i, j] for i in range(3) for j in range(3-i)]
        acts = [[0], [1]] + preparetable
        s.Strategies.__init__(self, 3, 3, acts)
        self.unpair = 0
        self.unpsea = 0
        self.loseair = 4
        self.losesea = 4

    def payoff1stround(self):
        """
        Calculate the payoff matrix for the first round
        """
        return [[pay(self.unpair), self.loseair, self.loseair -
                 self.costair], [self.losesea, pay(self.unpsea),
                                 self.losesea - self.costsea],
                [(self.losesea + pay(self.unpair)) / 2,
                 (self.loseair + pay(self.unpsea))/2, 0]]

    def payoff2ndround(self):
        """
        Calculate the payoff matrix for the second round
        """
        gain = [[pay(self.costair), self.loseair, self.loseair],
                [self.losesea, pay(self.costsea), self.losesea]]
        return [[member - self.costair - self.costsea
                 for member in row] for row in gain]

    def payoff(self, sender, receiver, initialstate, secondstate):
        """
        Calculate the payoff for certain sender and receiver strats, and
        certain first and second states
        """
        firstmessage = self.senderstrategies[sender][initialstate]
        firstresponse = self.receiverstrategies[receiver][firstmessage]
        firstr = firstresponse[0]
        try:
            self.costair = firstresponse[1]
            self.costsea = firstresponse[2]
        except IndexError:
            self.costair = 0
            self.costsea = 0
        if initialstate != 2 or len(firstresponse) == 1:
            return self.payoff1stround()[initialstate][firstr]
        else:
            secondmessage = self.senderstrategies[sender][secondstate]
            secondresponse = self.receiverstrategies[receiver][secondmessage]
            secondr = secondresponse[0]
            return self.payoff2ndround()[secondstate][secondr]

    def avgpayoff(self, sender, receiver):
        """
        Calculate the average payoff for certain sender and receiver strategies
        """
        airmonster = self.payoff(sender, receiver, 0, 1)
        seamonster = self.payoff(sender, receiver, 1, 1)
        toairmonster = self.payoff(sender, receiver, 2, 0)
        toseamonster = self.payoff(sender, receiver, 2, 1)
        return (1/6 * airmonster + 1/6 * seamonster + 2/3 * toairmonster + 2/3
                * toseamonster)


def pay(cost):
    """
    A diminishing-returns pay function
    """
    return 20 - 10/s.np.exp(cost)
