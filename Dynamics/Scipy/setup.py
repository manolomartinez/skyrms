import numpy as np
import itertools as it
np.set_printoptions(precision=4)


class Strategies:
    """
    Construct strategies
    """
    def __init__(self, nstates, nsignals, nacts):
        self.states = np.arange(nstates)
        self.signals = np.arange(nsignals)
        self.acts = np.arange(nacts)
        self.signalsprobs = np.identity(nsignals)
        self.actsprobs = np.identity(nacts)
        self.chances = np.array([1/nstates for i in self.states])
        self.senderstrategies = np.array(list(it.product(self.signals,
                                                         repeat=len(self.states))))
        self.receiverstrategies = np.array(list(it.product(self.acts,
                                                           repeat=len(self.signals))))
        self.senderstrategiesprobs = np.array(list(it.product(self.signalsprobs,
                                                         repeat=len(self.states))))
        self.receiverstrategiesprobs = np.array(list(it.product(self.actsprobs,
                                                           repeat=len(self.signals))))
        self.lrs = len(self.receiverstrategies)
        self.lss = len(self.senderstrategies)

    def random_sender(self):
        """
        Return a random sender population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lss)])

    def random_receiver(self):
        """
        Return a random receiver population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lrs)])


class Game:
    """
    Compute a game
    """
    def __init__(self, payoffs, mutationrate, strats):
        self.payoffs = payoffs
        self.strats = strats
        self.senderpayoffs = np.array([[
            self.avgpayoff(i, j)[0] for j in range(self.strats.lss)] for i in
            range(self.strats.lrs)])
        self.receiverpayoffs = np.array([[
            self.avgpayoff(j, i)[1] for j in range(self.strats.lss)] for i in
            range(self.strats.lrs)])
        self.mm = self.mutationmatrix(mutationrate)


    def mutationmatrix(self, mutation):
        """
        Calculate a mutation matrix with mutation rate
        given by mutation
        """
        return np.array([[1 - mutation if i==j else mutation/(self.strats.lss - 1) for i in
                               np.arange(self.strats.lss)] for j in
                         np.arange(self.strats.lrs)])

    def payoff(self, sender, receiver, state):
        message = self.strats.senderstrategies[sender, state]
        act = self.strats.receiverstrategies[receiver, message]
        return [self.payoffs[2 * (len(self.strats.signals) * state + act)],
                self.payoffs[2 * (len(self.strats.signals) * state + act) + 1]]

    def avgpayoff(self, sender, receiver):
        return [sum([1/len(self.strats.states) * self.payoff(
            sender, receiver, state)[0] for state in self.strats.states]),
                sum([1/len(self.strats.states) * self.payoff(
                sender, receiver, state)[1] for state in self.strats.states])]

    def sender_avg_payoff(self, sender, receiver):
        return sender.dot(self.senderpayoffs.dot(receiver))

    def receiver_avg_payoff(self, receiver, sender):
        return receiver.dot(self.receiverpayoffs.dot(sender))

    def dX_dt(self, X, t):
        """
        Calculate the system of odes
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        Xsplit = np.array_split(X, 2)
        senderpops = Xsplit[0]
        receiverpops = Xsplit[1]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs * senderpops[...,None]).dot(receiverpops).dot(self.mm) - senderpops*avgfitnesssender
        receiverdot = (self.receiverpayoffs * receiverpops[...,None]).dot(senderpops).dot(self.mm) - receiverpops*avgfitnessreceiver
        return np.concatenate((senderdot, receiverdot))

    def tile14(self, i, j, us, them, payoffs, avgfitness):
        """
        Calculate coefficients of the 1st and 4th tiles 
        of the Jacobian matrix
        """
        return (self.mm[j, i] - us[i]) * sum([them[k] * payoffs[j, k] for k in
                                              range(self.strats.lrs)]) - (i == j)*avgfitness

    def tile14_repl(self, i, j, us, them, payoffs, avgfitness):
        """
        Calculate coefficients of the 1st and 4th tiles 
        of the Jacobian matrix for replicator dynamics
        """
        return ((i == j) - us[i]) * sum([them[k] * payoffs[j, k] for k in
                                              range(self.strats.lrs)]) - (i == j)*avgfitness

    def tile23(self, i, j, us, them, payoffs):
        """
        Calculate coefficients of the 2st and 3th tiles 
        of the Jacobian matrix
        """
        return sum([(self.mm[k, i] - us[i]) * us[k] * payoffs[k,j] for k
                         in range(self.strats.lss)])

    def tile23_repl(self, i, j, us, them, payoffs):
        """
        Calculate coefficients of the 2st and 3th tiles 
        of the Jacobian matrix
        """
        return us[i] * payoffs[i, j] - us[i] * sum([us[k] * payoffs[k,j] for k
                         in range(self.strats.lss)])

    def jacobian(self, X, t=0):
        """ Calculate the Jacobian of the system
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        Xsplit = np.array_split(X, 2)
        senderpops = Xsplit[0]
        receiverpops = Xsplit[1]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)

        tile1 = np.array([[self.tile14_repl(i,j, senderpops, receiverpops,
                                       self.senderpayoffs,
                                  avgfitnesssender) for i in
                           range(self.strats.lss)] for j in
                          range(self.strats.lss)])

        tile2 = np.array([[self.tile23_repl(i,j, senderpops, receiverpops,
                                       self.senderpayoffs) for i in range(self.strats.lss)] for j in
                          range(self.strats.lss)])

        tile3 = np.array([[self.tile23_repl(i,j, receiverpops, senderpops,
                                       self.receiverpayoffs) for i in range(self.strats.lrs)] for j in
                          range(self.strats.lrs)])

        tile4 = np.array([[self.tile14_repl(i,j, receiverpops, senderpops,
                                  self.receiverpayoffs, avgfitnessreceiver) for i in range(self.strats.lrs)] for j in
                          range(self.strats.lrs)])

        lefthalf = np.vstack((tile1, tile2))
        righthalf = np.vstack((tile3, tile4))
        jac = np.hstack((lefthalf, righthalf))
        return jac

    def jacobian_alobestia(self, X, t=0):
        """ Calculate the Jacobian of the system
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        Xsplit = np.array_split(X, 2)
        senderpops = Xsplit[0]
        receiverpops = Xsplit[1]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops) # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[...,None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[...,None]
        tile1 = (self.mm - senderpops[...,None]) * yS - np.eye(self.strats.lss,
                                                         self.strats.lrs) * avgfitnesssender
        tile2 = (self.mm - senderpops).transpose().dot(xS)
        tile3 = (self.mm - receiverpops).transpose().dot(yR)
        tile4 = (self.mm - receiverpops[...,None]) * xR - np.eye(self.strats.lrs,
                                                         self.strats.lss) * avgfitnessreceiver
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        return jac

    def jacobian_replicator(self, X, t):
        """ Calculate the Jacobian of the system
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        Xsplit = np.array_split(X, 2)
        senderpops = Xsplit[0]
        receiverpops = Xsplit[1]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops) # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[...,None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[..., None]
        tile1 = (np.eye(self.strats.lss, self.strats.lrs) - senderpops[...,None]) * yS[..., None] - np.eye(self.strats.lss,
                                                         self.strats.lrs) * avgfitnesssender
        tile2 = xS - senderpops.dot(self.senderpayoffs) * senderpops[...,None]
        tile3 = yR - receiverpops.dot(self.receiverpayoffs) * receiverpops[...,None]
        tile4 = (np.eye(self.strats.lss, self.strats.lrs) -
                 receiverpops[...,None]) * xR[..., None] - np.eye(self.strats.lss, self.strats.lrs) * avgfitnessreceiver
        lefthalf = np.vstack((tile1, tile2))
        righthalf = np.vstack((tile3, tile4))
        jac = np.hstack((lefthalf, righthalf))
        return jac

class Nash:
    """
    Calculate Nash equilibria
    """
    def __init__(self, game):
        self.game = game

    def receivers_vs_sender(self, sender):
        receivers = np.identity(self.game.strats.lrs)
        return [self.game.receiver_avg_payoff(receiver, sender)
                         for receiver in receivers]

    def senders_vs_receiver(self, receiver):
        senders = np.identity(self.game.strats.lss)
        return [self.game.sender_avg_payoff(sender, receiver)
                         for sender in senders]

    def is_Nash(self, sender, receiver):
        """
        Find out if sender and receiver are a Nash eqb
        """
        payoffsender = self.game.sender_avg_payoff(sender, receiver)
        payoffreceiver = self.game.receiver_avg_payoff(receiver, sender)
        senderisbest = abs(payoffsender -
                           max(self.senders_vs_receiver(receiver))) < 1e-5
        receiverisbest = abs(payoffreceiver -
                             max(self.receivers_vs_sender(sender))) < 1e-5
        return senderisbest and receiverisbest

class Information:
    """
    Calculate mutual information between populations
    """
    def __init__(self, strats, popvector):
        pops = np.array_split(popvector, 2)
        self.sender = pops[0]
        self.receiver = pops[1]
        self.strats = strats
        self.msg_cond_on_states, self.acts_cond_on_msg = self.population_to_mixed_strat()

    def population_to_mixed_strat(self):
        """
        Take sender and receiver populations and output the overall sender and
        receiver rules
        """
        mixedstratsender = self.strats.senderstrategiesprobs * self.sender[:,
                                                                      np.newaxis,
                                                                      np.newaxis]
        mixedstratreceiver = self.strats.receiverstrategiesprobs * self.receiver[:,
                                                                            np.newaxis,
                                                                            np.newaxis]
        return sum(mixedstratsender), sum(mixedstratreceiver)

    def joint_states_messages(self):
        return from_conditional_to_joint(self.strats.chances,
                                         self.msg_cond_on_states)

    def joint_messages_acts(self):
        joint_s_m = self.joint_states_messages()
        uncondmessages = sum(joint_s_m)
        return from_conditional_to_joint(uncondmessages,
                                         self.acts_cond_on_msg)

    def joint_states_acts(self):
        joint_s_m = self.joint_states_messages()
        return joint_s_m.dot(self.acts_cond_on_msg)

    def mutual_info_states_acts(self):
        return mutual_info_from_joint(self.joint_states_acts())

def conditional_entropy(conds, unconds):
    """
    Take a matrix of probabilities of the column random variable (r. v.)
    conditional on the row r.v.; and a vector of unconditional
    probabilities of the row r. v.. Calculate the conditional entropy of
    column r. v. on row r. v. That is:
    Input: 
        [[P(B1|A1), ...., P(Bn|A1)],..., [P(B1|Am),...,P(Bn|Am)]]
        [P(A1), ..., P(Am)]
    Output: 
        H(B|A)
    """
    return unconds.dot(np.apply_along_axis(entropy, 1, conds))

def mutual_info_from_joint(matrix):
    """ 
    Take a matrix of joint probabilities and calculate the mutual information
    between the row and column random variables
    """
    row_unconditionals = sum(matrix.transpose())
    column_unconditionals = sum(matrix)
    conditionals = from_joint_to_conditional(matrix)
    uncond_entropy = entropy(column_unconditionals)
    cond_entropy = conditional_entropy(conditionals, row_unconditionals)
    return uncond_entropy - cond_entropy

def unconditional_probabilities(unconditional_input, strategy):
    """
    Calculate the unconditional probability of messages for sender, or
    signals for receiver, given the unconditional probability of states
    (for sender) or of messages (for receiver)
    """
    return strategy * unconditional_input

def from_joint_to_conditional(matrix):
    """
    Normalizes a matrix row-wise
    """
    return matrix/sum(matrix.transpose())

def from_conditional_to_joint(unconds, conds):
    """
    Take a matrix of conditional probabilities of the column random variable on
    the row r. v., and a vector of unconditional probabilities of the row r. v.
    and output a matrix of joint probabilities.

    Input: 
            [[P(B1|A1), ...., P(Bn|A1)],..., [P(B1|Am),...,P(Bn|Am)]]
            [P(A1), ..., P(Am)]
    Output: 
            [[P(B1,A1), ...., P(Bn,A1)],..., [P(B1,Am),...,P(Bn,Am)]]
    """
    return conds * unconds[..., None]

def entropy(vector):
    """
    Calculate the entropy of a vector
    """
    nonzero = vector[vector > 1e-10]
    return -nonzero.dot(np.log2(nonzero))


def escalar_product_map(matrix, vector):
    """
    Take a matrix and a vector and return a matrix consisting of each element
    of the vector multiplied by the corresponding row of the matrix
    """
    trm = matrix.transpose()
    return np.vectorize(np.dot)(trm, vector).transpose()
