import numpy as np
import itertools as it
np.set_printoptions(precision=4)


class Strategies:
    """
    Construct strategies
    """
    def __init__(self, nstates, nsignals, nacts):
        self.states = int_or_list(nstates)
        self.signals = int_or_list(nsignals)
        self.acts = int_or_list(nacts)
        self.signalsprobs = np.identity(len(self.signals))
        self.actsprobs = np.identity(len(self.acts))
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


class MixedStrategies(Strategies):
    """
    Construct a set of strategies that include mixed ones
    """
    def __init__(self, nstates, nsignals, nacts, increment):
        """
        <increment> gives the minimum weight (!=0) that a pure strat may have
        in a mixed one
        """
        Strategies.__init__(nstates, nsignals, nacts)


class Random_Payoffs:
    """
    Take a list of payoffs and a strats object and calculate a function that
    gives the average payoff for sender and receiver strategies
    """
    def __init__(self, payoffs, strats):
        self.payoffs = payoffs
        self.strats = strats

    def payoff(self, sender, receiver, state):
        """
        Return the payoff for strategies <sender> and <receiver> in <state> in
        a list
        """
        message = self.strats.senderstrategies[sender, state]
        act = self.strats.receiverstrategies[receiver, message]
        return np.array(
            [self.payoffs[2 * (len(self.strats.acts) * state + act)],
                self.payoffs[2 * (len(self.strats.acts) * state + act) + 1]])

    def avgpayoff(self, sender, receiver):
        """
        Return the average payoff for strategies <sender> and <receiver> in a
        list
        """
        return sum([1/len(self.strats.states) * self.payoff(
            sender, receiver, state) for state in self.strats.states])


class Game:
    """
    Compute a game from an avgpayoff function, a strats object,
    and a mutation rate
    """
    def __init__(self, avgpayoff, strats, mutationrate=0):
        self.strats = strats
        payoffs = np.array([[
            avgpayoff(i, j) for j in range(self.strats.lrs)] for i in
            range(self.strats.lss)])
        self.senderpayoffs = payoffs[:, :, 0]
        self.receiverpayoffs = payoffs[:, :, 1].T
        self.mm_sender = self.mutationmatrix(mutationrate, self.strats.lss)
        self.mm_receiver = self.mutationmatrix(mutationrate, self.strats.lrs)

    def sender_avg_payoff(self, sender, receiver):
        return sender.dot(self.senderpayoffs.dot(receiver))

    def receiver_avg_payoff(self, receiver, sender):
        return receiver.dot(self.receiverpayoffs.dot(sender))

    def mutationmatrix(self, mutation, dimension):
        """
        Calculate a (square) mutation matrix for the with mutation rate
        given by <mutation> and dimension given by <dimension>
        """
        return np.array([[1 - mutation if i == j else mutation/(dimension - 1)
                          for i in np.arange(dimension)] for j in
                         np.arange(dimension)])

    def dX_dt(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint
        """
        # X's first part is the sender vector
        # its second part the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops = X[:self.strats.lss]
        receiverpops = X[self.strats.lss:]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs *
                     senderpops[..., None]).dot(
                         receiverpops).dot(
                             self.mm_sender) - senderpops*avgfitnesssender
        receiverdot = (self.receiverpayoffs *
                       receiverpops[..., None]).dot(
                           senderpops).dot(
                               self.mm_receiver) - receiverpops * avgfitnessreceiver
        return np.concatenate((senderdot, receiverdot))

    def dX_dt_ode(self, t, X):
        """
        Calculate the rhs of the system of odes for scipy.integrate.ode
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops = X[:self.strats.lss]
        receiverpops = X[self.strats.lss:]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs *
                     senderpops[...,None]).dot(receiverpops).dot(self.mm_sender) - senderpops*avgfitnesssender
        receiverdot = (self.receiverpayoffs *
                       receiverpops[...,None]).dot(senderpops).dot(self.mm_receiver) - receiverpops*avgfitnessreceiver
        return np.concatenate((senderdot, receiverdot))

    def jacobian(self, X, t=0):
        """
        Calculate the Jacobian of the system for scipy.integrate.odeint
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops = X[:self.strats.lss]
        receiverpops = X[self.strats.lss:]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops) # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[...,None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[...,None]
        tile1 = (self.mm_sender - senderpops[...,None]) * yS - np.identity(self.strats.lss) * avgfitnesssender
        tile2 = (self.mm_sender - senderpops).transpose().dot(xS)
        tile3 = (self.mm_receiver - receiverpops).transpose().dot(yR)
        tile4 = (self.mm_receiver - receiverpops[...,None]) * xR - np.identity(self.strats.lrs) * avgfitnessreceiver
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        return jac

    def jacobian_ode(self, t, X):
        """ 
        Calculate the Jacobian of the system for scipy.integrate.ode
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
#         senderpops = Xsplit[0]
 #        receiverpops = Xsplit[1]
        senderpops = X[:self.strats.lss]
        receiverpops = X[self.strats.lss:]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops) # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[...,None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[...,None]
        tile1 = (self.mm_sender - senderpops[...,None]) * yS - np.identity(self.strats.lss) * avgfitnesssender
        tile2 = (self.mm_sender - senderpops).transpose().dot(xS)
        tile3 = (self.mm_receiver - receiverpops).transpose().dot(yR)
        tile4 = (self.mm_receiver - receiverpops[...,None]) * xR - np.identity(self.strats.lrs) * avgfitnessreceiver
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        return jac.transpose()

    def delta_X(self, X):
        """
        Calculate a population vector for t' given the vector for t, using the
        discrete time replicator dynamics (Huttegger 2007)
        """
        # X's first part is the sender vector
        # its second part the receiver vector
        senderpops = X[:self.strats.lss]
        receiverpops = X[self.strats.lss:]
        avgfitnesssender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitnessreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdelta = (self.senderpayoffs *
                     senderpops[...,None]).dot(receiverpops).dot(self.mm_sender) / avgfitnesssender
        receiverdelta = (self.receiverpayoffs *
                       receiverpops[...,None]).dot(senderpops).dot(self.mm_receiver) / avgfitnessreceiver
        return np.concatenate((senderdelta, receiverdelta))

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
                           max(self.senders_vs_receiver(receiver))) < 1e-2
        receiverisbest = abs(payoffreceiver -
                             max(self.receivers_vs_sender(sender))) < 1e-2
        return senderisbest and receiverisbest

class Information:
    """
    Calculate mutual information between populations
    """
    def __init__(self, strats, popvector):
        self.sender = popvector[:strats.lss]
        self.receiver = popvector[strats.lss:]
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
    return matrix/sum(matrix.transpose())[..., None]

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


def int_or_list(intorlist):
    whatisit = type(intorlist)
    if whatisit == int:
        return np.arange(intorlist)
    if whatisit == list:
        return intorlist

