"""
Set up an evolutionary game, that can be then fed to the evolve module.
There are two main classes here:
    - Games with a chance player
    - Games without a chance player
    """
import numpy as np
import itertools as it
import sys
from scipy.integrate import ode
from scipy.integrate import odeint
np.set_printoptions(precision=4)


class Chance:
    """
    Construct a payoff function for a game with a chance player, that chooses a
    state, among m possible ones; a sender that chooses a message, among n
    possible ones; a receiver that chooses an act among o possible ones; and
    the number of messages
    """
    def __init__(self, state_chances, sender_payoff_matrix,
                 receiver_payoff_matrix, messages):
        """
        Take a mx1 numpy array with the unconditional probabilities of states,
        a mxo numpy array with the sender payoffs, a mxo numpy array with
        receiver payoffs, and the number of available messages
        """
        if any(state_chances.shape[0] != row for row in
               [sender_payoff_matrix.shape[0],
                receiver_payoff_matrix.shape[0]]):
            sys.exit("The number of rows in sender and receiver payoffs should"
                     "be the same as the number of states")
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            sys.exit("Sender and receiver payoff arrays should have the same"
                     "shape")
        if not isinstance(messages, int):
            sys.exit("The number of messages should be an integer")
        self.chance_node = True  # flag to know where the game comes from
        self.state_chances = state_chances
        self.sender_payoff_matrix = sender_payoff_matrix
        self.receiver_payoff_matrix = receiver_payoff_matrix
        self.states = state_chances.shape[0]
        self.messages = messages
        self.acts = sender_payoff_matrix.shape[1]

    def sender_pure_strats(self):
        """
        Return the set of pure strategies available to the sender
        """
        pure_strats = np.identity(self.messages)
        return np.array(list(it.product(pure_strats, repeat=self.states)))

    def receiver_pure_strats(self):
        """
        Return the set of pure strategies available to the receiver
        """
        pure_strats = np.identity(self.acts)
        return np.array(list(it.product(pure_strats, repeat=self.messages)))

    def payoff(self, sender_strat, receiver_strat):
        """
        Calculate the average payoff for sender and receiver given concrete
        sender and receiver strats
        """
        state_act = sender_strat.dot(receiver_strat)
        sender_payoff = self.state_chances.dot(np.sum(state_act *
                                               self.sender_payoff_matrix,
                                               axis=1))
        receiver_payoff = self.state_chances.dot(np.sum(state_act *
                                                 self.receiver_payoff_matrix,
                                                 axis=1))
        return (sender_payoff, receiver_payoff)

    def avg_payoffs(self, sender_strats, receiver_strats):
        """
        Return an array with the average payoff of sender strat i against
        receiver strat j in position <i, j>
        """
        payoff_ij = np.vectorize(lambda i, j: self.payoff(sender_strats[i],
                                                          receiver_strats[j]))
        shape_result = (sender_strats.shape[0], receiver_strats.shape[0])
        return np.fromfunction(payoff_ij, shape_result)

    def calculate_sender_mixed_strat(self, sendertypes, senderpop):
        mixedstratsender = sendertypes * senderpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratsender)

    def calcuate_receiver_mixed_strat(self, receivertypes, receiverpop):
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis,
                                                         np.newaxis]
        return sum(mixedstratreceiver)


class NonChance:
    """
    Construct a payoff function for a game without chance player: a sender that
    chooses a message, among n possible ones; a receiver that chooses an act
    among o possible ones; and the number of messages
    """
    def __init__(self, sender_payoff_matrix, receiver_payoff_matrix, messages):
        """
        Take a mxo numpy array with the sender payoffs, a mxo numpy array
        with receiver payoffs, and the number of available messages
        """
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            sys.exit("Sender and receiver payoff arrays should have the same"
                     "shape")
        if not isinstance(messages, int):
            sys.exit("The number of messages should be an integer")
        self.chance_node = False  # flag to know where the game comes from
        self.sender_payoff_matrix = sender_payoff_matrix
        self.receiver_payoff_matrix = receiver_payoff_matrix
        self.states = sender_payoff_matrix.shape[0]
        self.messages = messages
        self.acts = sender_payoff_matrix.shape[1]

    def sender_pure_strats(self):
        """
        Return the set of pure strategies available to the sender. For this
        sort of games, a strategy is a tuple of vector with probability 1 for
        the sender's state, and an mxn matrix in which the only nonzero row
        is the one that correspond's to the sender's type.
        """
        def build_strat(state, row):
            zeros = np.zeros((self.states, self.messages))
            zeros[state] = row
            return zeros
        states = range(self.states)
        over_messages = np.identity(self.messages)
        return np.array([build_strat(state, row) for state, row in
                         it.product(states, over_messages)])

    def receiver_pure_strats(self):
        """
        Return the set of pure strategies available to the receiver
        """
        pure_strats = np.identity(self.acts)
        return np.array(list(it.product(pure_strats, repeat=self.messages)))

    def payoff(self, sender_strat, receiver_strat):
        """
        Calculate the average payoff for sender and receiver given concrete
        sender and receiver strats
        """
        state_act = sender_strat.dot(receiver_strat)
        sender_payoff = np.sum(state_act * self.sender_payoff_matrix)
        receiver_payoff = np.sum(state_act * self.receiver_payoff_matrix)
        return (sender_payoff, receiver_payoff)

    def avg_payoffs(self, sender_strats, receiver_strats):
        """
        Return an array with the average payoff of sender strat i against
        receiver strat j in position <i, j>
        """
        payoff_ij = np.vectorize(lambda i, j: self.payoff(sender_strats[i],
                                                          receiver_strats[j]))
        shape_result = (len(sender_strats), len(receiver_strats))
        return np.fromfunction(payoff_ij, shape_result)

    def calculate_sender_mixed_strat(self, sendertypes, senderpop):
        mixedstratsender = sendertypes * senderpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratsender)

    def calculate_receiver_mixed_strat(self, receivertypes, receiverpop):
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis,
                                                         np.newaxis]
        return sum(mixedstratreceiver)


class Evolve:
    """
    Calculate the equations necessary to evolve a population of senders and one
    of receivers. It takes as input a <game>, (which as of now only can be a
    Chance object>, and a  tuple: the first (second) member of the tuple is a
    nxm array such that the <i,j> cell gives the expected payoff for the sender
    (receiver) of an encounter in which the sender follows strategy i and the
    receiver follows strategy j.
    """
    def __init__(self, game, sendertypes, receivertypes):
        self.game = game
        avgpayoffs = self.game.avg_payoffs(sendertypes, receivertypes)
        self.senderpayoffs = avgpayoffs[0]
        self.receiverpayoffs = avgpayoffs[1].T
        self.sendertypes = sendertypes
        self.receivertypes = receivertypes
        self.lss, self.lrs = self.senderpayoffs.shape
        # By default, mutation matrices are the identity matrices. You can
        # change that.
        self.mm_sender = np.identity(self.senderpayoffs.shape[0])
        self.mm_receiver = np.identity(self.receiverpayoffs.shape[0])

    def random_sender(self):
        """
        Return frequencies of a random sender population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lss)])

    def random_receiver(self):
        """
        Return frequencies of a random receiver population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lrs)])

    def sender_avg_payoff(self, sender, receiver):
        """
        Return the average payoff that senders get when the population vectors
        are <sender> and <receiver>
        """
        return sender.dot(self.senderpayoffs.dot(receiver))

    def receiver_avg_payoff(self, receiver, sender):
        """
        Return the average payoff that receivers get when the population
        vectors are <sender> and <receiver>
        """
        return receiver.dot(self.receiverpayoffs.dot(sender))

    def replicator_dX_dt_odeint(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint
        """
        # X's first part is the sender vector
        # its second part the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs *
                     senderpops[..., None]).dot(
                         receiverpops).dot(
                             self.mm_sender) - senderpops * avgfitsender
        receiverdot = ((self.receiverpayoffs * receiverpops[..., None]).dot(
            senderpops).dot(self.mm_receiver) - receiverpops * avgfitreceiver)
        return np.concatenate((senderdot, receiverdot))

    def replicator_dX_dt_ode(self, t, X):
        """
        Calculate the rhs of the system of odes for scipy.integrate.ode
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs *
                     senderpops[..., None]).dot(receiverpops).dot(
                         self.mm_sender) - senderpops * avgfitsender
        receiverdot = ((self.receiverpayoffs * receiverpops[..., None]).dot(
            senderpops).dot(self.mm_receiver) - receiverpops * avgfitreceiver)
        return np.concatenate((senderdot, receiverdot))

    def replicator_jacobian_odeint(self, X, t=0):
        """
        Calculate the Jacobian of the system for scipy.integrate.odeint
        """
        # X's first half is the sender vector
        # its second half the receiver vector
        # Xsplit = np.array_split(X, 2)
        # senderpops = Xsplit[0]
        # receiverpops = Xsplit[1]
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops)  # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[..., None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[..., None]
        tile1 = (self.mm_sender - senderpops[..., None]) * yS - np.identity(
            self.lss) * avgfitsender
        tile2 = (self.mm_sender - senderpops).transpose().dot(xS)
        tile3 = (self.mm_receiver - receiverpops).transpose().dot(yR)
        tile4 = ((self.mm_receiver - receiverpops[..., None]) *
                 xR - np.identity(self.lrs) * avgfitreceiver)
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        return jac

    def replicator_jacobian_ode(self, t, X):
        """
        Calculate the Jacobian of the system for scipy.integrate.ode
        """
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        yS = self.senderpayoffs.dot(receiverpops)  # [y1P11 +...+ ynP1n, ...,
        # y1Pn1 + ... ynPnn] This one works
        xS = self.senderpayoffs * senderpops[..., None]
        xR = self.receiverpayoffs.dot(senderpops)
        yR = self.receiverpayoffs * receiverpops[..., None]
        tile1 = (self.mm_sender - senderpops[..., None]) * yS - np.identity(
            self.lss) * avgfitsender
        tile2 = (self.mm_sender - senderpops).transpose().dot(xS)
        tile3 = (self.mm_receiver - receiverpops).transpose().dot(yR)
        tile4 = ((self.mm_receiver - receiverpops[..., None]) *
                 xR - np.identity(self.lrs) * avgfitreceiver)
        lefthalf = np.vstack((tile1.transpose(), tile2.transpose()))
        righthalf = np.vstack((tile3.transpose(), tile4.transpose()))
        jac = np.hstack((lefthalf, righthalf))
        return jac.transpose()

    def discrete_replicator_delta_X(self, X):
        """
        Calculate a population vector for t' given the vector for t, using the
        discrete time replicator dynamics (Huttegger 2007)
        """
        # X's first part is the sender vector
        # its second part the receiver vector
        senderpops, receiverpops = self.vector_to_populations(X)
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdelta = (self.senderpayoffs *
                       senderpops[..., None]).dot(
                           receiverpops).dot(self.mm_sender) / avgfitsender
        receiverdelta = (self.receiverpayoffs *
                         receiverpops[..., None]).dot(
                             senderpops).dot(
                                 self.mm_receiver) / avgfitreceiver
        return np.concatenate((senderdelta, receiverdelta))

    def replicator_odeint(self, sinit, rinit, times, **kwargs):
        """
        Calculate one run of the game following the replicator(-mutator)
        dynamics, with starting points sinit and rinit, in times <times> (a
        game.Times instance), using scipy.integrate.odeint
        """
        return odeint(
            self.replicator_dX_dt_odeint, np.concatenate((sinit, rinit)),
            times.time_vector, Dfun=self.replicator_jacobian_odeint,
            col_deriv=True, **kwargs)

    def replicator_ode(self, sinit, rinit, times):
        """
        Calculate one run of the game, following the replicator(-mutator)
        dynamics in continuous time, in <times> (a game.Times instance) with
        starting points sinit and rinit using scipy.integrate.ode
        """
        initialpop = np.concatenate((sinit, rinit))
        equations = ode(self.replicator_dX_dt_ode,
                        self.replicator_jacobian_ode).set_integrator('dopri5')
        equations.set_initial_value(initialpop, times.initial_time)
        while equations.successful() and equations.t < times.final_time:
            newdata = equations.integrate(equations.t + times.time_inc)
            try:
                data = np.append(data, [newdata], axis=0)
            except NameError:
                data = [newdata]
        return data

    def replicator_discrete(self, sinit, rinit, times):
        """
        Calculate one run of the game, following the discrete
        replicator(-mutator) dynamics, in <times> (a game.Times object) with
        starting population vector <popvector> using the discrete time
        replicator dynamics. Note that this solver will just calculate n points
        in the evolution of the population, and will not try to match them to
        the times as provided.
        """
        popvector = np.concatenate((sinit, rinit))
        data = np.array([popvector])
        for _ in range(len(times.time_vector)):
            np.append(data, [self.discrete_replicator_delta_X(data[-1])])
        return data

    def vector_to_populations(self, vector):
        """
        Take one of the population vectors returned by the solvers, and output
        two vectors, for the sender and receiver populations respectively.
        """
        return np.hsplit(vector, [self.lss])

    def sender_to_mixed_strat(self, senderpop):
        """
        Take a sender population vector and output the average
        sender strat implemented by the whole population
        """
        return self.game.calculate_sender_mixed_strat(self.sendertypes,
                                                      senderpop)

    def receiver_to_mixed_strat(self, receiverpop):
        """
        Take a receiver population vector and output the average
        receiver strat implemented by the whole population
        """
        return self.game.calculate_receiver_mixed_strat(self.receivertypes,
                                                        receiverpop)


class Times:
    """
    Provides a way of having a single time input to both odeint and ode
    """
    def __init__(self, initial_time, final_time, time_inc):
        """
        Takes the initial time for simulations <initial_time>, the final time
        <final_time> and the time increment <time_inc>, and creates an object
        with these values as attributes, and also a vector that can be fed into
        odeint.
        """
        self.initial_time = initial_time
        self.final_time = final_time
        self.time_inc = time_inc
        points = (final_time - initial_time) / time_inc
        self.time_vector = np.linspace(initial_time, final_time, points)


def mutationmatrix(mutation, dimension):
    """
    Calculate a (square) mutation matrix with mutation rate
    given by <mutation> and dimension given by <dimension>
    """
    return np.array([[1 - mutation if i == j else mutation/(dimension - 1)
                      for i in np.arange(dimension)] for j in
                     np.arange(dimension)])
