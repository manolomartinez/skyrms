"""
Set up an evolutionary game, that can be then fed to the evolve module.
There are two main classes here:
    - Games with a chance player
    - Games without a chance player
    """
import numpy as np
import itertools as it
import sys
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
        a mxo numpy array with the sender payoffs, and a mxo numpy array with
        receiver payoffs
        """
        if any(state_chances.shape[0] != row for row in
               [sender_payoff_matrix.shape[0],
                receiver_payoff_matrix.shape[0]]):
            sys.exit("The number of rows in sender and receiver payoffs should"
                     "be the same as the number of states")
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            sys.exit("Sender and receiver payoff arrays should have the same"
                     "shape")
        if type(messages) != int:
            sys.exit("The number of messages should be an integer")
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


class Evolve:
    """
    Calculate the equations necessary to evolve a population of senders and one
    of receivers. It takes as input a tuple: the first (second) member of the
    tuple is a nxm array such that the <i,j> cell gives the expected payoff for
    othe sender (receiver) of an encounter in which the sender follows strategy
    i and the receiver follows strategy j.
    As an additional input, it takes a tuple of a mutation matrix for the
    sender, and another for the receiver. If no mutation matrix is given, no
    mutation will be assumed.
    """
    def __init__(self, avgpayoff):
        self.senderpayoffs = avgpayoff[0]
        self.receiverpayoffs = avgpayoff[1].T
        self.lss, self.lrs = self.senderpayoffs.shape
        # By default, mutation matrices are the identity matrices. You can
        # change that.
        self.mm_sender = np.identity(self.senderpayoffs.shape[0])
        self.mm_receiver = np.identity(self.receiverpayoffs.shape[0])

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

    def sender_avg_payoff(self, sender, receiver):
        return sender.dot(self.senderpayoffs.dot(receiver))

    def receiver_avg_payoff(self, receiver, sender):
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
        senderpops = X[:self.lss]
        receiverpops = X[self.lss:]
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs *
                     senderpops[..., None]).dot(
                         receiverpops).dot(
                             self.mm_sender) - senderpops * avgfitsender
        receiverdot = ((self.receiverpayoffs *
                       receiverpops[..., None]).dot(
                           senderpops).dot(
                               self.mm_receiver) -
                       receiverpops * avgfitreceiver)
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
        senderpops = X[:self.lss]
        receiverpops = X[self.lss:]
        avgfitsender = self.sender_avg_payoff(senderpops, receiverpops)
        avgfitreceiver = self.receiver_avg_payoff(receiverpops, senderpops)
        senderdot = (self.senderpayoffs *
                     senderpops[..., None]).dot(receiverpops).dot(
                         self.mm_sender) - senderpops * avgfitsender
        receiverdot = ((self.receiverpayoffs *
                       receiverpops[..., None]).dot(
                           senderpops).dot(
                               self.mm_receiver) -
                       receiverpops * avgfitreceiver)
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
        senderpops = X[:self.lss]
        receiverpops = X[self.lss:]
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
        senderpops = X[:self.lss]
        receiverpops = X[self.lss:]
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
        senderpops = X[:self.lss]
        receiverpops = X[self.lss:]
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
        Calculate one run of the game following the , with starting
        points sinit and rinit, in times <times>, using scipy.integrate.odeint
        """
        from scipy.integrate import odeint
        return odeint(
            self.replicator_dX_dt_odeint, np.concatenate((sinit, rinit)),
            times, Dfun=self.replicator_jacobian_odeint, col_deriv=True,
            **kwargs)

    def replicator_ode(self, sinit, rinit):
        """
        Calculate one run of <game> with starting points sinit and rinit
        using scipy.integrate.ode
        """
        from scipy.integrate import ode
        initialpop = np.concatenate((sinit, rinit))
        initialtime = 0
        finaltime = 1000
        timeinc = 1
        equations = ode(self.replicator_dX_dt_ode,
                        self.replicator_jacobian_ode).set_integrator('dopri5')
        equations.set_initial_value(initialpop, initialtime)
        while equations.successful() and equations.t < finaltime:
            newdata = equations.integrate(equations.t + timeinc)
            try:
                data = np.append(data, [newdata], axis=0)
            except NameError:
                data = [newdata]
        return data

    def replicator_discrete(self, sinit, rinit, initialtime, finaltime,
                            timeinc):
        """
        Calculate one run of <game> with starting population vector <popvector>
        using the discrete time replicator dynamics
        """
        time = initialtime
        popvector = np.concatenate((sinit, rinit))
        data = [popvector]
        while time < finaltime:
            newpopvector = self.discrete_replicator_delta_X(popvector)
            popvector = newpopvector
            time += timeinc
            data = np.append(data, [popvector], axis=0)
        return data


def mutationmatrix(mutation, dimension):
    """
    Calculate a (square) mutation matrix with mutation rate
    given by <mutation> and dimension given by <dimension>
    """
    return np.array([[1 - mutation if i == j else mutation/(dimension - 1)
                      for i in np.arange(dimension)] for j in
                     np.arange(dimension)])


def int_or_list(intorlist):
    whatisit = type(intorlist)
    if whatisit == int:
        return np.arange(intorlist)
    if whatisit == np.ndarray:
        return intorlist
