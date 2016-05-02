"""
Calculate equations to evolve populations in a game. Right now, we can
calculate the replicator (-mutator) dynamics, with one or two populations, in
discrete or continuous time
"""

import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint


class OnePop:
    """
    Calculate the equations necessary to evolve one population. It takes as
    input a <game> and an array such that the <i,j> cell gives the expected
    payoff for the-strategist player of an encounter in which they  follow
    strategy i and their opponent follows strategy j.
    """
    def __init__(self, game, playertypes):
        self.game = game
        self.avgpayoffs = self.game.avg_payoffs(playertypes)
        self.playertypes = playertypes
        self.lps = self.playertypes.shape
        # By default, mutation matrices are the identity matrices. You can
        # change that.
        self.mm = np.identity(self.lps)
        # We can set a limit of precision in the calculation of diffeqs, to
        # avoid artifacts. By default, we do not
        self.precision = None

    def random_player(self):
        """
        Return frequencies of a random sender population
        """
        return np.random.dirichlet([1 for i in np.arange(self.lps)])

    def player_avg_payoff(self, player):
        """
        Return the average payoff that players get when the population vector
        is <player>
        """
        return player.dot(self.avgpayoffs.dot(player))

    def replicator_dX_dt_odeint(self, X, t):
        """
        Calculate the rhs of the system of odes for scipy.integrate.odeint
        """
        avgfitplayer = self.player_avg_payoff(X)
        result = self.avgpayoffs.dot(X) - X * avgfitplayer
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

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
        result = np.concatenate((senderdot, receiverdot))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

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
        if self.precision:
            np.around(jac, decimals=self.precision, out=jac)
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
        if self.precision:
            np.around(jac, decimals=self.precision, out=jac)
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
        result = np.concatenate((senderdelta, receiverdelta))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

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

    def replicator_ode(self, sinit, rinit, times, integrator='dopri5'):
        """
        Calculate one run of the game, following the replicator(-mutator)
        dynamics in continuous time, in <times> (a game.Times instance) with
        starting points sinit and rinit using scipy.integrate.ode
        """
        initialpop = np.concatenate((sinit, rinit))
        equations = ode(self.replicator_dX_dt_ode,
                        self.replicator_jacobian_ode).set_integrator(
                            integrator)
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
        data = np.empty([len(times.time_vector), len(popvector)])
        data[0, :] = popvector
        for i in range(1, len(times.time_vector)):
            data[i, :] = self.discrete_replicator_delta_X(data[i - 1, :])
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


class TwoPops:
    """
    Calculate the equations necessary to evolve a population of senders and one
    of receivers. It takes as input a <game>, (which as of now only can be a
    Chance object>, and a tuple: the first (second) member of the tuple is a
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
        # We can set a limit of precision in the calculation of diffeqs, to
        # avoid artifacts. By default, we do not
        self.precision = None

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
        result = np.concatenate((senderdot, receiverdot))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

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
        result = np.concatenate((senderdot, receiverdot))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

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
        if self.precision:
            np.around(jac, decimals=self.precision, out=jac)
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
        if self.precision:
            np.around(jac, decimals=self.precision, out=jac)
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
        result = np.concatenate((senderdelta, receiverdelta))
        if self.precision:
            np.around(result, decimals=self.precision, out=result)
        return result

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

    def replicator_ode(self, sinit, rinit, times, integrator='dopri5'):
        """
        Calculate one run of the game, following the replicator(-mutator)
        dynamics in continuous time, in <times> (a game.Times instance) with
        starting points sinit and rinit using scipy.integrate.ode
        """
        initialpop = np.concatenate((sinit, rinit))
        equations = ode(self.replicator_dX_dt_ode,
                        self.replicator_jacobian_ode).set_integrator(
                            integrator)
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
        data = np.empty([len(times.time_vector), len(popvector)])
        data[0, :] = popvector
        for i in range(1, len(times.time_vector)):
            data[i, :] = self.discrete_replicator_delta_X(data[i - 1, :])
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
