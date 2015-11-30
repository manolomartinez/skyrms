"""
Analyze results of the game evolutions
"""
import numpy as np
import itertools as it
from scipy.stats import kendalltau

from skyrms import exceptions


class Information:
    """
    Calculate information-theoretic quantities between strats. It expects a
    game, as created by game.Chance or game.NonChance, a sender strategy, and a
    receiver strategy
    """
    def __init__(self, game, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        self.game = game
        if self.game.game.chance_node:
            self.state_chances = game.state_chances
            self.msg_cond_on_states, self.acts_cond_on_msg = (self.sender,
                                                              self.receiver)
        else:
            # This is a game without a chance node. State chances are
            # calculated from the sender strategy, and msg_cond_on_states is
            # not given directly
            self.state_chances = np.sum(sender, axis=1)
            self.msg_cond_on_states, self.acts_cond_on_msg = (
                from_joint_to_conditional(sender), self.receiver)

    def joint_states_messages(self):
        return from_conditional_to_joint(self.state_chances,
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
        """
        Calculate the mutual info between states and acts
        """
        return mutual_info_from_joint(self.joint_states_acts())

    def mutual_info_states_messages(self):
        """
        Calculate the mutual info between states and messages
        """
        return mutual_info_from_joint(self.joint_states_messages())

    def mutual_info_messages_acts(self):
        """
        Calculate the mutual info between messages and acts
        """
        return mutual_info_from_joint(self.joint_messages_acts())


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


def from_joint_to_conditional(array):
    """
    Normalize a matrix row-wise, being sensible with all-zero rows

    """
    return np.apply_along_axis(normalize_vector, 1, array)


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


def normalize_vector(vector):
    """
    Normalize a vector, leaving all-zero vector as they are
    """
    if np.allclose(vector, np.zeros_like(vector)):
        return vector
    else:
        return vector / sum(vector)


class CommonInterest:
    """
    Calculate quantities useful for the study of the degree of common interest
    between senders and receivers
    """
    def __init__(self, game):
        self.sender = game.sender_payoff_matrix
        self.receiver = game.receiver_payoff_matrix
        try:
            self.state_chances = game.state_chances
            self.chance_node = True
        except AttributeError:
            self.chance_node = False

    def K(self, array):
        """
        Calculate K as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return intra_tau(self.state_chances, array)

    def sender_K(self):
        """
        Calculate K for the sender
        """
        return self.K(self.sender)

    def receiver_K(self):
        """
        Calculate K for the receiver
        """
        return self.K(self.receiver)

    def C_chance(self):
        """
        Calculate C as defined in Godfrey-Smith and Martinez (2013) -- but
        using scipy.stats.kendalltau
        """
        if not self.chance_node:
            raise exceptions.ChanceNodeError("This is not a chance-node game.")
        return tau_per_rows(self.state_chances, self.sender, self.receiver)

    def C_nonchance(self):
        """
        Calculate the C for non-chance games (using the total KTD)
        """
        return total_tau(self.sender, self.receiver)


def intra_tau(unconds, array):
    """
    Calculate the average (weighted by <unconds> of the pairwise Kendall's tau
    distance between rows (states) of <array>
    """
    taus = np.array([kendalltau(row1, row2)[0] for row1, row2 in
                     it.combinations(array, 2)])
    return unconds.dot(taus)


def total_tau(array1, array2):
    """
    Calculate the KTD between the flattened <array1> and <array2>. Useful for
    NonChance games
    """
    return kendalltau(array1, array2)[0]


def tau_per_rows(unconds, array1, array2):
    """
    Calculate the average (weighted by <unconds> of the Kendall's tau distance
    between the corresponding rows (states) of <array1> and <array2>
    """
    taus = np.array([kendalltau(row1, row2)[0] for row1, row2 in zip(array1,
                                                                     array2)])
    return unconds.dot(taus)



class Nash:
    """
    Calculate Nash equilibria
    """
    def __init__(self, game):
        self.game = game

    def receivers_vs_sender(self, sender):
        receivers = np.identity(self.game.lrs)
        return [self.game.receiver_avg_payoff(receiver, sender) for receiver in
                receivers]

    def senders_vs_receiver(self, receiver):
        senders = np.identity(self.game.lss)
        return [self.game.sender_avg_payoff(sender, receiver) for sender in
                senders]

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



class Stability:
    """
    Calculate some coarse stability measures of population vectors 
    """
    def __init__(self, evolution):
        """
        <evolution> is an (points, types) matrix, with a population vector in
        each row, representing an evolution from row 0 to row <points>
        """


def stable_vector(vector):
    """
    Return true if the vector does not move
    """
    return np.allclose(0, max(vector) - min(vector))


def periodic_vector(vector):
    """
    We take the FFT of a vector, and eliminate all components but the two main
    ones (i.e., the static and biggest sine amplitude) and compare the
    reconstructed wave with the original. Return true if close enough
    """
    rfft = np.fft.rfft(vector)
    magnitudes = np.abs(np.real(rfft))
    choice = magnitudes > sorted(magnitudes)[-3]
    newrfft = np.choose(choice, (np.zeros_like(rfft), rfft))
    newvector = np.fft.irfft(newrfft)
    return np.allclose(vector, newvector, atol=1e-2)
