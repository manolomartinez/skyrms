"""
Information-theoretic analyses
"""
from itertools import repeat
import multiprocessing

import numpy as np

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
        if self.game.chance_node:
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


class RDT:
    """
    Calculate the rate-distortion function for a game and one or two disortion
    measures
    """
    def __init__(self, game, distortion_tensor, epsilon=0.001):
        """
        Parameters
        ----------
        game: a skyrms.asymmetric_games.Chance object

        distortion_tensor: a collection of distortion matrices (same dimensions as payoff
        matrices), stacked along axis 2

        epsilon: the precision up to which the point should be calculated
        """
        self.pmf = game.state_chances
        self.outcomes = len(self.pmf)
        self.epsilon = epsilon
        self.dist_tensor = distortion_tensor


    def all_points(self, iterator=None, outputfile=None):
        """
        Calculate the R(D) function for as many points as given by <iterator>
        (all of them by default)
        Save them to <file>. Each line of the file (row of the array) is [K,
        distortion, rate]
        """
        if not iterator:
            iterator = range(self.K)
        return np.array([self.blahut(k, outputfile) for k in
                         iterator]).T

    def all_points_multiprocessing(self, iterator=None, outputfile=None):
        """
        Calculate the R(D) function for as many points as given by <iterator>
        (all of them by default)
        Save them to <file>. Each line of the file (row of the array) is [K,
        distortion, rate]
        """
        pool = multiprocessing.Pool(None)
        if not iterator:
            iterator = range(self.K)
        # nash = s.Nash(game)
        newsols = pool.imap_unordered(self.blahut_mp, zip(iterator,
                                                          repeat(outputfile)))
        data = np.array([sol for sol in newsols])
        pool.close()
        pool.join()
        return data.T

    def blahut_mp(self, args):
        """
        Unpack args for the official blahut function
        """
        return self.blahut(*args)

    def blahut(self, lambda_, return_cond=False):
        """
        Calculate the point in the R(D)-D curve with slope given by
        self.calc_s(<k>). Follows Cover & Thomas 2006, p. 334
        """
        # we start with the uniform output distribution
        output = np.ones(self.outcomes) / self.outcomes
        cond = self.update_conditional(lambda_, output)
        distortion = self.calc_distortion(cond)
        rate = self.calc_rate(cond, output)
        delta_dist = 2 * self.epsilon
        while delta_dist > self.epsilon:
            output = self.pmf @ cond
            cond = self.update_conditional(lambda_, output)
            new_distortion = self.calc_distortion(cond)
            rate = self.calc_rate(cond, output)
            delta_dist = np.abs(new_distortion - distortion)
            distortion = new_distortion
        if return_cond:
            tuple = (rate, distortion, cond)
        else:
            tuple = (rate, distortion)
        return tuple

    def blahut_two(self, lambda_, max_rounds=100, return_cond=False):
        """
        Calculate the point in the R(D, D') surface with slopes given by
        lambda_ and mu_. Follows Cover & Thomas 2006, p. 334
        """
        # we start with the uniform output distribution
        params = len(lambda_)
        output = np.ones(self.outcomes) / self.outcomes
        cond = self.update_conditional(lambda_, output)
        rate = self.calc_rate(cond, output)
        delta_dist = 2 * self.epsilon
        rounds = 0
        while delta_dist > self.epsilon and rounds <= max_rounds:
            output = self.pmf @ cond
            cond = self.update_conditional(lambda_, output)
            new_rate = self.calc_rate(cond, output)
            delta_dist = np.abs(new_rate - rate)
            rate = new_rate
            rounds = rounds + 1
            if rounds == max_rounds:
                print("Max rounds for {}".format(lambda_))
        distortion = [self.calc_distortion(cond, matrix) for matrix in
                      range(params)]
        if return_cond:
            return_tuple = (rate, *distortion, cond)
        else:
            return_tuple = (rate, *distortion)
        return return_tuple

    def blahut_berger(self, s_, max_rounds=100):
        """
        Calculate the point in the R(D)-D curve with slope given by
        s. Follows Berger (2003), p. 2074)
        """
        # qr = np.ones(self.outcomes) / self.outcomes
        qr = self.pmf
        lambda_ = 1 / ((qr * np.exp(s_ * self.dist_matrix)).sum(1))
        cr = ((self.pmf * lambda_)[..., None] * np.exp(s_ *
                                                        self.dist_matrix)).sum(0)
        rounds = 0
        while max(cr) > 1 + self.epsilon and rounds <= max_rounds:
            qr = cr * qr
            lambda_ = 1 / ((qr * np.exp(s_ * self.dist_matrix)).sum(1))
            cr = ((self.pmf * lambda_)[..., None] * np.exp(s_ *
                                                           self.dist_matrix)).sum(0)
            rounds = rounds + 1
            if rounds == max_rounds:
                print("Max rounds")
        distortion = np.sum((self.pmf * lambda_)[..., None] * (qr * np.exp(s_ *
                                                                           self.dist_matrix)
                                                               * self.dist_matrix))
        rate = s_ * distortion + np.sum(self.pmf * np.log2(lambda_))
        return rate, distortion

    def update_conditional(self, lambda_, output):
        """
        Calculate a new conditional distribution from the <output> distribution
        and the <lambda_> parameters.  The conditional probability matrix is such that
        cond[i, j] corresponds to P(x^_j | x_i)
        """
        params = len(lambda_)
        axes =tuple([Ellipsis] + [np.newaxis] * params)
        lagrange = (-1 * lambda_[axes] * self.dist_tensor).sum(0)
        cond = output * np.exp(lagrange)
        return normalize_axis(cond, 1)

    def calc_distortion(self, cond, matrix):
        """
        Calculate the distortion for a given channel (individuated by the
        conditional matrix in <cond>), for a certain slice of self.dist_tensor
        """
        # return np.sum(self.pmf @ (cond * self.dist_matrix))
        return np.matmul(self.pmf, (cond * self.dist_tensor[matrix])).sum()

    def calc_rate(self, cond, output):
        """
        Calculate the rate for a channel (given by <cond>) and output
        distribution (given by <output>)
        """
        return np.sum(self.pmf @ (cond * np.ma.log2(cond / output).filled(0)))


class Shea:
    """
    Calculate functional content vectors as presented in Shea, Godfrey-Smith
    andand Cao 2017.
    """
    def __init__(self, game):
        """
        Parameters
        ----------
        game: a skyrms.asymmetric_games.Chance object
        """
        self.game = game
        self.baseline_sender, self.baseline_receiver = self.baseline_payoffs()

    def baseline_payoffs(self):
        """
        Give a vector with the payoffs for sender, and another for receiver,
        when the receiver does the best possible act for it in the absence of any communication.
        I will choose, for now, the receiver act that gives the best possible sender payoff
        (this is not decided by Shea et al.; see fn.14)
        """
        vec_expected = np.vectorize(self.expected_for_act)
        payoffs = np.apply_along_axis(vec_expected, 0, np.arange(self.game.acts)).T
        maxreceiver = np.max(payoffs[:, 1])
        sender = payoffs[:, 0][payoffs[:, 1] == maxreceiver]
        return np.max(sender), maxreceiver

    def expected_for_act(self, act):
        """
        Calculate the expected payoff for sender and receiver of doing one act,
        in the absence of communication
        """
        sender_payoffs_per_state = self.game.sender_payoff_matrix[:, act]
        receiver_payoffs_per_state = self.game.receiver_payoff_matrix[:, act]
        expected_sender = self.game.state_chances.dot(sender_payoffs_per_state)
        expected_receiver = self.game.state_chances.dot(
            receiver_payoffs_per_state)
        return expected_sender, expected_receiver

    def normal_payoffs(self):
        """
        Calculate payoffs minus the baseline
        """
        normal_sender = self.game.sender_payoff_matrix - self.baseline_sender
        normal_receiver = self.game.receiver_payoff_matrix - self.baseline_receiver
        return normal_sender, normal_receiver

    def calc_dmin(self):
        """
        Calculate dmin as defined in Shea et al. (2917, p. 24)
        """
        normal_sender, normal_receiver = self.normal_payoffs()
        return np.minimum(normal_sender, normal_receiver)

    def calc_summation(self, norm_payoff, receiver_strat):
        """
        Calculate the summation in the entries of the functional content vector
        """
        inside_summation_raw = np.multiply.outer(norm_payoff, receiver_strat)
        inside_summation = np.einsum('ijkj->ijk', inside_summation_raw)
        return np.sum(inside_summation, axis=1)

    def calc_entries(self, sender_strat, receiver_strat, payoff_matrix):
        """
        Calculate the entries of the functional vector, given one choice for
        the (baselined) payoff matrix
        """
        bayes_sender = bayes_theorem(self.game.state_chances, sender_strat)
        summation = self.calc_summation(payoff_matrix, receiver_strat)
        return bayes_sender * summation

    def calc_entries_dmin(self, sender_strat, receiver_strat):
        """
        Calculate the entries of the functional vector, given one choice for
        the official dmin
        """
        return self.calc_entries(sender_strat, receiver_strat,
                                 self.calc_dmin())

    def calc_entries_sender(self, sender_strat, receiver_strat):
        """
        Calculate the entries of the functional vector, for the baselined
        sender
        """
        normal_sender = self.game.sender_payoff_matrix - self.baseline_sender
        return self.calc_entries(sender_strat, receiver_strat, normal_sender)

    def calc_entries_receiver(self, sender_strat, receiver_strat):
        """
        Calculate the entries of the functional vector, for the baselined
        receiver
        """
        normal_receiver = self.game.receiver_payoff_matrix - self.baseline_receiver
        return self.calc_entries(sender_strat, receiver_strat, normal_receiver)

    def calc_condition(self, receiver_strat, payoff_matrix, baseline):
        """
        Calculate the condition for nonzero vector entries
        """
        outer = np.multiply.outer(receiver_strat, payoff_matrix)
        precondition = np.einsum('kjij->ijk', outer)
        return np.sum(precondition, axis=1) > baseline

    def calc_condition_sender(self, receiver_strat):
        """
        Calculate condition() for the sender payoff matrix and baseline
        """
        return self.calc_condition(receiver_strat,
                                   self.game.sender_payoff_matrix,
                                   self.baseline_sender)

    def calc_condition_receiver(self, receiver_strat):
        """
        Calculate condition() for the receiver payoff matrix and baseline
        """
        return self.calc_condition(receiver_strat,
                                   self.game.receiver_payoff_matrix,
                                   self.baseline_receiver)

    def calc_condition_common(self, receiver_strat):
        """
        Calculate the condition for a nonzero functional vector entry in
        the definition in (op. cit., p. 24)
        """
        return self.calc_condition_sender(
            receiver_strat) & self.calc_condition_receiver(receiver_strat)

    def functional_content(self, entries, condition):
        """
        Put everything together in a functional vector per message
        """
        return normalize_axis(entries * condition, 0)

    def functional_content_sender(self, sender_strat, receiver_strat):
        """
        Calculate the functional content from the perspective of the sender
        """
        return self.functional_content(self.calc_entries_sender(sender_strat,
                                                                receiver_strat),
                                       self.calc_condition_sender(receiver_strat))

    def functional_content_receiver(self, sender_strat, receiver_strat):
        """
        Calculate the functional content from the perspective of the receiver
        """
        return self.functional_content(self.calc_entries_receiver(sender_strat,
                                                                  receiver_strat),
                                       self.calc_condition_receiver(receiver_strat))

    def functional_content_dmin(self, sender_strat, receiver_strat):
        """
        Calculate the functional content from the perspective of dmin
        """
        return self.functional_content(self.calc_entries_dmin(sender_strat,
                                                              receiver_strat),
                                       self.calc_condition_common(receiver_strat))


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


def normalize_axis(array, axis):
    """
    Normalize a matrix along <axis>, being sensible with all-zero rows
    """
    return np.apply_along_axis(normalize_vector, axis, array)


def from_joint_to_conditional(array):
    """
    Normalize row-wise

    """
    return normalize_axis(array, 1)


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


def bayes_theorem(unconds, conds):
    """
    Perform Bayes' theorem on a matrix of conditional probabilities

    Parameters
    ----------
    unconds:
        a (n x 1) numpy array of unconditional probabilities [P(A1), ... , P(An)]
    conds:
        a (m x n) numpy array of conditional probabilities
        [[P(B1|A1), ... , P(Bm|A1)], ... , [P(B1|An), ..., P(Bm|An)]]

    Returns
    -------
    A (n x m) numpy array of conditional probabilities
        [[P(A1|B1), ... , P(An|B1)], ... , [P(PA1|Bm), ... , P(An|Bm)]]
    """
    joint = from_conditional_to_joint(unconds, conds)
    return from_joint_to_conditional(joint.T).T


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
