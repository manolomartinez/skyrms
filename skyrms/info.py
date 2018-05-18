"""
Information-theoretic analyses
"""

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
        vec_expected = np.vectorize(lambda x: self.expected_for_act(x))
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
        expected_sender = self.game.state_chances @ sender_payoffs_per_state
        expected_receiver = self.game.state_chances @ receiver_payoffs_per_state
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

    def calc_entries(self, sender_strat, receiver_strat):
        """
        Calculate the by_message() tensor between the P(S|M) matrix 
        (the bayes'ed sender_strat) and the P(A|M) matrix (which is receiver_strat)
        """
        bayes_sender = bayes_theorem(self.game.state_chances, sender_strat)
        normal_sender, normal_receiver = self.normal_payoffs(game)
        dmin = self.calc_dmin(game)
        summation_common = self.calc_summation(dmin, receiver_strat)
        summation_sender = self.calc_summation(normal_sender, receiver_strat)
        summation_receiver = self.calc_summation(normal_receiver, receiver_strat)
        return bayes_sender * summation_common, bayes_sender * summation_sender, bayes_sender * summation_receiver

    def precondition(self, receiver_strat, payoff_matrix):
        """
        Calculate $P(A_j|M_k)\cdot v(S_i,A_j)$, where v is some
        <payoff_matrix>.  
        Put those values in an array D such that $D[i,
        j, k] = P(A_j|M_k)\cdot v(S_i,A_j)$ 
        """
        outer = np.multiply.outer(receiver_strat, payoff_matrix)
        return np.einsum('kjij->ijk', outer)

    def precondition_sender(self, receiver_strat):
        """
        Calculate precondition() for the sender payoff matrix
        """
        return self.precondition(self.game.sender_payoff_matrix)

    def precondition_receiver(self, receiver_strat):
        """
        Calculate precondition() for the receiver payoff matrix
        """
        return self.precondition(self.game.receiver_payoff_matrix)

    def calc_conditions(self, receiver_strat):
        """
        Calculate the condition for a nonzero functional vector entry in
        the definition in (op. cit., p. 24)
        """
        precond_sender, precond_receiver = self.precondition(receiver_strat)
        sender_cond = np.sum(precond_sender, axis=1) > self.baseline_sender
        receiver_cond = np.sum(precond_receiver, axis=1) > self.baseline_receiver
        return sender_cond, receiver_cond

    def calc_condition(game, receiver_strat):
        sender_cond, receiver_cond = calc_conditions(game, receiver_strat)
        return sender_cond & receiver_cond

    def functional_content(game, sender_strat, receiver_strat):
        entries_common, entries_sender, entries_receiver = calc_entries(game, sender_strat, receiver_strat)
        sender_cond, receiver_cond = calc_conditions(game, receiver_strat)
        condition = calc_condition(game, receiver_strat)
        return info.normalize_axis(entries_common * condition, 0), info.normalize_axis(entries_sender * sender_cond, 0), info.normalize_axis(entries_receiver * receiver_cond, 0) # this normalizes column-wise


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
