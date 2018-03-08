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
