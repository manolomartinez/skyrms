import numpy as np

def n_in_m(total, strats, weights):
    """
    Create mixed strategies with <strats> strat choices and <weights> possible
    weights for pure strats.
    """
    if strats==1:
        return np.array([[total]])
    else:
        relevant_weights = [weight for weight in weights if weight <= total]
        for weight in relevant_weights:
            lower = n_in_m(total - weight, strats - 1, weights)
            newsols = conc_element(weight, lower)
            try:
                sols = np.append(sols, newsols, axis=0)
            except:
                sols = newsols
        return sols


def conc_element(elem, array):
    col = array.shape[0]
    elem_vector = elem * np.ones((col, 1))
    return np.concatenate((elem_vector, array), axis=1)


