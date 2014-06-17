import itertools as it
import math
import random
import commoninterest as ci

cstardict = {
    0.0: [[0, 0, 0]],
    0.3333333333333333: [[0, 0, 1]],
    0.6666666666666666: [[0, 1, 1], [0, 0, 2]],
    1.0: [[0, 1, 2], [1, 1, 1], [0, 0, 3]],
    1.3333333333333333: [[0, 1, 3], [1, 1, 2], [0, 2, 2], [0, 0, 4]],
    1.6666666666666667: [
        [1, 1, 3], [0, 1, 4], [0, 0, 5],
        [1, 2, 2], [0, 2, 3]],
    2.0: [
        [0, 2, 4],
        [1, 1, 4],
        [1, 2, 3],
        [0, 1, 5],
        [0, 3, 3],
        [2, 2, 2],
        [0, 0, 6]],
    2.3333333333333335: [
        [1, 3, 3],
        [2, 2, 3],
        [1, 2, 4],
        [0, 2, 5],
        [1, 1, 5],
        [0, 1, 6],
        [0, 3, 4]],
    2.6666666666666665: [
        [1, 2, 5],
        [0, 3, 5],
        [2, 3, 3],
        [1, 3, 4],
        [2, 2, 4],
        [1, 1, 6],
        [0, 2, 6],
        [0, 4, 4]],
    3.0: [
        [2, 2, 5],
        [2, 3, 4],
        [1, 2, 6],
        [1, 4, 4],
        [1, 3, 5],
        [3, 3, 3],
        [0, 3, 6],
        [0, 4, 5]],
    3.3333333333333335: [
        [1, 4, 5],
        


        [0, 5, 5],
        [2, 2, 6],
        [2, 4, 4],
        [1, 3, 6]],
    3.6666666666666665: [
        [2, 4, 5],
        [3, 3, 5],
        [1, 4, 6],
        [1, 5, 5],
        [0, 5, 6],
        [2, 3, 6],
        [3, 4, 4]],
    4.0: [
        [2, 5, 5],
        [3, 4, 5],
        [1, 5, 6],
        [2, 4, 6],
        [3, 3, 6],
        [4, 4, 4],
        [0, 6, 6]],
    4.333333333333333: [[4, 4, 5], [3, 5, 5], [1, 6, 6], [2, 5, 6], [3, 4, 6]],
    4.666666666666667: [[3, 5, 6], [4, 4, 6], [4, 5, 5], [2, 6, 6]],
    5.0: [[4, 5, 6], [3, 6, 6], [5, 5, 5]],
    5.333333333333333: [[5, 5, 6], [4, 6, 6]],
    5.666666666666667: [[5, 6, 6]],
    6.0: [[6, 6, 6]]
}

possible_cstars = [
    0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0,
    3.33, 3.67, 4.0]

possible_orders = [
    i for i in it.permutations(range(4)) if i[3] == 1 or i[3] == 2]


def matrix_to_intertwined(sender, receiver):
    return [item for pair in zip(
        it.chain(*sender), it.chain(*receiver)) for item in pair]


def points(sender, receiver, element1, element2):
    pairwise = math.floor(
        abs(ci.preferable(sender, element1, element2) -
            ci.preferable(receiver, element1, element2)))
    return pairwise


def kendall(sender, receiver):
    kendall = sum([points(sender, receiver,
                          pair[0], pair[1]) for pair in
                   it.combinations(range(4), 2)])
    return kendall


def game_with_cstar(cstar):
    for key in cstardict:
        if abs(cstar - key) < 0.1:
            cstar = key
    target_kendall = random.choice(cstardict[cstar])
    sender = []
    receiver = []
    for i in range(3):
        senderstate = list(random.choice(possible_orders))
        receiverstate = list(random.choice(possible_orders))
        while kendall(senderstate, receiverstate) != target_kendall[i]:
            senderstate = list(random.choice(possible_orders))
            receiverstate = list(random.choice(possible_orders))
        sender.append(senderstate)
        receiver.append(receiverstate)
    return sender, receiver


def payoffs_with_order(matrix):
    payoffs = []
    for state in matrix:
        candidate = [random.randrange(6) for i in range(3)]
        candidateplusmean = candidate + [sum(candidate)/len(candidate)]
        while ci.order_list(candidateplusmean) != state:
            candidate = [random.randrange(6) for i in range(3)]
            candidateplusmean = candidate + [sum(candidate)/len(candidate)]
        payoffs.append(candidate)
    return payoffs


def create_game(pair, target):
    osender, oreceiver = game_with_cstar(pair[0])
    game = ci.Game(matrix_to_intertwined(payoffs_with_order(osender),
                                         payoffs_with_order(oreceiver)))
    if "sender" in target:
        compare = game.petersender
    else:
        compare = game.peterreceiver
    while compare != pair[1] or game.petersender != pair[1]:
        osender, oreceiver = game_with_cstar(pair[0])
        game = ci.Game(matrix_to_intertwined(payoffs_with_order(osender),
                                             payoffs_with_order(oreceiver)))
        if "sender" in target:
            compare = game.petersender
        else:
            compare = game.peterreceiver
    return game


def create_game2(cstar):
    osender, oreceiver = game_with_cstar(cstar)
    game = ci.Game(matrix_to_intertwined(payoffs_with_order(osender),
                                         payoffs_with_order(oreceiver)))
    return game


def ultrabad():
    game = create_game([6.0, 4.0], "receiver")
    print(game.payoffs)
    a, b, c, info = game.info_in_equilibria()
    print(info)
    while info < 10e-4:
        game = create_game2(6.0)
        print(game.payoffs)
        a, b, c, info = game.info_in_equilibria()
        print(info)
    return game.payoffs

def create_constant_sum(dimension):
    """
    Return a square <dimension>x<dimension> constant-sum game
    """
    originalpayoffs = ci.payoffs(dimension)
    constant_sum_per_state = [random.randrange(100) for _ in range(dimension)]
    constant_sum_payoffs = []
    for state in range(dimension):
        for act in range(dimension):
            sender = originalpayoffs[state + 2 * act]
            receiver = originalpayoffs[state + 2 * act + 1]
            average = (sender + receiver)/2
            norm_sender = sender - average
            norm_receiver = receiver - average
            newsender = norm_sender + constant_sum_per_state[state]
            newreceiver = norm_receiver + constant_sum_per_state[state]
            constant_sum_payoffs += [newsender, newreceiver]
    return constant_sum_payoffs
