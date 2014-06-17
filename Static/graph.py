import commoninterest as ct
import networkx as nx

class Graph(ct.Game):
    def __init__(self, payoffs):
        ct.Game.__init__(self, payoffs)
        self.graph = nx.DiGraph()
        self.add_edges()

    def create_nodes(self):
        for strat in self.wholepurestrats:
            self.graph.add_node(nice_sender(strat))
            self.graph.add_node(nice_receiver(strat))

    def add_edges(self):
        for strat in self.wholepurestrats:
            brs = self.best_response_receiver(strat)
            for rstrat in brs:
                self.graph.add_edge(nice_sender(strat), nice_receiver(rstrat))
            bss = self.best_response_sender(strat)
            for sstrat in bss:
                self.graph.add_edge(nice_receiver(strat), nice_sender(sstrat))

    def isolated_scc(self, scc):
        condition = []
        setscc = set(scc)
        for strat in scc:
            reachable = set(nx.single_source_shortest_path_length(self.graph,
                                                              strat).keys())
            condition.append(setscc == reachable)
        return all(condition)

    def sink(self, scc):
        sinklist = []
        setscc = set(scc)
        for strat in self.graph.nodes():
            reachable = set(nx.single_source_shortest_path_length(self.graph,
                                                              strat).keys())
            if len(setscc & reachable) > 0:
                sinklist.append(strat)
        if set(sinklist) == set(self.graph.nodes()):
            print("The whole graph is in the basin")
        return sinklist




def nice_sender(strat):
    parts = [''.join(['M', str(state.index(1) + 1)]) for state in strat]
    return '-'.join(parts)

def nice_receiver(strat):
    parts = [''.join(['A', str(message.index(1) + 1)]) for message in strat]
    return '-'.join(parts)


