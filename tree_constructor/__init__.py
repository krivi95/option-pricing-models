import math

class Calculate:
    """
    calculates node values on stock and option trees
    
    """
    def __init__(self, S, X, T, r, u, dp, opttype='Europ'):
        #
        self.opttype = opttype
        self.S = S
        self.X = X
        self.T = T
        self.r = r
        self.u = u
        self.d = 1/u
        self.dp = dp
    
    def p_value(self):
        return (math.exp(self.r) - self.d) / (self.u - self.d)

    def option_value(self, Vu, Vd):
        """
        returns option value from Vu and Vd
        
        """
        return math.exp(-self.r) * (self.p_value() * Vu + (1 - self.p_value()) * Vd)
    
    def stock_prices(self):
        """
        returns binary tree for stock prices
        
        """
        tree = [[self.S]]
        for t in range(self.T):
            for state in tree:
                new_state = set()
                for p in range(len(state)):
                    if len(new_state) == 0:
                        Su = state[p] * self.u
                        Sd = state[p] * self.d
                        new_state.add(round(Sd, self.dp))
                        new_state.add(round(Su, self.dp))
                    else:
                        Sd = state[p] * self.u
                        new_state.add(round(Sd, self.dp))
            tree.append(sorted(list(new_state)))
        return tree
    
    def call_values(self):
        """
        returns reversed binary tree of call option prices
        
        """
        end_values = map(lambda s: round(max(s - self.X, 0), self.dp), 
                         reversed(self.stock_prices()[-1]))
        if self.opttype != 'Europ':
            end_values = [round(max(s - self.X, prev), self.dp) for prev,s in zip(end_values, reversed(self.stock_prices()[-1]))]
        reverse_tree = [[*end_values]]
        for t in range(self.T):
            for state in reverse_tree:
                previous_state = []
                for VT in range(1, len(state)):
                    Vt = self.option_value(state[VT-1], state[VT])
                    previous_state.append(max(round(Vt, self.dp), 0))
            reverse_tree.append(previous_state)
        return reverse_tree
        
    def put_values(self):
        """
        returns reversed binary tree of put option prices
        
        """
        end_values = map(lambda s: round(max(self.X - s, 0), self.dp), 
                         reversed(self.stock_prices()[-1]))
        if self.opttype != 'Europ':
            end_values = [round(max(self.X - s, prev), self.dp) for prev,s in zip(end_values, reversed(self.stock_prices()[-1]))]
        reverse_tree = [[*end_values]]
        for t in range(self.T):
            for state in reverse_tree:
                previous_state = []
                for VT in range(1, len(state)):
                    Vt = self.option_value(state[VT-1], state[VT])
                    previous_state.append(max(round(Vt, self.dp), 0))
            reverse_tree.append(previous_state)
        return reverse_tree