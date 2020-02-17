import numpy as np


class MainEngine:
    def __init__(self):
        self.cultures = []
        self.cult_member_amt = []
        self.culture_bases = []
        self.critical_angles = []
        self.mu = 0

    def initialize_experiment(self, cultures, member_amt, bases, critical_angles):
        self.cultures = cultures
        self.cult_member_amt = member_amt
        self.culture_bases = bases
        self.critical_angles = critical_angles

    def clasterisation(self):
        #TODO
        print('kek')
