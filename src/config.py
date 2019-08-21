
import os
import sys

class Config():
    def __init__(self):
        self.identity = str(sys.argv[1])
        self.serve_address = str(sys.argv[2])
        self.bind_address = str(sys.argv[3])
        self.port = str(sys.argv[4])
        self.eosurl = str(sys.argv[5])
        self.logdir = str(sys.argv[6])
        self.k = 3

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return  "\nconfig = {\n\tidentity: " + self.identity + " \n" +\
                "\tserve_address: " + self.serve_address + " \n" +\
                "\tbind_address: " + self.bind_address + " \n" +\
                "\tport: " + self.port + "  \n" +\
                "\tk: " + str(self.k) + "  \n" + \
                "\teosurl: " + self.eosurl + " \n}."
