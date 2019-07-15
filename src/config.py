
import os
import sys

class Config():
    def __init__(self):
        self.identity = str(sys.argv[1])
        self.address = str(sys.argv[2])
        self.port = str(sys.argv[3])
        self.eosurl = str(sys.argv[4])
        self.logdir = str(sys.argv[5])
        self.k = 3

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return  "\nconfig = {\n\tidentity: " + self.identity + " \n" +\
                "\taddress: " + self.address + " \n" +\
                "\tport: " + self.port + "  \n" +\
                "\tk: " + str(self.k) + "  \n" + \
                "\teosurl: " + self.eosurl + " \n}."
