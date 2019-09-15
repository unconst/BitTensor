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
        self.alpha = 0.01
        self.time_till_expire = 10

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return  "\nconfig = {\n\tidentity: " + self.identity + " \n" +\
                "\tserve_address: " + self.serve_address + " \n" +\
                "\tbind_address: " + self.bind_address + " \n" +\
                "\teosurl: " + self.eosurl ++ " \n" +\
                "\tport: " + self.port + "  \n" +\
                "\tk: " + str(self.k) + "  \n" + \
                "\talpha: " + str(self.alpha) + "  \n" +\
                "\ttime_till_expire: " + str(self.time_till_expire) + " \n}."
