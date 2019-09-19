import os
import sys
import tensorflow as tf

# Tensorflow flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

# TODO (const): This needs to be global bittensor.config
flags.DEFINE_string("identity", "xxxxxxx", "Nueron Identity")
flags.DEFINE_string("serve_address", "0.0.0.0", "Address serve synapse.")
flags.DEFINE_string("bind_address", "0.0.0.0", "Address bind synapse.")
flags.DEFINE_string("port", "9090", "Port to serve on.")
flags.DEFINE_string("eosurl", "http://0.0.0.0:8888", "EOS Url.")
flags.DEFINE_string("logdir", "/tmp/", "logginf directory.")
flags.DEFINE_integer("k", 3, "Out edge degree.")
flags.DEFINE_float("alpha", 0.01, "Learning rate.")


class Config():
    def __init__(self):
        self.identity = FLAGS.identity
        self.serve_address = FLAGS.serve_address
        self.bind_address = FLAGS.bind_address
        self.port = FLAGS.port
        self.eosurl = FLAGS.eosurl
        self.logdir = FLAGS.logdir
        self.k = FLAGS.k
        self.alpha = FLAGS.alpha

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
