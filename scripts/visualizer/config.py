class Config():
    def __init__(self):
        self.eos_account = "bittensoracc"
        self.eos_table = "metagraph"
        self.eos_scope = "bittensoracc"
        self.eos_key_type = "i64"
        self.eos_code = "bittensoracc"
        self.eos_port = "8888"
        self.eos_url = "http://host.docker.internal:{}".format(self.eos_port)
        
        self.eos_get_table_rows = self.eos_url + "/v1/chain/get_table_rows"