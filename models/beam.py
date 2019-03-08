class beam():
    def __init__(self, config, h):
        self.beam_size = config.beam_size
        self.path = [[[config.bos], h, 0.0]]

    def get_h(self):
        pass

    def get_node(self):
        pass

    def max_path(self):
        pass

    def sort_path(self):
        pass

    def advance(self, ):
        for i in range(len(self.path)):
