import torch
import math


class Beam():
    def __init__(self, config, h):
        self.beam_size = config.beam_size
        self.bos = config.bos
        self.eos = config.eos
        self.finish_flag = False
        self.cell = config.cell
        self.path = []

        self.path.append([[config.bos], h, 0])
        for _ in range(config.beam_size-1):
            self.path.append([[config.bos], h, -9999])

    def finish(self):
        for i in range(len(self.path)):
            if self.path[i][0][-1] != self.eos:
                return
        self.finish_flag = True

    # return the node of current node
    def get_node(self):
        a = []
        for i in range(len(self.path)):
            node = self.path[i][0]
            a.append(torch.tensor(node).type(torch.LongTensor))
        return torch.stack(a)

    # return the hidden state of current node
    def get_h(self):
        if self.cell == 'lstm':
            a = []
            b = []
            for i in range(len(self.path)):
                a.append(self.path[i][1][0])
                b.append(self.path[i][1][1])
            a = torch.stack(a)
            b = torch.stack(b)
            a = (a, b)
        else:
            a = []
            for i in range(len(self.path)):
                a.append(self.path[i][1])
            a = torch.stack(a)
        return a

    def max_path(self, candidate):
        pos = 0 # position of max
        v = -999 # value of max
        for i in range(len(candidate)):
            if candidate[i][-1] > v:
                v = candidate[i][-1]
                pos = i
        return pos

    def sort_path(self, candidate):
        # initialization path
        self.path = []
        for i in range(self.beam_size):
            self.path.append(candidate.pop(self.max_path(candidate)))

    def advance(self, h, data):
        self.finish()
        if self.finish_flag:
            return
        else:
            candidate = []
            for i in range(len(self.path)):
                sorted, indices = torch.sort(data[i], descending=True)
                pre_path = self.path[i][0]
                pre_scorce = self.path[i][-1]
                for k in range(self.beam_size):
                    p = pre_path.copy()
                    p.append(int(indices[k]))
                    if sorted[k].item() == 0:
                        scorce = -999 + pre_scorce
                    # print(sorted[k])
                    else:
                        scorce = math.log(sorted[k]) + pre_scorce
                    if self.cell == 'lstm':
                        candidate.append([p, (h[0][:, i], h[1][:, i]), scorce])
                    else:
                        candidate.append([p, h[i], scorce])
            self.sort_path(candidate)
