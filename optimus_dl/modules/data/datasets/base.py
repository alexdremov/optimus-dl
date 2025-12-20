import torchdata.nodes


class BaseDataset(torchdata.nodes.BaseNode):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

    def load_state_dict(self, state_dict):
        self.reset(state_dict)
