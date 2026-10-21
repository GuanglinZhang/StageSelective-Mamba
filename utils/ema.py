import copy

import torch


class EMAShadow:
    """EMA used by the formal protocol; it is updated after every optimizer step."""

    def __init__(self, model, decay=0.999):
        if float(decay) != 0.999:
            raise ValueError("The paper protocol locks EMA decay to 0.999")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        shadow = self.module.state_dict()
        source = model.state_dict()
        if tuple(shadow) != tuple(source):
            raise RuntimeError("EMA source and shadow state keys differ")
        for name, shadow_value in shadow.items():
            source_value = source[name].detach()
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(source_value)
        self.updates += 1
