from torch import nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, prediction, target):
        batch = prediction.size(0)
        return self.loss(prediction.view(batch, -1), target.view(batch, -1))


class DiceLoss(nn.Module):
    def forward(self, prediction, target):
        batch = prediction.size(0)
        prediction = prediction.view(batch, -1)
        target = target.view(batch, -1)
        score = (2.0 * (prediction * target).sum(1) + 1.0) / (
            prediction.sum(1) + target.sum(1) + 1.0
        )
        return 1.0 - score.sum() / batch


class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.dice(prediction, target)
