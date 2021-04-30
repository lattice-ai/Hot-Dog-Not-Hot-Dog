# External
import unittest

import hub
import torch
from torchvision import models


class TestModel(unittest.TestCase):
    def testForward(self):
        self.model = models.resnet18(pretrained=True)
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        torch.manual_seed(0)
        self.x = torch.rand((4, 3, 224, 224))
        self.assertEqual(list(self.model.forward(self.x).shape), [4, 2])


if __name__ == "__main__":
    unittest.main()
