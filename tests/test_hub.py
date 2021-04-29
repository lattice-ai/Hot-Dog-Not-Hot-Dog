# External
import unittest

import hub
import torch


class TestDataLoader(unittest.TestCase):
    def test_dataloader(self):
        """
        Test for importing Dataset using the dataloader class
        """
        pytorch_dataset = hub.Dataset(
            "sauravmaheshkar/resized-hot-dog-not-hot-dog")
        pytorch_dataset = pytorch_dataset.to_pytorch(
            lambda x: (x["resized_image"], x["label"]))
        train_loader = torch.utils.data.DataLoader(pytorch_dataset,
                                                   batch_size=32,
                                                   num_workers=4)
        self.assertIsInstance(pytorch_dataset, list)
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(train_loader,
                              hub.collections.dataset.core.TorchDataset)


if __name__ == "__main__":
    unittest.main()
