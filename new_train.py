import time

import torch.utils.data

from dataloaders.unaligned_dataset import UnalignedDataset
from options.train_options import TrainOptions
from models.pix2pix import Pix2PixClassifierModel
# from util.visualizer import Visualizer


if __name__ == '__main__':
    op = TrainOptions().parse()
    ds = UnalignedDataset()
    ds.initialize(op)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=op.batchSize,
        shuffle=True,
        num_workers=int(op.nThreads)
    )
    dataset_size = len(dl)
    model = Pix2PixClassifierModel(op)

    x = torch.randn(5, 3, 256, 256)

    y = model.vgg_model(x.cuda())
    z = model.netC(y[4])
    print(1)