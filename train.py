import time

import torch.utils.data

from dataloaders.unaligned_dataset import UnalignedDataset
from options.train_options import TrainOptions
from models.pix2pix import Pix2PixClassifierModel
from util.visualizer import Visualizer


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
    visualizer = Visualizer(op)

    total_steps = 0

    for epoch in range(1, op.citer + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dl):
            iter_start_time = time.time()
            total_steps += op.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            model.optimize_C_parameters()

            if total_steps % op.print_freq == 0:
                errors = model.get_C_errors()
                t = (time.time() - iter_start_time) / op.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if epoch % op.save_epoch_freq == 0:
            print('saving model at the end of epoch {}, iters {}'.format(epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            pass

        print('End of epoch {} / {} \t Time Taken: {:.3f} sec'.format(epoch, op.citer, time.time() - epoch_start_time))

        if epoch <= op.citer:
            model.update_C_rate()

    model.old_lr = op.lr
    for epoch in range(1, op.niter + op.niter_decay + 1):
        epoch_start_time = time.time()

        for i, data in enumerate(dl):
            iter_start_time = time.time()
            total_steps += op.batchSize
            epoch_iter = total_steps - dataset_size * (epoch -1)
            model.set_input(data)
            model.optimize_parameters()

            # if total_steps % op.display_freq == 0:
            #     visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % op.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / op.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # if op.display_id > 0:
                #     visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, op, errors)

            if total_steps % op.save_latest_freq == 0:
                print('saving the latest model (epoch {}, total_steps {}'.format(epoch, total_steps))
                model.save('latest')

        if epoch % op.save_epoch_freq == 0:
            print('saving the model at the end of epoch {}, iters {}'.format(epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, op.niter + op.niter_decay, time.time() -
                                                                  epoch_start_time))

        if epoch > op.niter:
            model.update_learning_rate()
    print(1)