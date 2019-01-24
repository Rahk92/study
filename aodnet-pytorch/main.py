import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.optim
import os
import network
import dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train():
    orig_path = "E:/Data/dehazing/aod_net/original_image/"
    haze_path = "E:/Data/dehazing/aod_net/training_image/"
    lr = 0.0001
    wd = 0.0001
    num_epoch = 5
    train_batch_size = 8
    val_batch_size = 8
    num_workers = 4
    grad_clip_norm = 0.1
    snapshot_iter = 10
    display_iter = 10
    snapshots_folder = "snapshot/"
    output_folder = "output/"
    if not os.path.exists(snapshots_folder):
        os.mkdir(snapshots_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    model = network.k_estimation_module().cuda()
    model.apply(weights_init)

    train_dataset = dataloader.dataset_loader(orig_path, haze_path)
    val_dataset = dataloader.dataset_loader(orig_path, haze_path, mode="val")
    train_loader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # sets the module in training mode
    model.train()

    for epoch in range(num_epoch):
        print("Epoch ", epoch)
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean = model(img_haze)
            loss = criterion(clean, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip_norm)
            optimizer.step()
            if ((iteration + 1) % display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % snapshot_iter) == 0:
                torch.save(model.state_dict(), snapshots_folder + "Epoch" + str(epoch) + '.pth')

            # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = model(img_haze)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         output_folder + str(iter_val + 1) + ".jpg")

        torch.save(model.state_dict(), snapshots_folder + "dehazer.pth")

        print("Done")


if __name__ == "__main__":

    train()