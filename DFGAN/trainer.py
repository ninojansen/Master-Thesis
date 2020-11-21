from tqdm import tqdm
from misc.config import cfg
from datasets import prepare_data
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import shutil


def init_output_folders(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    output_train_examples = os.path.join(output_dir, "train_examples")

    if not os.path.exists(output_train_examples):
        os.makedirs(output_train_examples)

    # self.output_checkpoints = os.path.join(output_dir, "checkpoints")
    # if not os.path.exists(self.output_checkpoints):
    #     os.makedirs(self.output_checkpoints)

    output_models = os.path.join(output_dir, "models")
    if not os.path.exists(output_models):
        os.makedirs(output_models)

    # self.output_loss = os.path.join(output_dir, "loss")
    # if not os.path.exists(self.output_loss):
    #     os.makedirs(self.output_loss)
    return output_train_examples, output_models


def save_examples(output_dir, fake_images, number):
    plt.clf()
    it_batch_size = fake_images.shape[0]
    n_rows, n_cols = 4, 4

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, dpi=100, sharex=True, sharey=True)
    fig.tight_layout()
    ax = ax.flatten()
    for i in range(n_rows * n_cols):
        ax[i].axis('off')
        if i < it_batch_size:
            ax[i].imshow((fake_images[i].cpu().reshape(256, 256, 3) + 1) / 2)
            # ax[i].set_title(f'{captions[i]} 64x64',
            #                 fontdict={'fontsize': 7, 'family': 'sans-serif',  'weight': 'normal'})

    plt.savefig(os.path.join(output_dir, "gen_snapshot_{:04d}.png".format(number)))
    plt.close("all")


def train_step(data, netG, netD, text_encoder, optimizerG, optimizerD, batch_size, device):
    imags, captions_ix, captions_str, cap_lens, class_ids, keys = prepare_data(data)

    sent_emb = text_encoder.encode(captions_str)
    sent_emb = torch.from_numpy(sent_emb).to(device)
    sent_emb = sent_emb.detach()
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    # hidden = text_encoder.init_hidden(batch_size)
    # words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

    imgs = imags[0].to(device)

    real_features = netD(imgs)
    output = netD.COND_DNET(real_features, sent_emb)
    errD_real = torch.nn.ReLU()(1.0 - output).mean()

    output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
    errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

    # synthesize fake images
    noise = torch.randn(batch_size, 100)
    noise = noise.to(device)
    fake = netG(noise, sent_emb)

    # G does not need update with D
    fake_features = netD(fake.detach())

    errD_fake = netD.COND_DNET(fake_features, sent_emb)
    errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

    errD = errD_real + (errD_fake + errD_mismatch)/2.0
    optimizerD.zero_grad()
    optimizerG.zero_grad()
    errD.backward()
    optimizerD.step()

    # MA-GP
    interpolated = (imgs.data).requires_grad_()
    sent_inter = (sent_emb.data).requires_grad_()
    features = netD(interpolated)
    out = netD.COND_DNET(features, sent_inter)
    grads = torch.autograd.grad(outputs=out,
                                inputs=(interpolated, sent_inter),
                                grad_outputs=torch.ones(out.size()).cuda(),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0, grad1), dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = torch.mean((grad_l2norm) ** 6)
    d_loss = 2.0 * d_loss_gp
    optimizerD.zero_grad()
    optimizerG.zero_grad()
    d_loss.backward()
    optimizerD.step()

    # update G
    features = netD(fake)
    output = netD.COND_DNET(features, sent_emb)
    errG = - output.mean()
    optimizerG.zero_grad()
    optimizerD.zero_grad()
    errG.backward()
    optimizerG.step()

    return errD, errG, fake.data,


def train(output_dir, dataloader, netG, netD, text_encoder, state_epoch, batch_size, device):
    output_train_examples, output_models = init_output_folders(output_dir)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    snapshot_interval = 0
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        mu_errD, mu_errG = 0, 0
        start = time.time()

        cnt = 0
        pbar = tqdm(total=len(dataloader), postfix={"d_loss": 0, "g_loss": 0})
        for data in dataloader:

            errD, errG, fake_images = train_step(data, netG, netD, text_encoder,
                                                 optimizerG, optimizerD, batch_size, device)
            mu_errD += errD.item()
            mu_errG += errG.item()

            cnt += 1
            snapshot_interval += 1
            pbar.set_postfix({"d_loss":  errD.item(), "g_loss": errG.item()})
            pbar.update(1)
            if snapshot_interval % 100 == 0:
                vutils.save_image(fake_images, os.path.join(
                    output_train_examples, "gen_snapshot_{:04d}.png".format(
                        snapshot_interval // 100)),
                    normalize=True)
               # save_examples(output_train_examples, fake_img, snapshot_interval // 5)
        pbar.set_description(
            f'Epoch {epoch} took {time.time()-start} sec, avg d_loss={mu_errD/cnt} g_loss={mu_errG/cnt}')
        pbar.close()

        if epoch % 10 == 0:
            print("10th epoch saving models...")
            torch.save(netG.state_dict(), os.path.join(output_models, 'netG_%03d.pth' % (epoch)))
            torch.save(netD.state_dict(), os.path.join(output_models, 'netD_%03d.pth' % (epoch)))
