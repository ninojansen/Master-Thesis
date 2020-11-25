from tqdm import tqdm
from misc.config import cfg
from misc.utils import mkdir_p
from datasets import prepare_data
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import shutil
import datetime
import dateutil.tz
import random
from apex import amp


def init_output_folders(output_loc):
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(
        output_loc, f"output_{cfg.DATASET_NAME}_{cfg.TEXT.ENCODER}_x{cfg.TREE.BASE_SIZE}_{timestamp}")
    os.mkdir(output_dir)
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


def train_step(data, netG, netD, optimizerG, optimizerD, batch_size, device):
    imags, caption_embeds, captions_str, class_ids, keys = (data)

    imgs = imags[0].to(device)
    caption_embeds = caption_embeds.to(device)

    real_features = netD(imgs)
    output = netD.COND_DNET(real_features, caption_embeds)
    errD_real = torch.nn.ReLU()(1.0 - output).mean()

    output = netD.COND_DNET(real_features[:(batch_size - 1)], caption_embeds[1:batch_size])
    errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

    # synthesize fake images
    noise = torch.randn(batch_size, 100)
    noise = noise.to(device)
    fake = netG(noise, caption_embeds)

    # G does not need update with D
    fake_features = netD(fake.detach())

    errD_fake = netD.COND_DNET(fake_features, caption_embeds)
    errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

    errD = errD_real + (errD_fake + errD_mismatch)/2.0
    optimizerD.zero_grad()
    optimizerG.zero_grad()
    errD.backward()
    optimizerD.step()

    # MA-GP
    interpolated = (imgs.data).requires_grad_()
    sent_inter = (caption_embeds.data).requires_grad_()
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
    output = netD.COND_DNET(features, caption_embeds)
    errG = - output.mean()
    optimizerG.zero_grad()
    optimizerD.zero_grad()
    errG.backward()
    optimizerG.step()

    return errD, errG, fake.data,


def train(output_dir, dataloader, netG, netD, state_epoch, batch_size, device, hide_progress=False):
    output_train_examples, output_models = init_output_folders(output_dir)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        mu_errD, mu_errG = 0, 0
        start = time.time()

        cnt = 0

        pbar = tqdm(total=len(dataloader), postfix={"d_loss": 0, "g_loss": 0})
        rand = random.randint(0, epoch // batch_size)
        for data in dataloader:

            errD, errG, fake_images = train_step(data, netG, netD,
                                                 optimizerG, optimizerD, batch_size, device)
            mu_errD += errD.item()
            mu_errG += errG.item()

            if not hide_progress:
                pbar.set_postfix({"d_loss":  errD.item(), "g_loss": errG.item()})
                pbar.update(1)

            if cnt == rand:
                vutils.save_image(fake_images, os.path.join(
                    output_train_examples, "gen_snapshot_{:04d}.png".format(epoch)),
                    normalize=True)
            cnt += 1

        pbar.set_description(
            f'Epoch {epoch} took {time.time()-start} sec, avg d_loss={mu_errD/cnt} g_loss={mu_errG/cnt}')
        pbar.close()

        if epoch % 10 == 0:
            print("10th epoch saving models...")
            torch.save(netG.state_dict(), os.path.join(output_models, 'netG_%03d.pth' % (epoch)))
            torch.save(netD.state_dict(), os.path.join(output_models, 'netD_%03d.pth' % (epoch)))


def sampling(netG, dataloader, device):
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for data in tqdm(dataloader):
            imags, caption_embeds, captions_str, class_ids, keys = (data)
            cnt += batch_size

            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise = noise.to(device)
                fake_imgs = netG(noise, caption_embeds)

            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp, i)
                im.save(fullpath)

# def evaluate(self, split_dir):
#     if cfg.TRAIN.NET_G == '':
#         print('Error: the path for models is not found!')
#     else:
#         # Build and load the generator
#         if split_dir == 'test':
#             split_dir = 'valid'
#         netG = G_NET()
#         netG.apply(weights_init)
#         netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
#         print(netG)
#         # state_dict = torch.load(cfg.TRAIN.NET_G)
#         state_dict = \
#             torch.load(cfg.TRAIN.NET_G,
#                         map_location=lambda storage, loc: storage)
#         netG.load_state_dict(state_dict)
#         print('Load ', cfg.TRAIN.NET_G)

#         # the path to save generated images
#         s_tmp = cfg.TRAIN.NET_G
#         istart = s_tmp.rfind('_') + 1
#         iend = s_tmp.rfind('.')
#         iteration = int(s_tmp[istart:iend])
#         s_tmp = s_tmp[:s_tmp.rfind('/')]
#         save_dir = '%s/iteration%d' % (s_tmp, iteration)

#         nz = cfg.GAN.Z_DIM
#         noise = Variable(torch.FloatTensor(self.batch_size, nz))
#         if cfg.CUDA:
#             netG.cuda()
#             noise = noise.cuda()

#         # switch to evaluate mode
#         netG.eval()
#         for step, data in enumerate(self.data_loader, 0):
#             imgs, t_embeddings, filenames = data
#             if cfg.CUDA:
#                 t_embeddings = Variable(t_embeddings).cuda()
#             else:
#                 t_embeddings = Variable(t_embeddings)
#             # print(t_embeddings[:, 0, :], t_embeddings.size(1))

#             embedding_dim = t_embeddings.size(1)
#             batch_size = imgs[0].size(0)
#             noise.data.resize_(batch_size, nz)
#             noise.data.normal_(0, 1)

#             fake_img_list = []
#             for i in range(embedding_dim):
#                 fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])
#                 if cfg.TEST.B_EXAMPLE:
#                     # fake_img_list.append(fake_imgs[0].data.cpu())
#                     # fake_img_list.append(fake_imgs[1].data.cpu())
#                     fake_img_list.append(fake_imgs[2].data.cpu())
#                 else:
#                     self.save_singleimages(fake_imgs[-1], filenames,
#                                             save_dir, split_dir, i, 256)
#                     # self.save_singleimages(fake_imgs[-2], filenames,
#                     #                        save_dir, split_dir, i, 128)
#                     # self.save_singleimages(fake_imgs[-3], filenames,
#                     #                        save_dir, split_dir, i, 64)
#                 # break
#             if cfg.TEST.B_EXAMPLE:
#                 # self.save_superimages(fake_img_list, filenames,
#                 #                       save_dir, split_dir, 64)
#                 # self.save_superimages(fake_img_list, filenames,
#                 #                       save_dir, split_dir, 128)
#                 self.save_superimages(fake_img_list, filenames,
#                                         save_dir, split_dir, 256)
