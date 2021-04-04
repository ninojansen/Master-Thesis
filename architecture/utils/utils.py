from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def generate_figure(image, text):
    image = image.clone()
    size = image.shape[2]
    min = float(image.min())
    max = float(image.max())
    image.clamp_(min=min, max=max)
    image.add_(-min).div_(max - min + 1e-5)

    processed_image = image.mul(255).add_(0.5).clamp_(
        0, 255).permute(
        1, 2, 0).to(
        'cpu', torch.uint8).numpy()

    plt.clf()
    figure = plt.figure()
    plt.rcParams.update({'font.size': 20})
    plt.imshow(processed_image)
    plt.xticks([])
    plt.yticks([])
    plt.title(add_linebreaks(text, n_split=5))
    plt.tight_layout()
    figure.canvas.draw()
    buf = figure.canvas.tostring_rgb()
    ncols, nrows = figure.canvas.get_width_height()
    shape = (nrows, ncols, 3)
    img_arr = np.fromstring(buf, dtype=np.uint8).reshape(shape)
    plt.close("all")
    resize = transforms.Resize(128)
    out = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    return out(torch.from_numpy(img_arr).permute(2, 0, 1))


def gen_image_grid(images_tensor, labels):
    # Create a figure to contain the plot.
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    images_tensor = images_tensor.clone()
    max_range = 25
    if images_tensor.size(0) < 25:
        max_range = images_tensor.size(0)
    min = float(images_tensor.min())
    max = float(images_tensor.max())
    images_tensor.clamp_(min=min, max=max)
    images_tensor.add_(-min).div_(max - min + 1e-5)

    processed_images = images_tensor.mul(255).add_(0.5).clamp_(
        0, 255).permute(
        0, 2, 3, 1).to(
        'cpu', torch.uint8).numpy()

    plt.clf()
    figure = plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 6})
    plt.tight_layout()

    for i in range(16):
        # Start next subplot.
        plt.subplot(4, 4, i + 1, xlabel=add_linebreaks(labels[i]))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(processed_images[i])
    # plt.show()
    figure.canvas.draw()
    buf = figure.canvas.tostring_rgb()
    ncols, nrows = figure.canvas.get_width_height()
    shape = (nrows, ncols, 3)
    img_arr = np.fromstring(buf, dtype=np.uint8).reshape(shape)
    # io_buf = io.BytesIO()
    # figure.savefig(io_buf, format='raw')
    # io_buf.seek(0)
    # img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
    #                      newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
    # io_buf.close()
    plt.close("all")

    return torch.from_numpy(img_arr).permute(2, 0, 1)


def add_linebreaks(str, n_split=7):
    split = str.split()
    out = []
    for i, part in enumerate(split):
        out.append(part)
        if (i + 1) % n_split == 0:
            out.append("\n")
    return " ".join(out)


def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        print()
    param = 0
    try:
        bias = (i.bias is not None)
    except:
        bias = False
    if not bias:
        param = model_parameters[j].numel() + model_parameters[j + 1].numel()
        j = j + 2
    else:
        param = model_parameters[j].numel()
        j = j + 1
    print(str(i) + "\t" * 3 + str(param))
    total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")
