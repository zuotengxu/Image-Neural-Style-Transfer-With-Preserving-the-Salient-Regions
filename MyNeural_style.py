import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


from tensorboardX import SummaryWriter
import io
import requests
from PIL import Image
from torchvision import models
from torch.nn import functional as F

#load the model for the cam map generation,here we use the squeezenet
model_id = 1
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True).cuda()
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

features_blobs = []
# hook the feature extractor
def hook_feature(module, input, output):
    #features_blobs.append(output.data.cpu().numpy())
    features_blobs.append(output)
net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = torch.squeeze(params[-2])


# calculate and return the cam image through inner product operations
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    nc, h, w = feature_conv.shape
    cam = torch.matmul(weight_softmax[class_idx],feature_conv.view(nc, h*w))
    return cam



def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformer = TransformerNet()
    if (args.premodel != ""):
        transformer.load_state_dict(torch.load(args.premodel))
        print("load pretrain model:"+args.premodel)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.repeat(args.batch_size, 1, 1, 1)
    style = utils.preprocess_batch(style)
    if args.cuda:
        style = style.cuda()
    style_v = Variable(style, volatile=True)
    style_v = utils.subtract_imagenet_mean_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]


    hori=0 
    writer = SummaryWriter(args.logdir,comment=args.logdir)
    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_cate_loss = 0.
        agg_cam_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()
            y = transformer(x)  
            xc = Variable(x.data.clone(), volatile=True)
            #print(y.size()) #(4L, 3L, 224L, 224L)

            
            # Calculate focus loss and category loss
            y_cam = utils.depreprocess_batch(y)
            y_cam = utils.subtract_mean_std_batch(y_cam) 
            
            xc_cam = utils.depreprocess_batch(xc)
            xc_cam = utils.subtract_mean_std_batch(xc_cam)
            

            del features_blobs[:]
            logit_x = net(xc_cam)
            logit_y = net(y_cam)
            
            label=[]
            cam_loss = 0
            for i in range(len(xc_cam)):
                h_x = F.softmax(logit_x[i])
                probs_x, idx_x = h_x.data.sort(0, True)
                label.append(idx_x[0])
                
                h_y = F.softmax(logit_y[i])
                probs_y, idx_y = h_y.data.sort(0, True)
                
                x_cam = returnCAM(features_blobs[0][i], weight_softmax, idx_x[0])
                x_cam = Variable(x_cam.data,requires_grad = False)
 
                y_cam = returnCAM(features_blobs[1][i], weight_softmax, idx_y[0])
                
                cam_loss += mse_loss(y_cam, x_cam)
            
            #the focus loss
            cam_loss *= 80
            #the category loss
            label = Variable(torch.LongTensor(label),requires_grad = False).cuda()
            cate_loss = 10000 * torch.nn.CrossEntropyLoss()(logit_y,label)
         
         

           
            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            #f_xc_c = Variable(features_xc[1].data, requires_grad=False)
            #content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)


            f_xc_c = Variable(features_xc[2].data, requires_grad=False)
            content_loss = args.content_weight * mse_loss(features_y[2], f_xc_c)
            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])
            #add the total four loss and backward
            total_loss = style_loss + content_loss  + cam_loss + cate_loss
            total_loss.backward()
            optimizer.step()

            #something for display
            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            agg_cate_loss += cate_loss.data[0]
            agg_cam_loss += cam_loss.data[0]
            
            writer.add_scalar("Loss_Cont", agg_content_loss / (batch_id + 1), hori)
            writer.add_scalar("Loss_Style", agg_style_loss / (batch_id + 1), hori)
            writer.add_scalar("Loss_CAM", agg_cam_loss / (batch_id + 1), hori)
            writer.add_scalar("Loss_Cate", agg_cate_loss / (batch_id + 1), hori)
            hori += 1
            
            if (batch_id + 1) % args.log_interval == 0:
               mesg = "{}Epoch{}:[{}/{}] content:{:.2f} style:{:.2f} cate:{:.2f} cam:{:.2f}  total:{:.2f}".format(
                    time.strftime("%a %H:%M:%S"),e + 1, count, len(train_dataset),
                                 agg_content_loss / (batch_id + 1),
                                 agg_style_loss / (batch_id + 1),
                                 agg_cate_loss / (batch_id + 1),
                                 agg_cam_loss / (batch_id + 1),
                                 (agg_content_loss + agg_style_loss + agg_cate_loss + agg_cam_loss ) / (batch_id + 1)
               )
               print(mesg)
               
            if (batch_id + 1) % 2500 == 0:    
                transformer.eval()
                transformer.cpu()
                save_model_filename = "epoch_" + str(e+1) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(transformer.state_dict(), save_model_path)
                transformer.cuda()
                transformer.train()
                print("saved at ",count)
    
    
    
    
    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)
    
    writer.close()
    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def stylize(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.content_scale)
    content_image = content_image.unsqueeze(0)

    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))

    if args.cuda:
        style_model.cuda()

    output = style_model(content_image)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--premodel", type=str, default="",
                                  help="pretrain model")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
                                  help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=224,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=224,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--style-weight", type=float, default=20.0,
                                  help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=50,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--logdir", type=str, default="Logdir",help="")
    train_arg_parser.add_argument("--self", type=int, default=50,
                                  help="number of images after which the training loss is logged, default is 500")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")


    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
