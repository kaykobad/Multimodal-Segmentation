import argparse
import os
import json
# import dataloaders
from dataloaders.datasets.mcubes_dataset import MCubeSDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from PIL import Image

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.mmsnet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator, ConfusionMatrix
import matplotlib.image
import cv2
# from dataloaders.nyudv2_cmnext_transforms import get_patched_train_augmentation, get_val_augmentation
from dataloaders.datasets.patched_nyudv2_cmnext_dataset import PatchedNYU, PatchedNYUViz
from dataloaders.nyudv2_cmnext_transforms_for_cl import get_patched_train_augmentation, get_val_augmentation
from dataloaders.datasets.patched_nyudv2_cmnext_dataset_for_cl import PatchedNYUForCL, PatchedNYUForCLViz

## TODO: Some part of this code has been changed for missing modality test
## TODO: Check before running code
# m = []
# fm = []
mIoUs = {}
FWIoUs = {}   
output_path = 'predictions/Nyudv2Patch/Blurred_RGB_to_Grayscale_Model' 
input_path = 'datasets/SmallDataset2' 
model_path = "run/SmallDataset/MMSNetAttnCL/experiment_0/Patched-NYU-P64-B64-BlurredRGB-Std-Avg-R50-T-RGB_best_test.pth.tar"
top_k = 10
split = 'val'


def get_my_labels():
    " r,g,b"
    return np.array([
        [128,0,0],
        [0,128,0],
        [128,128,0],
        [0,0,128],
        [128,0,128],
        [0,128,128],
        [128,128,128],
        [64,0,0],
        [192,0,0],
        [64,128,0],
        [192,128,0],
        [64,0,128],
        [192,0,128],
        [64,128,128],
        [192,128,128],
        [0,64,0],
        [128,64,0],
        [0,192,0],
        [128,192,0],
        [0,64,128],
        [128,64,128],
        [0,192,128],
        [128,192,128],
        [64,64,0],
        [192,64,0],
        [64,192,0],
        [192,192,0],
        [64,64,128],
        [192,64,128],
        [64,192,128],
        [192,192,128],
        [0,0,64],
        [128,0,64],
        [0,128,64],
        [128,128,64],
        [0,0,192],
        [128,0,192],
        [0,128,192],
        [128,128,192],
        [64,0,64]])


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 40
    label_colours = get_my_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] = 0
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def generate_viz(result_list, save_name):
    fig, axes = plt.subplots(nrows=top_k, ncols=3, figsize=(3*3, top_k*3))

    for i in range(top_k):
        name, miou = result_list[i]
        rgb = Image.open(f"{input_path}/RGB_Patch/{name}.jpg")
        blurred_rgb = Image.open(f"{input_path}/GBlurred_RGB_Patch/{name}.jpg")
        blurred_grayscale = Image.open(f"{input_path}/GBlurred_Patch/{name}.jpg").convert("L")
        hha = Image.open(f"{input_path}/HHA_Patch/{name}.jpg")
        target = Image.open(f"{input_path}/Label_Patch/{name}.png").convert("L")
        pred = Image.open(f"{output_path}/{split}/{name}.png").convert("L")

        axes[i, 0].imshow(blurred_rgb)
        axes[i, 0].set_title(f"Blurred RGB {name}")
        axes[i, 0].axis("off")

        # axes[i, 1].imshow(hha)
        # axes[i, 1].set_title(f"HHA {name}")
        # axes[i, 1].axis("off")

        axes[i, 1].imshow(decode_segmap(np.asarray(target).astype(np.uint8)-1), cmap='jet', alpha=1.0)
        axes[i, 1].set_title(f"Target Label {name}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(decode_segmap(np.asarray(pred)), cmap='jet', alpha=1.0)
        axes[i, 2].set_title(f"Blurred RGB Prediction - IoU: {round(miou*100, 2)}%")
        axes[i, 2].axis("off")
    
    plt.savefig(f'{output_path}/{split}/{save_name}.png')

        
class TesterMultimodal(object):
    def __init__(self, args):
        self.args = args

        # Define Tensorboard Summary
        # self.summary = TensorboardSummary(f'{os.path.dirname(args.pth_path)}/test')
        # self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        traintransform = get_patched_train_augmentation([64, 64], seg_fill=255)
        valtransform = get_val_augmentation([64, 64])
        trainset = PatchedNYU("datasets/SmallDataset", 'train', traintransform)
        valset = PatchedNYUForCLViz("datasets/SmallDataset2", split, valtransform)
        self.train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=2, pin_memory=False, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=2, pin_memory=False, shuffle=False)
        self.nclass = 40

        # Define network
        input_dim = 3

        checkpoint = torch.load(model_path)
        print("Best Prediction:", checkpoint['best_pred'])
        
        # self.model = MMSNetForPatchedRGBD(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn,
        #                 use_rgb=args.use_rgb,
        #                 use_depth=args.use_depth,
        #                 norm=args.norm)

        self.model = MMSNetForPatchedRGBDForCL(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        use_rgb=args.use_rgb,
                        use_depth=args.use_depth,
                        norm=args.norm,
                        is_teacher=False)

        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        self.model.cuda()

        ## TODO: Remove this line
        # self.args.use_depth = False

        self.model.module.load_state_dict(checkpoint['state_dict'])
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(pytorch_total_params)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=0).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        
        # # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # self.confmat = ConfusionMatrix(num_classes=self.nclass, average=None)

    def test(self, epoch=0):
        self.model.eval()
        self.evaluator.reset()
        # self.confmat.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        # image_all = None
        # target_all = None
        # output_all = None
        for i, (images, labels, name) in enumerate(tbar):
            image, target, depth, brgb, bgray = images[0], labels, images[1], images[2], images[3]
            name = name[0]
            if len(depth.shape) != 4:  # avoide automatic squeeze in later version of pytorch data loading
                depth = depth.unsqueeze(1)
                # print(depth.shape)

            if self.args.cuda:
                # image, target, depth, hha, depth3 = image.cuda(), target.cuda(), depth.cuda(), hha.cuda(), depth3.cuda()
                image, target, depth, brgb, bgray = image.cuda(), target.cuda(), depth.cuda(), brgb.cuda(), bgray.cuda()

            rgb = image if self.args.use_rgb else None
            depth = depth if self.args.use_depth else None

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    # output = self.model(rgb=bgray, depth=depth)
                    output, _, proj1, proj2 = self.model(rgb=brgb, depth=depth)
                loss = self.criterion(output, target)
                # if np.isnan(loss.item()):
                #     print(" XXXXXX Image Name:", name, "Output:", output, "Target:", target, output.shape, target.shape)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            pred = output.data.cpu().numpy()
            target_ = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # if image_all is None:
            #     image_all  = image.cpu().clone()
            #     target_all = target.cpu().clone()
            #     output_all = output.cpu().clone()
            # else:
            #     image_all  = torch.cat(( image_all, image.cpu().clone()),dim=0)
            #     target_all = torch.cat((target_all, target.cpu().clone()),dim=0)
            #     output_all = torch.cat((output_all, output.cpu().clone()),dim=0)
                
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_, pred)
            # self.confmat.update((output, target.long()))

            # My evaluator
            ev = Evaluator(self.nclass)
            ev.add_batch(target_, pred)
            this_mIoU = ev.Mean_Intersection_over_Union()
            this_FWIoU = ev.Frequency_Weighted_Intersection_over_Union()
            # --> print(f"Image {i} IoU: {this_mIoU} and FWIoU: {this_FWIoU}")
            # m.append(this_mIoU)
            # fm.append(this_FWIoU)
            # FWIoUs.append(this_FWIoU)
            mIoUs[name] = this_mIoU
            FWIoUs[name] = this_FWIoU

            # Save the images
            # print(f"Output shape: {output.shape}, Target Shape: {target.shape}")
            # img = image.cpu().numpy()[0]
            # # print(img.shape)
            # # img = img.reshape(1024, 1024, 3)
            # t = target.cpu().numpy()
            # t = t.reshape(64, 64, 1)
            p = pred.reshape(64, 64, 1)
            # print(f"Image reShape: {img.shape}, Output reshape: {p.shape}, Target reShape: {t.shape}")
            # matplotlib.image.imsave(f'predictions/{i}-traget.png', t)
            # matplotlib.image.imsave(f'predictions/{i}-prediction.png', o)
            # cv2.imwrite(f'predictions/{i}-image.png', img)
            # cv2.imwrite(f'predictions/{i}-target.png', t)
            ## >--> cv2.imwrite(f'{output_path}/{split}/{name}.png', p)
            # cv2.imwrite(f'predictions/{i}-image.png', img)

            # out = output.data.cpu().numpy()[0]
            # # print(f"Out shape: {out.shape}, Type: {type(out)}")
            # for j in range(out.shape[0]):
            #     filter = out[j, :, :].astype(int)
            #     # print(f"Filter shape: {filter.shape}")
            #     filter = filter.reshape(1024, 1024, 1)
            #     cv2.imwrite(f'predictions/{i}-Filter-{j}.png', filter)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        confusion_matrix = self.evaluator.confusion_matrix
        # print(confusion_matrix)
        ## >--> np.save(f'{output_path}/{split}/confusion_matrix.npy',confusion_matrix)

        # self.writer.add_scalar('test/mIoU', mIoU, epoch)
        # self.writer.add_scalar('test/Acc', Acc, epoch)
        # self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
        # self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
        # self.summary.visualize_test_image(self.writer, self.args.dataset, image_all, target_all, output_all, 0)
        
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # print("Individual mIoU:", np.nanmean(np.array(m)), "FWmIoU:", np.nanmean(np.array(fm)))
        # print('CmfMat mIoU:', self.confmat.miou(ignore_index=None))
        # print("mIoUs:", mIoUs)
        # print("FWIoUs:", FWIoUs)

        ## >--> Dump result to file
        # with open(f'{output_path}/{split}/mIoUs.json', 'w') as fp:
        #     json.dump(mIoUs, fp)
        # with open(f'{output_path}/{split}/FWIoUs.json', 'w') as fp:
        #     json.dump(FWIoUs, fp)

        # # >--> Generate the visualization
        # # Sort by ascending order (1, 2, 3, ...)
        # sort_by_value = dict(sorted(mIoUs.items(), key=lambda item: item[1]))

        # # Best k
        # best_k = list(sort_by_value.items())[-top_k:]
        # generate_viz(best_k, 'Best_Blurred_RGB_to_Blurred_Grayscale_Model_Prediction')

        # # Worst k
        # worst_k = list(sort_by_value.items())[:top_k]
        # generate_viz(worst_k, 'Worst_Blurred_RGB_to_Blurred_Grayscale_Model_Prediction')

        # print(f"Output shape: {output.shape}, Target Shape: {target.shape}")
        # matplotlib.image.imsave(f'predictions/{i}-traget.png', target.cpu().numpy())
        # matplotlib.image.imsave(f'predictions/{i}-prediction.png', output.data.cpu().numpy())
        #print('Loss: %.3f' % test_loss)



# def test_visualizer(args):
#     # Load the model
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model_path = "run/multimodal_dataset/MCubeSNet/experiment_10/checkpoint-latest-pytorch.pth.tar"
#     model = torch.load(model_path)
#     model.to(device)

#     # Prepare dataloader

#     # Take five images

#     # Fed to model

#     # Save the outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet50', 'resnet', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'xception_adv','resnet_condconv'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='multimodal_dataset',
                        choices=['SmallDataset', 'pascal', 'coco', 'cityscapes', 'kitti', 'kitti_advanced', 'kitti_advanced_manta', 'handmade_dataset', 'handmade_dataset_stereo', 'multimodal_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--ratio', type=float, default=None, metavar='N',
                        help='number of ratio in RGFSConv (default: 1)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--use-rgb', action='store_true', default=False,
                        help='use rgb')
    parser.add_argument('--use-depth', action='store_true', default=False,
                        help='use depth')

    # propagation and positional encoding option
    parser.add_argument('--propagation', type=int, default=0,
                        help='image propagation length (default: 0)')
    parser.add_argument('--positional-encoding', action='store_true', default=False,
                        help='use positional encoding')
    parser.add_argument('--use-aolp', action='store_true', default=False,
                        help='use aolp')
    parser.add_argument('--use-dolp', action='store_true', default=False,
                        help='use dolp')
    parser.add_argument('--use-nir', action='store_true', default=False,
                        help='use nir')
    parser.add_argument('--use-pol', action='store_true', default=False,
                        help='use pol')
    parser.add_argument('--use-segmap', action='store_true', default=False,
                        help='use segmap')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--list-folder', type=str, default='list_folder1')
    parser.add_argument('--is-multimodal', action='store_true', default=False,
                        help='use multihead architecture')
    parser.add_argument('--enable-se', action='store_true', default=False,
                        help='use se block on decoder')
    parser.add_argument('--pth-path', type=str, default=None,
                        help='set the pth file path')
    parser.add_argument('--norm', type=str, default='avg',
                        help='avg, bn or bnr')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    # if args.epochs is None:
    #     epoches = {
    #         'coco': 30,
    #         'cityscapes': 200,
    #         'pascal': 50,
    #         'kitti': 50,
    #         'kitti_advanced': 50
    #     }
    #     args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    # if args.lr is None:
    #     lrs = {
    #         'coco': 0.1,
    #         'cityscapes': 0.01,
    #         'pascal': 0.007,
    #         'kitti' : 0.01,
    #         'kitti_advanced' : 0.01
    #     }
    #     args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    # if args.checkname is None:
    #     args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    # input('Check arguments! Press Enter...')
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    tester = TesterMultimodal(args)
    exit

    # if args.is_multimodal:
    #     print("USE Multimodal Model")
    #     tester = TesterMultimodal(args)
    # else:
    #     tester = TesterAdv(args)
    # print('Starting Epoch:', tester.args.start_epoch)
    # print('Total Epoches:', tester.args.epochs)
    tester.test()
    # tester.writer.close()
    print(args)