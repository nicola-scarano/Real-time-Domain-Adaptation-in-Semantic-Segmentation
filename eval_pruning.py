from dataset.CamVid import CamVid
import torch
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
from utils import test_pruning, prune_module, prune_global
import tqdm


def eval(model, dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = predict.detach().cpu().numpy()

            # predict = colour_code_segmentation(np.array(predict), label_info)
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = label.detach().cpu().numpy()
            # label = colour_code_segmentation(np.array(label), label_info)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou_dict, miou = cal_miou(miou_list, csv_path)
        # print('IoU for each class:')
        # for key in miou_dict:
        #     print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision, miou

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # create dataset and dataloader
    test_path = os.path.join(args.data, 'test')
    # test_path = os.path.join(args.data, 'train')
    test_label_path = os.path.join(args.data, 'test_labels')
    # test_label_path = os.path.join(args.data, 'train_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset = CamVid(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width), mode='test')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    print('start pruning analysis...\n')
    precs, mious = test_pruning(prune_global, eval, dataloader, args, csv_path)

    # Visualization of the results
    sparsities = [0]
    sparsities = sparsities + [i*0.05 for i in range(1, 18)]

    fig = plt.figure(1, figsize=(20,8))
    ax = plt.subplot(111)
    ax.set_xlabel('sparsity', fontsize=15)
    ax.grid()

    ax.plot(sparsities,precs, c='steelblue', label="precision")
    ax.plot(sparsities,mious, c='orange', label="miou")

    ax.legend()
    plt.savefig("pruning_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './MLDL/best_dice_loss_adv_paper6/best_dice_loss.pth',
        '--data', './BiseNetv2-master/data/CamVid/',
        '--cuda', '0',
        '--context_path', 'resnet101',
        '--num_classes', '12'
    ]
    main(params)