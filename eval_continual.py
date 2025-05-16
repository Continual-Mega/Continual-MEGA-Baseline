import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from CLIP.clip import create_model
from CLIP.adapter import CLIPAD
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset.continual import ImageDataset
import csv
import logging
from CoOp import PromptMaker
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_logger(output_dir):
    # set log file
    log_file = f"{output_dir}/log.log"
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--img_size', type=int, default=336)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument("--meta_file", type=str, default="meta_files/scenario1_5classes_tasks.json")
    parser.add_argument("--base_meta_file", type=str, default="meta_files/scenario1_base.json")    
    parser.add_argument("--num_tasks", type=int, default=12, help="number of tasks")
    parser.add_argument("--n_learnable_token", type=int, default=8, help="number of learnable token")
    parser.add_argument("--checkpoints", type=str, default="checkpoints/scenario2/30classes_tasks", help="folder path to checkpoints")
    parser.add_argument("--checkpoint_base", type=str, default="checkpoints/scenario2/checkpoint_base.pth", help="checkpoint base path")
    parser.add_argument("--task_id", type=int, default=1, help="test task id")  # 0 - base classes
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--data_root", type=str, default="data")

    args = parser.parse_args()

    setup_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")

    save_path = args.save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # for logging
    logger = get_logger(save_path)
    logger.info(args)
    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    
    # prompt learner
    prompts = {
        "normal": [
        "This is an example of a normal object",
        "This is a typical appearance of the object",
        "This is what a normal object looks like",
        "A photo of a normal object",
        "This is not an anomaly",
        "This is an example of a standard object.",
        "This is the standard appearance of the object.",
        "This is what a standard object looks like.",
        "A photo of a standard object.",
        "This object meets standard characteristics."
    ],
        "abnormal": [
        "This is an example of an anomalous object",
        "This is not the typical appearance of the object",
        "This is what an anomaly looks like",
        "A photo of an anomalous object",
        "An anomaly detected in this object",
        "This is an example of an abnormal object.",
        "This is not the usual appearance of the object.",
        "This is what an abnormal object looks like.",
        "A photo of an abnormal object.",
        "An abnormality detected in this object."
    ]
    }

    clip_model.device = device
    clip_model.to(device)

    prompt_maker = PromptMaker(
        prompts=prompts,
        clip_model=clip_model,
        n_ctx= args.n_learnable_token,
        CSC = True,
        class_token_position=['end'],
    ).to(device)

    model = CLIPAD(clip_model=clip_model, features=args.features_list)
    model.to(device)
    model.eval()

    # load checkpoint
    if args.task_id == 0:
        checkpoint_path = args.checkpoint_base
    else:
        checkpoint_path = f"{args.checkpoints}/checkpoint_task_{args.task_id}.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    prompt_maker.prompt_learner.load_state_dict(checkpoint['prompt_state_dict'])
    model.adapters.load_state_dict(checkpoint['adapters'])
    logger.info(f"load model from {checkpoint_path}")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # save results
    num_tasks = args.num_tasks + 1
    results_image = np.full((num_tasks, num_tasks), 0)  # save for csv image-level
    results_pixel = np.full((num_tasks, num_tasks), 0)  # save for csv pixel-level

    # load saved_results
    csv_image = f"{save_path}/results_image.csv"
    csv_pixel = f"{save_path}/results_pixel.csv"
    
    if os.path.exists(csv_image):
        with open(csv_image, mode="r") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if not i == 0:
                    results_image[i-1] = row
        logger.info(f"load previous results from {csv_image}")

    if os.path.exists(csv_pixel):
        with open(csv_pixel, mode="r") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if not i == 0:
                    results_pixel[i-1] = row
        logger.info(f"load previous results from {csv_pixel}")

    prompt_maker.eval()
    model.eval()
    
    task_all_meta_info = json.load(open(args.meta_file, 'r'))

    # test all previous tasks
    for i in range(args.task_id + 1):
        if i == 0: # base classes
            task_meta = json.load(open(args.base_meta_file, 'r'))
            logging.info(f"start base task test")
        else:
            task_meta = task_all_meta_info[f"task_{i}"]
            logging.info(f"start task_{i} test")

        class_name_list = list(task_meta["test"].keys())
        test_dataset_list = [ImageDataset(data_root=args.data_root, meta_file=task_meta, resize=args.img_size, mode="test", test_class=class_name) for class_name in class_name_list]
        test_loader_list = [torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs) for test_dataset in test_dataset_list]

        with torch.cuda.amp.autocast(), torch.no_grad():
            # test all class
            seg_ap_list = []
            img_auc_list = []
            prompt_maker.eval()
            model.eval()
            text_features = prompt_maker()

            for test_loader, class_name in zip(test_loader_list, class_name_list):
                logger.info(f"start test {class_name}")
                roc_auc_im, seg_ap = test(args, model, test_loader, text_features, device)
                logger.info(f'{class_name} P-AP : {round(seg_ap,4)}')
                logger.info(f'{class_name} I-AUC : {round(roc_auc_im, 4)}')
                seg_ap_list.append(seg_ap)
                img_auc_list.append(roc_auc_im)

            seg_ap_mean = np.mean(seg_ap_list)
            img_auc_mean = np.mean(img_auc_list)

            logger.info(f'Average P-AP : {round(seg_ap_mean,4)}')
            logger.info(f'Average I-AUC : {round(img_auc_mean, 4)}')

        # save results csv (i task)
        seg_ap_mean = round(seg_ap_mean,4)
        img_auc_mean = round(img_auc_mean, 4)
        results_image[args.task_id, i] = img_auc_mean
        results_pixel[args.task_id, i] = seg_ap_mean

        logger.info(f"save results csv task {i}")

        # save results csv
        with open(csv_image, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Base"] + ["Task " + str(i + 1) for i in range(num_tasks-1)])
            for row in results_image:
                writer.writerow(row)
        with open(csv_pixel, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Base"] + ["Task " + str(i + 1) for i in range(num_tasks-1)])
            for row in results_pixel:
                writer.writerow(row)

def test(args, model, test_loader, text_features, device):
    gt_list = []
    gt_mask_list = []

    seg_score_map_zero = []
    image_scores = []
    for data in tqdm(test_loader):
        image, mask, cls_name, label = data['image'], data['mask'], data['cls_name'], data['anomaly']
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ada_patch_tokens = model(image)
            ada_patch_tokens = [p[0, 1:, :] for p in ada_patch_tokens]

            anomaly_maps = []
            image_score = 0
            for layer in range(len(ada_patch_tokens)):
                ada_patch_tokens[layer] /= ada_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * ada_patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))

                # image
                anomaly_score = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                image_score += anomaly_score.max()
        
                anomaly_maps.append(anomaly_map)

            score_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            score_map = F.interpolate(score_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=args.img_size, mode='bilinear', align_corners=True)
            score_map = torch.softmax(score_map, dim=1)[:, 1, :, :]
            score_map = score_map.squeeze(0).cpu().numpy()
            seg_score_map_zero.append(score_map)
            image_scores.append(image_score.cpu() / len(ada_patch_tokens))
                        
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(label.cpu().detach().numpy())

            
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(seg_score_map_zero)
    image_scores = np.array(image_scores)

    roc_auc_im = roc_auc_score(gt_list, image_scores)

    seg_pr = average_precision_score(gt_mask_list.flatten(), segment_scores.flatten())

    return roc_auc_im, seg_pr


if __name__ == '__main__':
    main()


