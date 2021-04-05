import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
import copy
import torch
import pickle

from inference_tools.pose_inference import AlphaPose
from inference_tools.custom_multiprocessing import process_pool
from inference_tools.pasta_inference import pasta_model
from inference_tools.visualize import vis_tool

from activity2vec.dataset.hake_dataset import im_read, rgba2rgb
from activity2vec.ult.config import get_cfg
from activity2vec.ult.logging import setup_logging

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *

class Activity2Vec():
    """
        Activity2Vec pipeline class to predict interactions between humans and objects

        Attributes:
            - mode: can be in form of image/video (deprecated since we only accept webcam as the input)
            - cfg: configuration file
            - logger: for output logging purposes
            - vis_tool: a.k.a visualization tool, this is what will be shown in the output screen
            - alphapose: the AlphaPose object. See inference_tools.pose_inference for more information
            - pasta_model: PaSta or "Part States" model. See inference_tools.pasta_inference for more information

        Methods:
            - __init__ : class constructor, initialize attributes
            - inference : make an inference on an image/frame
    """
    def __init__(self, mode, cfg, logger):
        """
            Class constructor

            Arguments:
                - mode: str, 'Image' or 'Video' depending on the input desired by the user (deprecated)
                - cfg: str, Path to the configuration file (configs/a2v/a2v.yaml)
                - logger: activity2vec.ult.logging, output logs

            Returns:
                None
        """
        self.mode = mode
        self.cfg = cfg
        self.logger = logger
        self.vis_tool = vis_tool(cfg, mode)
        self.alphapose = AlphaPose(cfg.DEMO.DETECTOR, cfg.DEMO.YOLO_CFG, cfg.DEMO.YOLO_WEIGHT, cfg.DEMO.POSE_CFG, cfg.DEMO.POSE_WEIGHT, cfg.DEMO.TRACKER_WEIGHT, logger)
        self.pasta_model = pasta_model(cfg)
        logger.info('Loading Activity2Vec model from {}...'.format(cfg.DEMO.A2V_WEIGHT))
        
    def inference(self, image):
        """
            Make an inference based on the input

            Arguments:
                - image: numpy.ndarray, input image/frame

            Returns:
                - ori_image: numpy.ndarray, the original image
                - annos_cpu:  dict, dictionary of annotations (detached from GPU to CPU)
                - vis: inference_tools.visualize.vis_tool, visualization canvas
        """

        # input the frame directly
        ori_image = image
        alpha_image = image
        pose = self.alphapose.process(alpha_image)
        if pose is None:
#             self.logger.info('[Activity2Vec] no pose result for {:s}'.format(image_path))
            vis = ori_image
            vis = self.vis_tool.draw(vis, None, None, None, None, None, None)
            return ori_image, None, vis
        else:
            try:
                pasta_image, annos = self.pasta_model.preprocess(ori_image, pose['result'])

                human_ids = []
                for human in pose['result']:
                    human_idx = int(np.array(human['idx']).flatten()[0])
                    human_ids.append(human_idx)

                annos_cpu = copy.deepcopy(annos)
                pasta_image = pasta_image.cuda(non_blocking=True)
                for key in annos:
                    if isinstance(annos[key], dict):
                        for sub_key in annos[key]:
                            annos[key][sub_key] = annos[key][sub_key].cuda()
                            annos[key][sub_key] = annos[key][sub_key].squeeze(0)
                    else:
                        annos[key] = annos[key].cuda()
                        annos[key] = annos[key].squeeze(0)
                annos['human_bboxes'] = torch.cat([torch.zeros(annos['human_bboxes'].shape[0], 1).cuda(), annos['human_bboxes']], 1)
                annos['part_bboxes'] = torch.cat([torch.zeros(annos['part_bboxes'].shape[0], annos['part_bboxes'].shape[1], 1).cuda(), annos['part_bboxes']], 2)
                
                f_pasta, p_pasta, p_verb = self.pasta_model.inference(pasta_image, annos)
                vis = ori_image

                scores = annos_cpu['human_scores'][0].numpy()[:, 0]
                bboxes = annos_cpu['human_bboxes'][0].numpy()
                keypoints = annos_cpu['keypoints'][0].numpy()

                score_filter = scores > self.cfg.DEMO.SCORE_THRES
                scores = scores[score_filter]
                bboxes = bboxes[score_filter]
                keypoints = keypoints[score_filter]
                p_pasta = p_pasta[score_filter]
                p_verb = p_verb[score_filter]
                vis = self.vis_tool.draw(vis, bboxes, keypoints, scores, p_pasta, p_verb, human_ids)

                annos_cpu['human_bboxes'] = annos_cpu['human_bboxes'].squeeze(0)
                annos_cpu['part_bboxes'] = annos_cpu['part_bboxes'].squeeze(0)
                annos_cpu['keypoints'] = annos_cpu['keypoints'].squeeze(0)
                annos_cpu['human_scores'] = annos_cpu['human_scores'].squeeze(0)
                annos_cpu['skeletons'] = annos_cpu['skeletons'].squeeze(0)
                annos_cpu['f_pasta'] = f_pasta
                annos_cpu['p_pasta'] = p_pasta
                annos_cpu['p_verb'] = p_verb
                return ori_image, annos_cpu, vis

            except Exception as e:
#                 self.logger.info('[Activity2Vec] unsuccess for {:s}'.format(image_path))
                self.logger.info('{:s}'.format(str(e)))
                vis = ori_image
                vis = self.vis_tool.draw(vis, None, None, None, None, None, None)
                return ori_image, None, vis
        

def parse_args():
    """
        Parsing important and required arguments to execute the script

        Arguments:
            None

        Returns:
            args: dict, dictionary of arguments
    """

    parser = argparse.ArgumentParser(description='Activity2Vec Demo')

    parser.add_argument('--cfg', type=str, required=True, 
                        help='configuration file')
    parser.add_argument('--output', type=str, default='', 
                        help='output directory, empty string means do not output anything')
    parser.add_argument('--show-res', action='store_true', 
                        help='choose whether to show the output')
    parser.add_argument('--save-res', action='store_true', 
                        help='choose whether to save the raw results')
    parser.add_argument('--save-vis', action='store_true', 
                        help='choose whether to save the visualization results')
    parser.add_argument(
        "opts",
        help="See activity2vec/ult/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def setup():
    """
        Setup utility

        Arguments:
            None

        Returns:
            - cfg: activity2vec.ult.config, configuration file
            - args: argparse, arguments parsed
    """
    cfg = get_cfg()
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    cfg.freeze()
    return cfg, args
        
if __name__ == '__main__':
    # setup with a configuration file
    cfg, args = setup()
    if len(args.output) > 0:
        assert os.path.splitext(args.output)[-1] == '', 'output should be directory!'
        if args.save_vis:
            os.makedirs(os.path.join(args.output, 'vis'), exist_ok=True)
        if args.save_res:
            os.makedirs(os.path.join(args.output, 'res'), exist_ok=True)
    else:
        if args.save_vis or args.save_res:
            raise RuntimeError('output should not be empty when save_vis or save_res is set!')

    loggers = setup_logging(cfg.LOG_DIR, func='inference')
    logger  = loggers.Activity2Vec
    logger.info('cfg:\n')
    logger.info(cfg)
    args.logger = logger
    # initiate Activity2Vec pipeline
    a2v = Activity2Vec(args.mode, cfg, logger)
    
    # capture from webcam
    cap = cv2.VideoCapture(0)
    vises = []
    while True:
        ret, frame = cap.read()
        # make an inference for each frame
        ori_image, annos, vis = a2v.inference(frame)
        if args.show_res:
            if vis is None:
                vis = ori_image
            # show result
            cv2.imshow('Webcam Activity2Vec', vis)
            # quit upon hitting 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            vises.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
