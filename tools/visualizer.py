import argparse
import glob
from pathlib import Path
import logging
import pickle

logger = logging.getLogger() 

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
import sys
sys.path.append("/home/mcw/Documents/MCW/OpenPCDet")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/mcw/Documents/MCW/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/mcw/Downloads/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/0000000000.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--pickle', type=str, default='/home/mcw/Downloads/all_preds.pickle',
                        help='pickle file to be visualised')

    args = parser.parse_args()
    print(cfg.keys())
    cfg_from_yaml_file(args.cfg_file, cfg)
    print(cfg.keys())

    return args, cfg


def main():
    args, cfg = parse_config()
    print('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    print(f'Total number of samples: \t{len(demo_dataset)}')

    with open(args.pickle, "rb") as f:
        all_preds = pickle.load(f)
    
    for idx, data_dict in enumerate(demo_dataset):
        print(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        pred_dicts = all_preds[idx]

        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)

    print('Demo done.')


if __name__ == '__main__':
    main()
