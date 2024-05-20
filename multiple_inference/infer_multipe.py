
import os
import pickle
import shutil
import subprocess

mydict = {

#"cfgs/kitti_models/pointpillar.yaml" : "/media/ava/DATA3/DATA/thrisha/models/pointpillar_7728.pth",
#"cfgs/kitti_models/pointrcnn.yaml" : "/media/ava/DATA3/DATA/thrisha/models/pointrcnn_7870.pth",
#"cfgs/kitti_models/second.yaml" : "/media/ava/DATA3/DATA/thrisha/models/second_7862.pth",
#"cfgs/kitti_models/second_iou.yaml" : "/media/ava/DATA3/DATA/thrisha/models/second_iou7909.pth",
#"cfgs/kitti_models/voxel_rcnn_car.yaml" : "/media/ava/DATA3/DATA/thrisha/models/voxel_rcnn_car_84.54.pth"
 #"cfgs/nuscenes_models/cbgs_pp_multihead.yaml" : "/media/ava/DATA3/DATA/thrisha/models/pp_multihead_nds5823_updated.pth",
 "cfgs/nuscenes_models/bevfusion.yaml" : "/media/ava/DATA3/DATA/thrisha/models/cbgs_bevfusion.pth",
#"cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml" : "/media/ava/DATA3/DATA/thrisha/models/VoxelNeXt_Argo2.pth"

}

pickle_files = "/media/ava/DATA3/DATA/thrisha/OpenPCDet/multiple_inference/pickle_files"
if os.path.exists(pickle_files):
    shutil.rmtree(pickle_files)
data_path = "/media/ava/DATA3/DATA/murali/nuscenes/v1.0-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151606998613.pcd.bin"

for cfg_file_path, ckpt_path in mydict.items():
        
    model_name = os.path.basename(cfg_file_path) #centerpoint.yaml
    model_name = os.path.splitext(model_name)[0] #centerpoint
    output_path = os.path.join(pickle_files, model_name, "all_preds.pickle")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command_string = f"python3 demo.py --cfg_file {cfg_file_path} --ckpt {ckpt_path} --data_path {data_path} --output_pickle {output_path}"

    print(f"Running {command_string}\n\n")
    
    result = subprocess.run(command_string, shell=True)    
    # if result.returncode != 0:
    #     print(result.stderr.decode())
    # result.stdout.decode()
    
    print(f"Done with {model_name}")
