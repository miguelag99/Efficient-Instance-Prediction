import os
from types import SimpleNamespace
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from torchvision.io import read_image

from prediction.utils.geometry import (
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    invert_matrix_egopose_numpy,
    mat2pose_vec,
)
from prediction.utils.instance import convert_instance_mask_to_center_and_offset_label


class ImageDataAugmentator:

    def __init__(self, config: SimpleNamespace,
                 mode: Literal['train', 'val', 'test'] = 'train',
                 p_grid: float = 0.0) -> None:

        self.config = config
        self.mode = mode
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.p_grid = p_grid


    def __call__(self, images: torch.Tensor, intrinsics: torch.Tensor, 
                 ret_original_im: bool = False) -> torch.Tensor:
              
        T, N, C, H, W = images.shape
        
        if ret_original_im:
            original_images = images.clone()/255.0
        else:
            original_images = None

        OH, OW = self.config.IMAGE.ORIGINAL_DIM
        FH, FW = self.config.IMAGE.FINAL_DIM

        # Parameters for resize and crop.
        resize = max(FH / OH, FW / OW)
        RH, RW = (int(OH * resize), int(OW * resize))
        TC = RH - FH

        # Resize and crop the images. TODO
        images = images.view(T * N, C, H, W)
        images = F.resize(images, (RH, RW), antialias=True)
        images = F.crop(images, TC, 0, RH - TC, RW)
        images = images.view(T, N, C, RH - TC, RW)
        
        # Update intrinsics.
        updated_intrinsics = intrinsics.clone()
        
        updated_intrinsics[:, :, 0, 0] *= resize
        updated_intrinsics[:, :, 0, 2] *= resize
        updated_intrinsics[:, :, 1, 1] *= resize
        updated_intrinsics[:, :, 1, 2] *= resize
        
        updated_intrinsics[:, :, 0, 2] -= 0   # No left crop
        updated_intrinsics[:, :, 1, 2] -= TC
        
        images = images / 255.0

                
        if self.mode == 'train':
            # Normalize the image and color jitter.
            images = F.adjust_brightness(images, torch.rand(1).item() + 0.5)
                      
            # Grid mask. Cover the image with a grid of black squares.
            if torch.rand(1).item() <= self.p_grid:
                n_squares = torch.randint(0, 12, size=(1,)).item()
                rx = torch.randint(0, images.shape[-2], size=(n_squares,))
                ry = torch.randint(0, images.shape[-1], size=(n_squares,))
                i = torch.randint(0, 6, size=(n_squares,))

                for sx, sy, si in zip(rx, ry, i):
                    images[:, si, :, sx:sx+20, sy:sy+20] = 0.0
            
            
        images = F.normalize(images, self.mean, self.std)
            
        return images, updated_intrinsics, original_images

                

class NuscenesDataset(Dataset):

    def __init__(self, config: SimpleNamespace, mode = 'train',
                 return_orig_images: bool = False) -> None:
        
        self.nusc = NuScenes(
            version=config.DATASET.VERSION,
            dataroot=config.DATASET.DATAROOT,
            verbose=True
        )
        self.config = config
        self.sequence_length = config.TIME_RECEPTIVE_FIELD + config.N_FUTURE_FRAMES
        self.mode = mode

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            config.LIFT.X_BOUND, config.LIFT.Y_BOUND, config.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )


        # Check if image dimensions are correct.
        scale_ratio = self.config.IMAGE.RESIZE_SCALE
        assert int(self.config.IMAGE.ORIGINAL_HEIGHT*scale_ratio-self.config.IMAGE.TOP_CROP) == \
            self.config.IMAGE.FINAL_DIM[0]
        assert int(self.config.IMAGE.ORIGINAL_WIDTH*scale_ratio) == \
            self.config.IMAGE.FINAL_DIM[1]

        self.scenes = self._get_scenes()
        self.ixes = self._get_samples()
        self.indices = self._get_indices()

        

        # self.normalise_image = torchvision.transforms.Compose(
        #     [
        #     # torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                      std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        self.return_orig_images = return_orig_images
        self.ida = ImageDataAugmentator(config, self.mode, p_grid=0.0)


    def _get_scenes(self) -> list:
        """Obtain the list of scenes in the given split.

        Attributes:
            config.dataset.version (str): version of nuScenes dataset.
            config.mode (str): mode of the dataset (train, val).        
        Returns:
            list: _description_
        """
        scenes = {
            "v1.0-trainval": {
                "train": splits.train,
                "val": splits.val,
            },
            "v1.0-test": {
                "train": [],
                "val": splits.test,
            },
            "v1.0-mini": {
                "train": splits.mini_train,
                "val": splits.mini_val,
            },
        }
        return scenes[self.config.DATASET.VERSION][self.mode]



    def _get_samples(self):
        """Find and sort the samples in the given split by scene.

        Attributes:
            scenes (list): list of scenes in the given split.
        Returns:
            list: list of samples in the given split.
        """
        samples = [sample for sample in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [sample for sample in samples if self.nusc.get('scene',sample['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    

    def _get_indices(self):
        """Group the sample indices by sequence length.

        Attributes:
            ixes (list): list of samples in the given split.
            config.n_frames (int): number of frames in the sequence.
        Returns:
            np.ndarray: array of indices grouped by sequence length.
        """
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)


    def _get_top_lidar_pose(self, rec):
        """
        Obtain the vehicle attitude at the current moment.
        """
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data',rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot


    def __len__(self) -> int:
        return len(self.indices)
    
    def get_input_data(self, sample_info_indices):
        """Obtain the input image as well as the intrinsics and extrinsics parameters
        of the corresponding camera.

        Args:
            sample_info (dict): a dict containing the information tokens of the sample.
        """
        scale_ratio = self.config.IMAGE.RESIZE_SCALE
        T = self.sequence_length
        N = self.config.DATASET.N_CAMERAS
        H1 = self.config.IMAGE.ORIGINAL_HEIGHT
        W1 = self.config.IMAGE.ORIGINAL_WIDTH
        TC = self.config.IMAGE.TOP_CROP
        CAMERAS = self.config.IMAGE.NAMES

        images = torch.zeros((T, N, 3, H1, W1), dtype=torch.float)
        intrinsics = torch.zeros((T, N, 3, 3), dtype=torch.float)
        extrinsics = torch.zeros((T, N, 4, 4), dtype=torch.float)
        lidar_2_sensor = torch.zeros((T, N, 4, 4), dtype=torch.float)

        for i, n in enumerate(sample_info_indices):

            sample_info = self.ixes[n]

            # From lidar egopose to world.
            lidar_sample = self.nusc.get('sample_data',
                                         sample_info['data']['LIDAR_TOP'])
            lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
            yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
            lidar_rotation = Quaternion(scalar=np.cos(yaw / 2),
                                        vector=[0, 0, np.sin(yaw / 2)])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])
            
            # From lidar sensor to lidar frame
            lidar_sensor_sample = self.nusc.get('calibrated_sensor',
                                                lidar_sample['calibrated_sensor_token'])
            sensor_rotation = Quaternion(lidar_sensor_sample['rotation'])
            sensor_translation = np.array(lidar_sensor_sample['translation'])[:, None]
            lidar_sensor_to_lidar_ego = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            lidar_ego_to_lidar_sensor = np.linalg.inv(lidar_sensor_to_lidar_ego)

            for j, cam in enumerate(CAMERAS):
                                  
                camera_sample = self.nusc.get('sample_data', sample_info['data'][cam])

                # From world to ego pose.
                car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
                egopose_rotation = Quaternion(car_egopose['rotation']).inverse
                egopose_translation = -np.array(car_egopose['translation'])[:, None]
                world_to_car_egopose = np.vstack([
                    np.hstack((egopose_rotation.rotation_matrix,
                               egopose_rotation.rotation_matrix @ egopose_translation)),
                    np.array([0, 0, 0, 1])
                ])

                # From egopose to sensor
                sensor_sample = self.nusc.get('calibrated_sensor',
                                              camera_sample['calibrated_sensor_token'])
                intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
                sensor_rotation = Quaternion(sensor_sample['rotation'])
                sensor_translation = np.array(sensor_sample['translation'])[:, None]
                car_egopose_to_sensor = np.vstack([
                    np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                    np.array([0, 0, 0, 1])
                ])
                car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

                ## Combine all the transformation.
                # From sensor to lidar_frame.
                lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
                sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()
                # From sensor to lidar sensor (sensor -> ego -> map -> ego' -> sensor)
                cam_2_lidar_sensor = lidar_ego_to_lidar_sensor @ np.linalg.inv(lidar_to_sensor)
                lidar_sensor_2_cam_sensor = torch.from_numpy(np.linalg.inv(cam_2_lidar_sensor)).float()
                # Direct projection form 3D point in LiDAR sensor frame to camera plane
                # ext_intrinsic = np.eye(4)
                # ext_intrinsic[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                # lidar_2_img = ext_intrinsic @ lidar_2_cam_sensor

                # Load image.
                image_filename = os.path.join(self.nusc.dataroot,
                                              camera_sample['filename'])
                # image = torch.from_numpy(
                #     cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
                #     .transpose(2, 0, 1)
                # ).float()
                images[i,j,:,:] = read_image(image_filename).float()

                # # Normalize image.
                # image = self.normalise_image(image)

                intrinsics[i,j,:,:] = intrinsic
                extrinsics[i,j,:,:] = sensor_to_lidar
                lidar_2_sensor[i,j,:,:] = lidar_sensor_2_cam_sensor

                # # Apply data augmentation to the images and update instrinsics.
                
        images, intrinsics, original_img = self.ida(images, intrinsics,
                                                    self.return_orig_images)

        return images, intrinsics, extrinsics, lidar_2_sensor, original_img

    def record_instance(self, sample_info_indices):
        """
        Record information about each visible instance in the sequence,
        assigning a unique ID to each instance.
        
        Args:
            sample_info_indices (list): list of indices of the samples in the sequence.
        Returns:
            instance_map (dict): a dictionary mapping each instance token to a unique
            ID.
            instance_dict (dict): a dictionary containing detailed information about
            each instance token.
            egopose_list (np.ndarray): array of egoposes of the sequence.
            visible_instance_set (set): set of instance tokens that are visible in the
            sequence.
        """

        instance_map = {}
        instance_dict = {}
        egopose_list = np.empty(len(sample_info_indices), dtype=object)
        visible_instance_set = set()

        for i, n in enumerate(sample_info_indices):
            sample_info = self.ixes[n]

            traslation, rotation = self._get_top_lidar_pose(sample_info)
            egopose_list[i] = [traslation, rotation]

            for annotation_token in sample_info['anns']:

                annotation = self.nusc.get('sample_annotation', annotation_token)

                # Filter out all non vehicle instances
                if 'vehicle' not in annotation['category_name']:
                    continue
                # Filter out invisible vehicles
                if self.config.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in visible_instance_set:
                    continue
                # Filter out vehicles that have not been seen in the past
                if i >= self.config.TIME_RECEPTIVE_FIELD and annotation['instance_token'] not in visible_instance_set:
                    continue
                visible_instance_set.add(annotation['instance_token'])

                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                instance_id = instance_map[annotation['instance_token']]
                instance_attribute = int(annotation['visibility_token'])

                if annotation['instance_token'] not in instance_dict:
                    # For the first occurrence of an instance
                    instance_dict[annotation['instance_token']] = {
                        'timestep': [i],
                        'translation': [annotation['translation']],
                        'rotation': [annotation['rotation']],
                        'size': annotation['size'],
                        'instance_id': instance_id,
                        'attribute_label': [instance_attribute],
                    }
                else:
                    # For the instance that have appeared before
                    instance_dict[annotation['instance_token']]['timestep'].append(i)
                    instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                    instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                    instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map, instance_dict, egopose_list, visible_instance_set

    def get_future_egomotion(self, sample_info_indices) -> np.ndarray:
        """
        Obtain the egomotion in the corresponding sequence.

        Args:
            sample_info_indices (list): list of indices of the samples in the sequence.
        Returns:
            future_egomotion (np.ndarray): array with egomotion of the sequence.
        """

        future_egomotion = np.empty((len(sample_info_indices),6))

        for i, n in enumerate(sample_info_indices):
            sample_info = self.ixes[n]

            rec_t0 = sample_info

            future_ego = np.eye(4, dtype=float)

            if n < len(self.ixes) - 1:
                rec_t1 = self.ixes[n + 1]

                if rec_t0['scene_token'] == rec_t1['scene_token']:
                    egopose_t0 = self.nusc.get(
                        'ego_pose', self.nusc.get('sample_data',
                                                  rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                    )
                    egopose_t1 = self.nusc.get(
                        'ego_pose', self.nusc.get('sample_data',
                                                  rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                    )
                    egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                    egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                    future_ego = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                    future_ego[3,:3] = 0.0
                    future_ego[3,3] = 1.0


            future_ego = torch.from_numpy(future_ego)
            future_ego = mat2pose_vec(future_ego).unsqueeze(0)
            future_egomotion[i,:] = future_ego
            
        return torch.from_numpy(future_egomotion).float()


    def get_label(self, instance_dict, egopose_list):
        """
        Generate labels for semantic segmentation, instance segmentation, z position,
        attribute from the raw data of nuScenes.
        """

        visible_instance_set = set()

        segmentation = np.zeros((self.sequence_length,self.bev_dimension[0],
                                 self.bev_dimension[1]))
        instance = np.zeros((self.sequence_length,self.bev_dimension[0],
                             self.bev_dimension[1]))
        z_position = np.zeros((self.sequence_length,self.bev_dimension[0],
                               self.bev_dimension[1]))
        attribute_label = np.zeros((self.sequence_length,self.bev_dimension[0],
                                    self.bev_dimension[1]))

        for i in range(self.sequence_length):
            timestep = i

            for instance_token, instance_annotation in instance_dict.items():
                if timestep not in instance_annotation['timestep']:
                    continue
                pointer = instance_annotation['timestep'].index(timestep)
                annotation = {
                    'translation': instance_annotation['translation'][pointer],
                    'rotation': instance_annotation['rotation'][pointer],
                    'size': instance_annotation['size'],
                }
                
                # Aply BEV rotation for augmentation here
                
                # Obtain the bounding box polygon of the instance.
                present_ego_translation, present_ego_rotation = egopose_list[self.config.TIME_RECEPTIVE_FIELD - 1]

                box = Box(
                    annotation['translation'], annotation['size'], Quaternion(annotation['rotation'])
                )
                box.translate(present_ego_translation)
                box.rotate(present_ego_rotation)
                pts = box.bottom_corners()[:2].T

                if self.config.LIFT.X_BOUND[0] <= pts.min(axis=0)[0] and pts.max(axis=0)[0] <= self.config.LIFT.X_BOUND[1] \
                    and self.config.LIFT.Y_BOUND[0] <= pts.min(axis=0)[1] and pts.max(axis=0)[1] <= self.config.LIFT.Y_BOUND[1]:
                    
                    pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    z = box.bottom_corners()[2, 0]
                    poly_region = pts
                else:
                    z = None
                    poly_region = None
                
                # poly_region, z = self._get_poly_region_in_image(annotation, egopose_list[self.config.TIME_RECEPTIVE_FIELD - 1]) 
                if isinstance(poly_region, np.ndarray):
                    if i >= self.config.TIME_RECEPTIVE_FIELD and instance_token not in visible_instance_set:
                        continue
                    visible_instance_set.add(instance_token)

                    cv2.fillPoly(instance[i], [poly_region], instance_annotation['instance_id'])
                    cv2.fillPoly(segmentation[i], [poly_region], 1.0)
                    cv2.fillPoly(z_position[i], [poly_region], z)
                    cv2.fillPoly(attribute_label[i], [poly_region], instance_annotation['attribute_label'][pointer]) 
        

        segmentation = torch.from_numpy(np.expand_dims(segmentation,1)).long()
        instance = torch.from_numpy(instance).long()
        z_position = torch.from_numpy(np.expand_dims(z_position,1)).float()
        attribute_label = torch.from_numpy(np.expand_dims(attribute_label,1)).long()

        return segmentation, instance, z_position, attribute_label   

    @staticmethod
    def generate_flow(flow, instance_img, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on
        instance segmentation.
        """        
        _, h, w = instance_img.shape
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float),
                              torch.arange(w, dtype=torch.float), indexing='ij')
        grid = torch.stack((x, y), dim=0)

        # Set the first frame
        instance_mask = (instance_img[0] == instance_id)
        flow[0, 1, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[0, 0, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (instance_img[timestep] == instance_id)
            prev_instance_mask = (instance_img[timestep-1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            # Centripetal backward flow is defined as displacement vector from each foreground pixel at time t to the object center of the associated instance identity at time t−1
            flow[timestep, 1, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
            flow[timestep, 0, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        return flow
    
    def get_flow_label(self, instance_img, instance_dict,
                       instance_map, ignore_index=255):
        """
        Generate the global map of the flow ground truth.
        """
        seq_len, h, w = instance_img.shape
        flow = ignore_index * torch.ones(seq_len, 2, h, w)

        for token, instance in instance_dict.items():
            flow = self.generate_flow(flow, instance_img, instance, instance_map[token])
        return flow

    def __getitem__(self, index) -> Any:
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            data: a dictionary containing the following keys.
                images: torch.Tensor<float> (N, 3, H, W) or (T, N, 3, H, W)
                    normalized images with T = config.n_frames and N = config.n_cameras.
        """
        # Parameters.
        T = self.sequence_length
        N = self.config.DATASET.N_CAMERAS
        H = self.config.IMAGE.FINAL_DIM[0]
        W = self.config.IMAGE.FINAL_DIM[1]

        # Data.
        data = {}

        # for counter, index_t in enumerate(self.indices[index]):
        #     rec = 

        # Warning: the original implementation uses lists to store the data, resulting in a 
        # extra dimension when indexing one element (POSSIBLE CAUSE OF ERRORS).

        sample_info_indices = self.indices[index].tolist()
        data["image"], data["intrinsics"], data["extrinsics"], data["lidar_2_sensor"], orig_img = self.get_input_data(sample_info_indices)

        if orig_img is not None and self.return_orig_images:
            data["orig_image"] = orig_img

        # Record instance information.
        instance_map, instance_dict, egopose_list, visible_instance_set = self.record_instance(sample_info_indices)

        # Obtain future egomotion.
        data["future_egomotion"] = self.get_future_egomotion(sample_info_indices)

        # Refine the generated instance polygons    
        for token in instance_dict.keys():
            instance_dict[token] = self.refine_instance_poly(instance_dict[token])

        # Returns tensors with:
        #   - segmentation: (T, H, W)
        #   - instance: (T, H, W)
        #   - z_position: (T, H, W)
        #   - attribute: (T, H, W)
                    
        data['segmentation'], data['instance'], data['z_position'], data['attribute'] = self.get_label(instance_dict, egopose_list)
        

        data['flow'] = self.get_flow_label(instance_img=data['instance'],
                                           instance_dict=instance_dict,
                                           instance_map=instance_map,
                                           ignore_index=self.config.DATASET.IGNORE_INDEX)
        

        # Generate the ground truth of centerness and offset
        instance_centerness, instance_offset = convert_instance_mask_to_center_and_offset_label(
            data['instance'], num_instances=len(instance_map),
            ignore_index=self.config.DATASET.IGNORE_INDEX,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['num_instances'] = torch.tensor([len(instance_map)])

        return data
    
    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise.
        """
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.sequence_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer-1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
                pointer += 1
                continue
            
            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
                instance['translation'][pointer] = instance['translation'][pointer-1]
                instance['rotation'][pointer] = instance['rotation'][pointer-1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
            pointer += 1
        
        return instance


    def _get_poly_region_in_image(self, instance_annotation, present_egopose):
        """
        Obtain the bounding box polygon of the instance.
        """
        present_ego_translation, present_ego_rotation = present_egopose

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)
        pts = box.bottom_corners()[:2].T

        if self.config.LIFT.X_BOUND[0] <= pts.min(axis=0)[0] and pts.max(axis=0)[0] <= self.config.LIFT.X_BOUND[1] \
            and self.config.LIFT.Y_BOUND[0] <= pts.min(axis=0)[1] and pts.max(axis=0)[1] <= self.config.LIFT.Y_BOUND[1]:
            
            pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            z = box.bottom_corners()[2, 0]
            return pts, z
        else:
            return None, None


    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments.
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True
    


# dataset = NuscenesDataset(config)
# dataloader = DataLoader(
#     dataset, batch_size=16, shuffle=True, num_workers=0,
#     drop_last=True
# )

# import psutil
# from tqdm.auto import tqdm
# pbar = tqdm(dataloader)

# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
#     record_shapes=True,
#     with_stack=True,
#     profile_memory=True,
# ) as prof:
#     for batch in pbar:
#         memory = psutil.virtual_memory()[2]
#         pbar.set_description(f"Memory usage: {memory:.2f}%")

# import gc
# gc.collect()