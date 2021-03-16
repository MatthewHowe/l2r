import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
import json
import copy
import math
import os
from . import data_parser
from .utils import sort_nicely

# Normalize speed to 0-1
SPEED_FACTOR = 12.0
AUGMENT_LATERAL_STEERINGS = 6

class ILDataset(Dataset):

    def __init__(self, root_dir, dataset_name, split_name, preload_name=None):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, dataset_name, split_name)
        self.preload_name = preload_name
        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
            print(" Loading from NPY ")
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'), 
                allow_pickle=True)
            print(self.sensor_data_names)
        else:
            self.sensor_data_names, self.measurements = self._pre_load_image_folders(self.data_dir) 

        self.transform_op = transforms.Compose([
             transforms.Resize(256),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.data_dir,
                                self.sensor_data_names[idx])
        input_image = Image.open(img_path)
        image_tensor = self.transform_op(input_image)
        measurement = self.measurements[idx]

        try: next_measurement = self.measurements[idx+1]
        except IndexError: next_measurement = measurement

        direction = torch.LongTensor([measurement['directions']])
        speed = torch.Tensor([measurement['speed_module']])
        acceleration =  measurement['throttle'] - measurement['brake']
        steer = measurement['steer']
        ctrl = torch.Tensor([acceleration, steer])

        next_steer = next_measurement['steer']
        next_speed = torch.Tensor([next_measurement['speed_module']])

        sample = {'image_tensor': image_tensor, 
                'direction': direction,
                'speed': speed,
                'acceleration': acceleration,
                'steer': steer,
                'ctrl': ctrl,
                'next_steer': next_steer,
                'next_speed': next_speed}

        #return (image_tensor, speed, direction), ctrl
        return sample


    def _pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now.

        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """

        episodes_list = glob.glob(os.path.join(path, 'episode_*'))
        sort_nicely(episodes_list)
        # Do a check if the episodes list is empty
        if len(episodes_list) == 0:
            raise ValueError("There are no episodes on the training dataset folder %s" % path)

        sensor_data_names = []
        float_dicts = []

        number_of_hours_pre_loaded = 0

        # Now we do a check to try to find all the
        for episode in episodes_list:

            print('Episode ', episode)

            available_measurements_dict = data_parser.check_available_measurements(episode)

            # if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS:
            #     # The number of wanted hours achieved
            #     break

            # Get all the measurements from this episode
            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)

            if len(measurements_list) == 0:
                print("EMPTY EPISODE")
                continue

            # A simple count to keep track how many measurements were added this episode.
            count_added_measurements = 0

            for measurement in measurements_list[:-3]:

                data_point_number = measurement.split('_')[-1].split('.')[0]

                with open(measurement) as f:
                    measurement_data = json.load(f)

                # depending on the configuration file, we eliminated the kind of measurements
                # that are not going to be used for this experiment
                # We extract the interesting subset from the measurement dict

                speed = data_parser.get_speed(measurement_data)

                directions = measurement_data['directions']
                final_measurement = self._get_final_measurement(speed, measurement_data, 0,
                                                                directions,
                                                                available_measurements_dict)

                # if self.is_measurement_partof_experiment(final_measurement):
                float_dicts.append(final_measurement)
                rgb = 'CentralRGB_' + data_point_number + '.png'
                sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                count_added_measurements += 1

                # We do measurements for the left side camera
                # We convert the speed to KM/h for the augmentation

                # We extract the interesting subset from the measurement dict

                final_measurement = self._get_final_measurement(speed, measurement_data, -30.0,
                                                                directions,
                                                                available_measurements_dict)

                #if self.is_measurement_partof_experiment(final_measurement):
                float_dicts.append(final_measurement)
                rgb = 'LeftRGB_' + data_point_number + '.png'
                sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                count_added_measurements += 1

                # We do measurements augmentation for the right side cameras

                final_measurement = self._get_final_measurement(speed, measurement_data, 30.0,
                                                                directions,
                                                                available_measurements_dict)

                # if self.is_measurement_partof_experiment(final_measurement):
                float_dicts.append(final_measurement)
                rgb = 'RightRGB_' + data_point_number + '.png'
                sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                count_added_measurements += 1

            # Check how many hours were actually added

            last_data_point_number = measurements_list[-4].split('_')[-1].split('.')[0]
            number_of_hours_pre_loaded += (float(count_added_measurements / 10.0) / 3600.0)
            print(" Loaded ", number_of_hours_pre_loaded, " hours of data")


        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts
    # def is_measurement_partof_experiment(self, measurement_data):

    #     # If the measurement data is not removable is because it is part of this experiment dataa
    #     return not self._check_remove_function(measurement_data, self._remove_params)

    def _get_final_measurement(self, speed, measurement_data, angle,
                               directions, avaliable_measurements_dict):
        """
        Function to load the measurement with a certain angle and augmented direction.
        Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.

        Returns
            The final measurement dict
        """
        if angle != 0:
            measurement_augmented = self.augment_measurement(copy.copy(measurement_data), angle,
                                                             3.6 * speed,
                                                 steer_name=avaliable_measurements_dict['steer'])
        else:
            # We have to copy since it reference a file.
            measurement_augmented = copy.copy(measurement_data)

        if 'gameTimestamp' in measurement_augmented:
            time_stamp = measurement_augmented['gameTimestamp']
        else:
            time_stamp = measurement_augmented['elapsed_seconds']

        final_measurement = {}
        # We go for every available measurement, previously tested
        # and update for the measurements vec that is used on the training.
        for measurement, name_in_dataset in avaliable_measurements_dict.items():
            # This is mapping the name of measurement in the target dataset
            final_measurement.update({measurement: measurement_augmented[name_in_dataset]})

        # Add now the measurements that actually need some kind of processing
        final_measurement.update({'speed_module': speed /SPEED_FACTOR})
        final_measurement.update({'directions': directions})
        final_measurement.update({'game_time': time_stamp})

        return final_measurement

    def augment_directions(self, directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))

        # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer

    def augment_measurement(self, measurements, angle, speed, steer_name='steer'):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, measurements[steer_name],
                                          speed)
        measurements[steer_name] = new_steer
        return measurements

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]


    """
        Methods to interact with the dataset attributes that are used for training.
    """

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    def extract_intentions(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)
