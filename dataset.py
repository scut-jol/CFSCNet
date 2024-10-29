
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from data_util import padding, frame, high_pass_filter


def collate_fn(data):
    return data[0][0], data[0][1]


class SwallowDataset(Dataset):
    def __init__(self, root_path, files_list, win_length, hop_length, sample_rate, batch_size) -> None:
        super().__init__()
        self.imu, self.label = self._data_process(root_path, files_list, win_length, hop_length, sample_rate)
        self.batch_size = batch_size

    def _data_process(self, root_path, files_list, win_length, hop_length, sample_rate):
        imu_frames_list, label_frames_list = [], []
        for file_name in files_list:
            file_path = os.path.join(root_path, "signals", f"{file_name}.npy")
            imu_data = np.transpose(np.load(file_path)[:, :3])
            imu_data[0] = high_pass_filter(imu_data[0], sample_rate, 2)
            imu_data[1] = high_pass_filter(imu_data[1], sample_rate, 2)
            imu_data[2] = high_pass_filter(imu_data[2], sample_rate, 2)
            imu = torch.tensor(imu_data)
            label_path = os.path.join(root_path, "labels", f"{file_name}.csv")
            label_meta = pd.read_csv(label_path)
            imu = padding(imu, int(sample_rate * win_length), int(sample_rate * hop_length))
            imu_frames = frame(imu, int(sample_rate * win_length), int(sample_rate * hop_length))
            label_frames = self._frame_cut(len(imu_frames), label_meta, win_length, hop_length, 15)
            imu_frames_list.append(imu_frames)
            label_frames_list.append(label_frames)
        imu_total = torch.cat(imu_frames_list, dim=0)
        label_total = torch.cat(label_frames_list, dim=0)
        return imu_total, label_total

    def __len__(self):
        return (self.label.shape[0] + self.batch_size - 1) // self.batch_size

    def _frame_cut(self, frame_length, label_meta, win_duration, hop_duartion, bins_num):
        labels = []
        bin_s = win_duration / bins_num
        labels = [[0] * bins_num for _ in range(frame_length)]
        for index, row in label_meta.iterrows():
            start_value = row['start'] / 60
            end_value = row['end'] / 60
            for i in range(frame_length):
                for bin in range(bins_num):
                    bin_onset = i * hop_duartion + bin * bin_s
                    bin_offset = bin_onset + bin_s
                    if bin_onset > end_value or bin_offset < start_value:
                        pass
                    else:
                        labels[i][bin] = 1.0
        return torch.tensor(labels).float()

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return self.imu[start_index:end_index, :, :], self.label[start_index:end_index, :]


class Aspiration(Dataset):
    def __init__(self, data_list, root_path) -> None:
        super().__init__()
        self.datas = data_list
        self.root_path = root_path

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        file_path = os.path.join(self.root_path, self.datas.iloc[index, 0])
        loaded_dict = torch.load(file_path)
        assert loaded_dict['audio'].shape[1] == 1 and loaded_dict['audio'].shape[2] == 48000
        assert loaded_dict['imu'].shape[1] == 3 and loaded_dict['imu'].shape[2] == 3000
        assert loaded_dict['gas'].shape[1] == 1 and loaded_dict['gas'].shape[2] == 300
        assert loaded_dict['labels'].shape[1] == 15 and loaded_dict['masks'].shape[1] == 15
        return loaded_dict
