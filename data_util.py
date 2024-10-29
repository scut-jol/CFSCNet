import torch
import torch.nn.init as init
from scipy import signal


def high_pass_filter(data, sampling_freq, cutoff_freq):
    b, a = signal.butter(4, cutoff_freq / (sampling_freq / 2), 'high')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def padding(signal, win_sample, hop_sample):
    total_length = signal.shape[1]
    num_windows = max((total_length - win_sample) // hop_sample + 1, 0)
    new_total_length = num_windows * hop_sample + win_sample
    num_zeros_to_pad = new_total_length - total_length
    if num_zeros_to_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, num_zeros_to_pad, 0, 0), mode='constant', value=0)
    return signal


def frame(signal, win_sample, hop_sample):
    num_channels = signal.shape[0]
    signal_len = signal.shape[1]
    num_frames = (signal_len - win_sample) // hop_sample + 1
    frames = torch.zeros((num_frames, num_channels, win_sample))

    for i in range(num_frames):
        start = i * hop_sample
        end = start + win_sample
        frames[i, :, :] = signal[:, start:end]

    return frames


def initialize_weights_normal(m):
    if isinstance(m, torch.nn.Linear):
        init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.normal_(m.bias, mean=0, std=0.01)


def predict_calculate_overlap(model, root_path, test_loader, pth_path, device):
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    overlap_list = []
    with torch.no_grad():
        for imu, label in test_loader:
            imu = imu.to(device)
            label = label.to(device)
            output = model(imu)
            events = Advanced_post_processing(output)
            label_events = Advanced_post_processing(label)
            for event in label_events:
                overlap_rate = 0
                for pre_event in events:
                    overlap_rate = max(overlap_rate, calculate_jaccard_index(pre_event[0], pre_event[1], event[0], event[1]))
                overlap_list.append(overlap_rate)
    return overlap_list


def Advanced_post_processing(pred, hop_duration=2, bin_duration=0.2):
    events = []
    swallow_flag = False
    for i in range(pred.shape[0]):
        pred_win = pred[i, :]
        for j in range(pred_win.shape[0]):
            bin_predicted = pred_win[j].item()
            if swallow_flag:
                if bin_predicted >= 0.4:
                    start_point = i * hop_duration + j * bin_duration
                    end_point = i * hop_duration + (j + 1) * bin_duration
                    events.append([start_point, end_point])
                else:
                    swallow_flag = False
            else:
                if bin_predicted >= 0.6:
                    swallow_flag = True
                    start_point = i * hop_duration + j * bin_duration
                    end_point = i * hop_duration + (j + 1) * bin_duration
                    events.append([start_point, end_point])
    events.sort(key=lambda x: x[0])
    max_silence = 0.4
    min_dur = 0.3

    merge_silence_events(events, max_silence)
    del_min_duration_events(events, min_dur)

    for i in range(len(events)):
        events[i][0] = round(events[i][0], 3)
        events[i][1] = round(events[i][1], 3)

    events.sort(key=lambda x: x[0])
    return events


def merge_silence_events(events, min_silence):
    count = 0
    while count < len(events) - 1:
        if (events[count][1] >= events[count + 1][0]) or (events[count + 1][0] - events[count][1] <= min_silence):
            events[count][1] = max(events[count + 1][1], events[count][1])
            del events[count + 1]
        else:
            count += 1


def del_min_duration_events(events, min_duration):
    count = 0
    while count < len(events) - 1:
        if events[count][1] - events[count][0] < min_duration:
            del events[count]
        else:
            count += 1
    if len(events) > 0 and events[count][1] - events[count][0] < min_duration:
        del events[count]


def calculate_jaccard_index(a, b, x, y):
    """
    计算两个时间段的 Jaccard 相似系数。

    参数:
    a, b: 第一个时间段的开始和结束时间
    x, y: 第二个时间段的开始和结束时间

    返回:
    Jaccard 相似系数
    """
    # 确保时间段是有效的
    if a > b or x > y:
        raise ValueError("Invalid time periods")

    # 计算交集的开始和结束时间
    intersection_start = max(a, x)
    intersection_end = min(b, y)

    # 如果没有交集
    if intersection_start >= intersection_end:
        return 0.0

    # 计算交集和并集的持续时间
    intersection_duration = intersection_end - intersection_start
    union_duration = (b - a) + (y - x) - intersection_duration

    # 计算 Jaccard 相似系数
    jaccard_index = intersection_duration / union_duration

    return jaccard_index


def merge_rows_and_remove_overlap(input):
    two_d_list = input.tolist()
    if not two_d_list:
        return []

    merged_list = []

    for i in range(len(two_d_list) - 1):
        # Current row
        current_row = two_d_list[i]
        # Next row
        next_row = two_d_list[i + 1]

        # Handle the overlapping part
        for j in range(-5, 0):
            next_row[j + 5] = (current_row[j] + next_row[j + 5]) / 2

        # Append the non-overlapping part of the current row
        merged_list.extend(current_row[:-5])

    # Append the last row in full
    merged_list.extend(two_d_list[-1])

    return merged_list


def post_process(input_data, rebuild=True):
    sequence = merge_rows_and_remove_overlap(input_data)
    high_threshold = 0.6
    low_threshold = 0.4
    current_state = "non_swallowing"
    output_sequence = []
    mid_threshold = 0.5
    for i, point in enumerate(sequence):
        if current_state == "non_swallowing":
            if point >= high_threshold:
                current_state = "swallowing"
            elif point >= mid_threshold:
                current_state = "intermediate"
        elif current_state == "intermediate":
            if point < low_threshold:
                current_state = "non_swallowing"
            elif point >= high_threshold:
                current_state = "swallowing"
        elif current_state == "swallowing":
            if point < low_threshold:
                current_state = "non_swallowing"
            elif point < mid_threshold:
                current_state = "intermediate"

        if current_state == "swallowing":
            output_sequence.append(1)
        elif current_state == "intermediate":
            output_sequence.append(point)
        else:
            output_sequence.append(0)
    output_sequence = process_intermediate(output_sequence)
    delete_short_event(output_sequence)
    if rebuild:
        return rebuild_label(output_sequence)
    else:
        return output_sequence


def rebuild_label(sequnce):
    ouput = []
    for i in range(0, len(sequnce) - 14, 10):
        ouput.append(sequnce[i:i + 15])
    return torch.tensor(ouput)


def delete_short_event(data):
    continue_one = process_type(data=data, distinct_val=0, continue_val=1)
    continue_zero = process_type(data=continue_one, distinct_val=1, continue_val=0)
    return continue_zero


def process_type(data, distinct_val=0, continue_val=1, count=2):
    n = len(data)
    i = 0
    result = data[:]
    while i < n:
        if result[i] == distinct_val:
            start = i
            while i < n and result[i] == distinct_val:
                i += 1
            end = i

            if start > 0 and end < n and result[start - 1] == continue_val and result[end] == continue_val and end - start < count + 1:
                for j in range(start, end):
                    result[j] = continue_val
        else:
            i += 1
    return result


def process_intermediate(sequence):
    result = []  # 存储处理后的结果
    i = 0

    while i < len(sequence):
        if sequence[i] == 0 or sequence[i] == 1:
            # 如果是0或1，直接添加到结果中
            result.append(sequence[i])
            i += 1
        else:
            # 如果是非0且非1的小数，开始找到该段连续的小数序列
            start = i
            while i < len(sequence) and sequence[i] != 0 and sequence[i] != 1:
                i += 1
            # 计算该连续段的小数平均值
            avg = sum(sequence[start:i]) / (i - start)
            # 根据平均值将该段修改为0或1
            fill_value = 1 if avg > 0.5 else 0
            result.extend([fill_value] * (i - start))

    return result