import mne
import numpy as np
import os

def save_regular_intervals(edf_path, output_folder, exclude_intervals, interval=600, duration=60, segment_duration=2):
    # 加载原始 EEG 数据
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # 应用滤波器
    raw.filter(0.53, 40, fir_design='firwin')

    # 获取数据的总时长
    total_duration = raw.times[-1]
    sfreq = raw.info['sfreq']  # 采样频率

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 初始化当前时间，开始采样过程
    current_time = 0

    while current_time < total_duration:
        # 检查当前区间是否与任何排除区间重叠
        if any(start <= current_time + duration <= end or current_time >= start and current_time <= end for start, end in exclude_intervals):
            # 跳过到最近的排除区间结束之后
            current_time = min([end for start, end in exclude_intervals if current_time <= end], default=current_time + interval)
            continue

        # 确定采样的结束时间
        end_time = min(current_time + duration, total_duration)

        # 裁剪当前的采样区间
        raw_sample = raw.copy().crop(tmin=current_time, tmax=end_time)

        # 将裁剪后的数据转换为 DataFrame，并只取前 22 通道
        raw_sample = raw_sample.to_data_frame()
        _, ndraw, _ = np.split(raw_sample, (0, 22,), axis=1)

        # 保存数据到 CSV 文件，每个文件包含 2 秒数据
        for i in range(0, int(duration / segment_duration)):
            segment_start_index = int(i * segment_duration * sfreq)
            segment_end_index = int((i + 1) * segment_duration * sfreq)

            # 获取片段
            segment = ndraw.iloc[segment_start_index:segment_end_index]

            # 文件名格式化
            start_seconds = current_time + i * segment_duration
            end_seconds = start_seconds + segment_duration
            start_str = f"{int(start_seconds // 3600):02}:{int((start_seconds % 3600) // 60):02}:{int(start_seconds % 60):02}"
            end_str = f"{int(end_seconds // 3600):02}:{int((end_seconds % 3600) // 60):02}:{int(end_seconds % 60):02}"
            filename = f"{start_str}_{end_str}.csv"

            # 保存文件
            segment.to_csv(os.path.join(output_folder, filename), index=False)

        # 移动到下一个采样区间
        current_time += interval

# 使用示例
edf_path = '../rawdata/4613/4613.edf'
output_folder = './eeg_segments/4613_normal'
exclude_intervals = [(10330, 10420)]  # 示例中包含两个排除区间
save_regular_intervals(edf_path, output_folder, exclude_intervals)
