import mne
import numpy as np
import os


def save_eeg_segments(edf_path, output_folder, tmin, tmax, segment_duration=2):
    # 加载原始 EEG 数据
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # 应用滤波器
    raw.filter(0.53, 40, fir_design='firwin')

    # 裁剪时间范围
    raw_crop = raw.copy().crop(tmin=tmin, tmax=tmax)

    # 将裁剪后的数据转换为 DataFrame，并只取前 22 通道
    raw_crop = raw_crop.to_data_frame()
    _, ndraw, _ = np.split(raw_crop, (0, 22,), axis=1)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 计算切割的次数
    total_duration = tmax - tmin
    num_segments = int(np.floor(total_duration / segment_duration))
    sfreq = raw.info['sfreq']  # 采样频率

    # 分段保存为 CSV
    for i in range(num_segments):
        segment_start_index = int(i * segment_duration * sfreq)  # 计算开始的索引
        segment_end_index = int((i + 1) * segment_duration * sfreq)  # 计算结束的索引

        # 对数据进行再次裁剪以获取每个 2 秒的数据段
        segment = ndraw.iloc[segment_start_index:segment_end_index]

        # 文件名格式化
        start_seconds = tmin + i * segment_duration
        end_seconds = start_seconds + segment_duration
        start_str = f"{int(start_seconds // 3600):02}:{int((start_seconds % 3600) // 60):02}:{int(start_seconds % 60):02}"
        end_str = f"{int(end_seconds // 3600):02}:{int((end_seconds % 3600) // 60):02}:{int(end_seconds % 60):02}"
        filename = f"{start_str}_{end_str}.csv"

        # 保存文件
        segment.to_csv(os.path.join(output_folder, filename), index=False)



# 使用示例
# edf_path = '../rawdata/4600/4600.edf'
# output_folder = './eeg_segments/4600'
# save_eeg_segments(edf_path, output_folder, 11855, 11963)

edf_path = '../rawdata/4613/4613.edf'
output_folder = './eeg_segments/4613'
save_eeg_segments(edf_path, output_folder, 10330, 10420)

