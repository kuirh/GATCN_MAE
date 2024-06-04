import cv2
import mediapipe as mp
import csv
import os
from datetime import timedelta

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def parse_time(time_str):
    # Converts a HH:MM:SS time string into total seconds
    h, m, s = map(int, time_str.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s).total_seconds()


def format_time(seconds):
    # Converts time in seconds to a HH:MM:SS format
    return str(timedelta(seconds=int(seconds)))


def process_video(video_path, output_folder, start_time, end_time, segment_length=3, half="left"):
    start_seconds = parse_time(start_time)
    end_seconds = parse_time(end_time)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frames_per_segment = int(fps * segment_length)  # 计算每个3秒段的帧数

    # 跳转到起始秒对应的帧
    start_frame = int(start_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 计算最大帧数
    max_frames = int((end_seconds - start_seconds) * fps)

    pose = mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.4)

    frame_count = 0
    segment_frame_count = 0
    person_detected = False
    segment_landmarks = []
    first_frame = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        width = frame.shape[1]
        if half == "left":
            frame = frame[:, :width // 2]  # 只处理视频的左半部分
        elif half == "right":
            frame = frame[:, width // 2:]  # 只处理视频的右半部分

        if segment_frame_count == frames_per_segment:
            if person_detected:
                segment_end = frame_count + start_frame - 1
                if len(segment_landmarks) >= frames_per_segment:  # Ensure segment is at least 3 seconds
                    save_to_csv_and_draw(segment_start, segment_end, segment_landmarks, output_folder, fps, first_frame,
                                         results.pose_landmarks)
                segment_landmarks = []
                person_detected = False
                first_frame = None
            segment_frame_count = 0  # 重置帧计数器

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 检测到人物关键点
        if results.pose_landmarks:
            if not person_detected:
                # 新段落开始
                person_detected = True
                segment_start = frame_count + start_frame
                first_frame = frame.copy()  # Store the first frame of the segment

            # 记录关键点数据
            landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in
                         results.pose_landmarks.landmark]
            segment_landmarks.append(landmarks)

        frame_count += 1
        segment_frame_count += 1

    # Handle the last segment
    if person_detected and len(segment_landmarks) >= frames_per_segment:
        segment_end = frame_count + start_frame - 1
        save_to_csv_and_draw(segment_start, segment_end, segment_landmarks, output_folder, fps, first_frame,
                             results.pose_landmarks)

    cap.release()
    pose.close()


def save_to_csv_and_draw(start_frame, end_frame, landmarks_data, output_folder, fps, first_frame, pose_landmarks):
    start_time_str = format_time(start_frame / fps)
    end_time_str = format_time(end_frame / fps)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # This will create the directory if it does not exist

    filename = f"{start_time_str}_{end_time_str}.csv"
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z', 'visibility'])
        for frame_index, frame_data in enumerate(landmarks_data):
            for lm_index, lm in enumerate(frame_data):
                writer.writerow([frame_index + start_frame, lm_index, lm['x'], lm['y'], lm['z'], lm['visibility']])

    # Draw landmarks on the first frame of the segment
    if first_frame is not None:
        mp_drawing.draw_landmarks(first_frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        img_filename = f"{start_time_str}_{end_time_str}.jpg"
        img_filepath = os.path.join(output_folder, img_filename)
        cv2.imwrite(img_filepath, first_frame)




#
# file_p = 'rawdata/4563/Patient7_BED22-1_t6/nrva0009.avi'
# output_folder = 'output_folder/4563'
# start_time = "00:08:00"
# end_time = "00:11:20"
# process_video(file_p, output_folder, start_time, end_time, half="right", segment_length=2)



# file_p = 'rawdata/4581/Patient7_BED22-1_t6/nrva0007.avi'
# output_folder = 'output_folder/4581_1'
# start_time = "00:08:41"
# end_time = "00:12:43"
# process_video(file_p, output_folder, start_time, end_time, half="right", segment_length=2)
#
#
# file_p = 'rawdata/4581/Patient7_BED22-1_t6/nrva000b.avi'
# output_folder = 'output_folder/4581_2'
# start_time = "00:12:05"
# end_time = "00:14:05"
# process_video(file_p, output_folder, start_time, end_time, half="right", segment_length=2)


#
# file_p = 'rawdata/4600/Patient7_BED22-1_t2/nrva000d.avi'
# output_folder = 'output_folder/4600'
# start_time = "00:02:19"
# end_time = "00:04:22"
# process_video(file_p, output_folder, start_time, end_time, half="right", segment_length=2)


file_p = 'rawdata/4613/Patient7_BED22-1_t7/nrva000b.avi'
output_folder = 'output_folder/4613'
start_time = "00:07:24"
end_time = "00:11:46"
process_video(file_p, output_folder, start_time, end_time, half="right", segment_length=2)

