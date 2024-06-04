import cv2
import mediapipe as mp
import csv
import os
from datetime import timedelta
import re

from moviepy.video.io.VideoFileClip import VideoFileClip

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def parse_time(time_str):
    # Converts a HH:MM:SS time string into total seconds
    h, m, s = map(int, time_str.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s).total_seconds()


def format_time(seconds):
    # Converts time in seconds to a HH:MM:SS format
    return str(timedelta(seconds=int(seconds)))

def natural_sort_key(s):
    """
    Assist in sorting strings that may contain hexadecimal numbers.
    """
    # This function identifies parts of the filename that are numbers, interprets them as hexadecimal,
    # and converts them to a decimal integer for correct natural sorting.
    return [int(text, 16) if text.isdigit() or is_hex(text) else text.lower() for text in re.split('([0-9a-fA-F]+)', s)]

def is_hex(s):
    """
    Check if a string represents a valid hexadecimal number.
    """
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

def list_and_sort_videos(directory):
    """
    List and sort video files in a directory using natural order considering hexadecimal values.
    """
    video_files = [os.path.join(root, file)
                   for root, _, files in os.walk(directory)
                   for file in files if file.endswith('.avi')]
    video_files.sort(key=natural_sort_key)
    return video_files

def process_videos(directory, output_folder, excluded_file, excluded_start, excluded_end):
    all_videos = list_and_sort_videos(directory)
    excluded_start_sec = parse_time(excluded_start)
    excluded_end_sec = parse_time(excluded_end)

    for video_path in all_videos:
        video_clip = VideoFileClip(video_path)
        total_duration = int(video_clip.duration)

        # Process clips every 5 minutes
        for start in range(0, total_duration, 300):  # 300 seconds = 5 minutes
            end = start + 60  # 60 seconds = 1 minute
            if end > total_duration:
                break  # If the end of the clip exceeds the video duration, skip it

            # Check if this clip overlaps with the excluded time range
            if video_path == excluded_file and ((excluded_start_sec <= start < excluded_end_sec) or (excluded_start_sec < end <= excluded_end_sec)):
                continue  # Skip this clip as it overlaps with the excluded range

            # Prepare parameters for processing
            clip_start_time = format_time(start)
            clip_end_time = format_time(end)
            process_video(video_path, output_folder, clip_start_time, clip_end_time, half="right", segment_length=2)
def process_video(video_path, output_folder, start_time, end_time, segment_length=2, half="right"):
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

    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

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
                                         results.pose_landmarks, video_path)
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
                             results.pose_landmarks, video_path)

    cap.release()
    pose.close()


def save_to_csv_and_draw(start_frame, end_frame, landmarks_data, output_folder, fps, first_frame, pose_landmarks,
                         video_path):
    def format_time_with_offset(seconds, suffix):
        """ Calculate new time with offset based on file suffix. """
        additional_minutes = int(suffix,
                                 16) * 15  # Suffix converted from hex to decimal, then multiplied by 15 minutes.
        time_with_offset = timedelta(seconds=seconds) + timedelta(minutes=additional_minutes)
        return f"{time_with_offset}"

    # Extract suffix from filename
    file_suffix = os.path.basename(video_path).split('.')[0][-1]  # Assumes format 'xxxxa.avi' where 'a' is the suffix.

    start_time_str = format_time_with_offset(start_frame / fps, file_suffix)
    end_time_str = format_time_with_offset(end_frame / fps, file_suffix)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Ensure the directory exists

    filename = f"{start_time_str}_{end_time_str}.csv"
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z', 'visibility'])
        for frame_index, frame_data in enumerate(landmarks_data):
            for lm_index, lm in enumerate(frame_data):
                writer.writerow([frame_index + start_frame, lm_index, lm['x'], lm['y'], lm['z'], lm['visibility']])

    # Optionally draw landmarks and save the image
    if first_frame is not None:
        mp_drawing.draw_landmarks(first_frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        img_filename = f"{start_time_str}_{end_time_str}.jpg"
        img_filepath = os.path.join(output_folder, img_filename)
        cv2.imwrite(img_filepath, first_frame)



process_videos('rawdata/4613/Patient7_BED22-1_t7', 'output_folder/4613_norm', 'rawdata/4613/Patient7_BED22-1_t7/nrva000d.avi', "00:07:24", "00:11:46")