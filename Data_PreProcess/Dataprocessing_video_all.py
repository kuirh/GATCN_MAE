import cv2
import os
from datetime import timedelta
import re

from moviepy.editor import VideoFileClip

import cv2
import os
from datetime import timedelta
import re

from moviepy.editor import VideoFileClip


def parse_time(time_str):
    # Converts a HH:MM:SS time string into total seconds
    h, m, s = map(int, time_str.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s).total_seconds()


def format_time(seconds):
    # Converts time in seconds to a HH:MM:SS format
    return str(timedelta(seconds=int(seconds)))


def natural_sort_key(s):
    # Assist in sorting strings that may contain hexadecimal numbers
    return [int(text, 16) if text.isdigit() or is_hex(text) else text.lower() for text in re.split('([0-9a-fA-F]+)', s)]


def is_hex(s):
    # Check if a string represents a valid hexadecimal number
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


def list_and_sort_videos(directory):
    # List and sort video files in a directory using natural order considering hexadecimal values
    video_files = [os.path.join(root, file)
                   for root, _, files in os.walk(directory)
                   for file in files if file.endswith('.avi')]
    video_files.sort(key=natural_sort_key)
    return video_files


def format_time_with_offset(seconds, suffix):
    # Calculate new time with offset based on file suffix
    additional_minutes = int(suffix, 16) * 15  # Suffix converted from hex to decimal, then multiplied by 15 minutes
    time_with_offset = timedelta(seconds=seconds) + timedelta(minutes=additional_minutes)
    return f"{time_with_offset}"


def process_videos(directory, output_folder, excluded_file, excluded_start, excluded_end):
    all_videos = list_and_sort_videos(directory)
    excluded_start_sec = parse_time(excluded_start)
    excluded_end_sec = parse_time(excluded_end)

    for video_path in all_videos:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

        excluded_start_frame = int(excluded_start_sec * fps) if video_path == excluded_file else -1
        excluded_end_frame = int(excluded_end_sec * fps) if video_path == excluded_file else -1

        # Process clips every 5 minutes
        for start in range(0, total_duration, 300):  # 300 seconds = 5 minutes
            end = start + 60  # 60 seconds = 1 minute
            if end > total_duration:
                break  # If the end of the clip exceeds the video duration, skip it

            # Check if this clip overlaps with the excluded time range
            if video_path == excluded_file and ((excluded_start_frame <= start * fps < excluded_end_frame) or (
                    excluded_start_frame < end * fps <= excluded_end_frame)):
                continue  # Skip this clip as it overlaps with the excluded range

            # Prepare parameters for processing
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames_per_segment = 50
            segment_length = frames_per_segment / fps
            current_frame = start_frame
            while current_frame + frames_per_segment <= end_frame:
                # Define output filename
                file_suffix = os.path.basename(video_path).split('.')[-2][-1]
                start_time_str = format_time_with_offset(current_frame / fps, file_suffix)
                end_time_str = format_time_with_offset((current_frame + frames_per_segment) / fps, file_suffix)
                video_filename = f"{start_time_str}_{end_time_str}.mp4"
                video_filepath = os.path.join(output_folder, video_filename)

                # Create VideoWriter object
                out = cv2.VideoWriter(video_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                # Write frames
                for _ in range(frames_per_segment):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                out.release()
                current_frame += frames_per_segment

        cap.release()


# file_p = '../rawdata/4581/Patient7_BED22-1_t6'
# output_folder = '../Video_CNNLSTM/video_segments/4581_norm'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# process_videos(file_p, output_folder, '../rawdata/4563/Patient7_BED22-1_t6/nrva0009.avi', "00:08:00", "00:11:20")


file_p = '../rawdata/4600/Patient7_BED22-1_t2'
output_folder = '../Video_CNNLSTM/video_segments/4600_norm'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
process_videos(file_p, output_folder, '../rawdata/4600/Patient7_BED22-1_t2/nrva000d.avi', "00:02:19", "00:04:22")

file_p = '../rawdata/4613/Patient7_BED22-1_t7'
output_folder = '../Video_CNNLSTM/video_segments/4613_norm'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
process_videos(file_p, output_folder, '../rawdata/4613/Patient7_BED22-1_t7/nrva000b.avi', "00:07:24", "00:11:46")

file_p = '../rawdata/4563/Patient7_BED22-1_t6'
output_folder = '../Video_CNNLSTM/video_segments/4563_norm'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
process_videos(file_p, output_folder, '../rawdata/4563/Patient7_BED22-1_t6/nrva0009.avi', "00:08:00", "00:11:20")
