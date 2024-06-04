import cv2
import os
from datetime import timedelta


def parse_time(time_str):
    # Converts a HH:MM:SS time string into total seconds
    h, m, s = map(int, time_str.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s).total_seconds()

def format_time(seconds):
    # Converts time in seconds to a HH:MM:SS format
    return str(timedelta(seconds=int(seconds)))

def process_video(video_path, output_folder, start_time, end_time, frames_per_segment=50, half="right"):
    start_seconds = parse_time(start_time)
    end_seconds = parse_time(end_time)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video frame rate

    # Jump to the frame corresponding to the start time
    start_frame = int(start_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Calculate maximum number of frames
    max_frames = int((end_seconds - start_seconds) * fps)

    frame_count = 0
    segment_frame_count = 0
    out = None  # Initialize VideoWriter outside the loop

    while cap.isOpened() and frame_count < max_frames:
        if max_frames - frame_count < frames_per_segment:
            break  # Skip if remaining frames are less than required per segment

        ret, frame = cap.read()
        if not ret:
            break  # Stop processing if no frame is fetched

        width = frame.shape[1]
        if half == "right":
            frame = frame[:, width // 2:]  # Process only the right half of the video

        if segment_frame_count == 0:
            # Calculate segment start frame for naming
            segment_start_frame = frame_count + start_frame
            segment_filename = f"{format_time(segment_start_frame / fps)}.mp4"
            segment_filepath = os.path.join(output_folder, segment_filename)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video Codec
            out = cv2.VideoWriter(segment_filepath, fourcc, fps, (frame.shape[1], frame.shape[0]))

        out.write(frame)

        frame_count += 1
        segment_frame_count += 1

        # If segment has reached desired frame count, save and close
        if segment_frame_count == frames_per_segment:
            out.release()  # Save and close the file
            segment_frame_count = 0  # Reset the frame counter for the next segment

    cap.release()
    # If the loop exits and there is an open video writer, close it
    if out and segment_frame_count != 0:
        out.release()

# Example usage
file_p = '../rawdata/4613/Patient7_BED22-1_t7/nrva000b.avi'
output_folder = '../Video_CNNLSTM/video_segments/4613'
start_time = "00:07:24"
end_time = "00:11:46"
process_video(file_p, output_folder, start_time, end_time, half="right")


file_p = '../rawdata/4581/Patient7_BED22-1_t6/nrva0007.avi'
output_folder = '../Video_CNNLSTM/video_segments/4581_1'
start_time = "00:08:41"
end_time = "00:12:43"
process_video(file_p, output_folder, start_time, end_time, half="right")
#
#
file_p = '../rawdata/4581/Patient7_BED22-1_t6/nrva000b.avi'
output_folder = '../Video_CNNLSTM/video_segments/4581_2'
start_time = "00:12:05"
end_time = "00:14:05"
process_video(file_p, output_folder, start_time, end_time, half="right")


#
file_p = '../rawdata/4600/Patient7_BED22-1_t2/nrva000d.avi'
output_folder = '../Video_CNNLSTM/video_segments/4600'
start_time = "00:02:19"
end_time = "00:04:22"
process_video(file_p, output_folder, start_time, end_time, half="right")

file_p = '../rawdata/4563/Patient7_BED22-1_t6/nrva0009.avi'
output_folder = '../Video_CNNLSTM/video_segments/4563'
start_time = "00:08:00"
end_time = "00:11:20"
process_video(file_p, output_folder, start_time, end_time, half="right")
