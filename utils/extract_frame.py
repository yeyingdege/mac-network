import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from decord import VideoReader
from PIL import Image


def find_video_path_by_id(video_id, video_folder):
    # List all subfolders in video_folder
    video_folder = Path(video_folder)
    subfolders = [f.name for f in video_folder.iterdir() if f.is_dir()]

    # Iteratre subfolders, and find the video under subfolder
    for subfolder in subfolders:
        # Iterate all files in subfolder
        for file in (video_folder / subfolder).rglob('*'):
            #print(file, video_id)
            if video_id in file.name:
                return str(subfolder / Path(file.name))
    return "None"


def split_words(input_string):
    if " " in input_string:
        return input_string
    formatted_string = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', input_string)
    return formatted_string
# print(split_words("MakeRJ45Cable"))


replacement = {"Assemble Desktop P C": "Assemble Desktop PC",
               "Replace Battery On T V Control": "Replace Battery On TV Control",
               "Make R J45 Cable": "Make RJ45 Cable",
               "Make R J45Cable": "Make RJ45 Cable",
               "Attend N B A Skills Challenge": "Attend NBA Skills Challenge",
               "Perform C P R": "Perform CPR",
               "Replace C D Drive With S S D": "Replace CD Drive With SSD",
               "Replace S I M Card": "Replace SIM Card"}


def format_answer(answer):
    answer = split_words(answer) # "MakeCandle"
    if answer in replacement.keys():
        answer = replacement[answer]
    return answer


def get_frame_indices_with_duration(start_sec, end_sec, fps, total_frame):
    start_frame = int(np.round(start_sec * fps))
    end_frame = int(np.round(end_sec * fps)) - 1
    if end_frame >= total_frame:
        end_frame = total_frame -1
    return range(start_frame, end_frame + 1)

def get_frame_indices(total_frame):
    start_frame = 0
    end_frame = total_frame -1
    return range(start_frame, end_frame + 1)

def uniform_sample_frame(frame_indices, max_frames):
    cur_len = len(frame_indices)
    if cur_len > max_frames:
        indices = np.linspace(0, cur_len-1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in indices]
    return frame_indices

def decord_video_given_start_end_seconds(video_pth:str, start_secs:float=-1, end_secs:float=-1,
        skip_interval:int=0, num_video_frames=64):
    # 1. Read video and fps
    vr = VideoReader(video_pth)
    frame_rate = vr.get_avg_fps()
    total_frame  = vr._num_frame
   # print("start {}, end {} fps {} total {}".format(start_secs, end_secs, frame_rate, total_frame))

    # 2. Calculate start, stop index
    if start_secs > 0 and end_secs > 0:
        frame_indices = get_frame_indices_with_duration(start_secs, end_secs, frame_rate, total_frame)
    else:
        frame_indices = get_frame_indices(total_frame)

    frame_indices = [i for i in frame_indices]
    frame_indices = uniform_sample_frame(frame_indices, num_video_frames)

    # . Fetch frames
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
        if frames.shape[0] == 0:
            print("WARNING: {} can not be decord successfuly".format(video_pth))
            frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            frame_indices = [0, 1]
        del vr
    except:
        # cann't decord correctly
        print("WARNING: {} can not be decord successfuly".format(video_pth))
        frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        frame_indices = [0, 1]
    return frames, frame_indices


def load_json(one_file, miss_vid_file, video_dir, image_dir):
    with open(miss_vid_file, "r") as f:
        lines = f.readlines()
        miss_list = [a.strip() for a in lines]

    annots = ""
    with open(one_file, "r") as f:
        annots = json.load(f)

    sft_annots = []
    global_idx = 0
    for one_line in tqdm(annots):
        video_id = one_line['video_id']
        if video_id in miss_list:
            print("Video file is missiong. {}".format(video_id))
            continue

        options  = one_line['options']
        options = [format_answer(opt) for opt in options]
        answer = options[one_line['answer']]
        start_secs = one_line['step']['segment'][0]
        end_secs   = one_line['step']['segment'][1]
        video_path = find_video_path_by_id(video_id, video_dir)
        # extract the middle frame of the video segment.
        frames, _ = decord_video_given_start_end_seconds(os.path.join(video_dir, video_path), 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=4)
        images =[ Image.fromarray(x).convert('RGB') for x in frames ]
        qid_idx = one_line['qid'].split("_")[-1]
        image_paths = []
        for image in images:
            image_path = os.path.join(image_dir, video_id+"_{}_{:08d}.png".format(qid_idx, global_idx))
            image.save(image_path)
            image_paths.append(image_path)
            global_idx += 1
            # print("Saved image to", image_path)

        one_sample = {
            "qid": one_line['qid'],
            "quest_type": one_line['quest_type'],
            "question": one_line['question'],
            "answer": answer,
            "image_paths": image_paths,
            "video": video_path,
            "start_secs": start_secs,
            "end_secs": end_secs,
            "task_label": one_line["task_label"],
            "step_label": one_line["step"]["label"],
            "options": options
        }
        sft_annots.append(one_sample)
    return sft_annots


def main(args):
    kgvqa_dir = args.kgvqa_dir
    miss_vid_file = os.path.join(kgvqa_dir, args.miss_vid_file)

    all_ann_files = [
        # "data/kgvqa/training_small_100.json",
        "data/kgvqa/validation_small_50.json",
        # "data/kgvqa/testing.json"
    ]

    for json_path in all_ann_files:
        print("Processing {}...".format(json_path))
        if "train" in json_path:
            split = "train"
        elif "val" in json_path:
            split = "val"
        else:
            split = "test"
        image_dir = os.path.join(args.image_dir, split)
        os.makedirs(image_dir, exist_ok=True)
        sft_annos = load_json(json_path, miss_vid_file, args.video_dir, image_dir)
        # sufix = json_path.split("/")[-1]
        out_file = os.path.join(args.image_dir, f"mac_{split}")
        with open(out_file, "w") as f:
            json.dump(sft_annos, f, indent=2)
    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miss-vid-file", type=str, default="miss_vid_list_1494.txt")
    parser.add_argument("--kgvqa-dir", type=str, default="data/kgvqa")
    parser.add_argument("--video-dir", type=str, default="data/COIN/videos")
    parser.add_argument("--image_dir", type=str, default="data/coin-image")
    parser.add_argument("--split", type=str, default="")
    args = parser.parse_args()
    main(args)
