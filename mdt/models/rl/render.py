from pathlib import Path
import torch
import os
import numpy as np
import cv2
import wandb
from moviepy.editor import ImageSequenceClip

def delete_tmp_video(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def add_text(img: np.ndarray, lang_text: str) -> np.ndarray:
    height, width, _ = img.shape
    if lang_text != "":
        coord = (1, int(height - 10))
        font_scale = (0.7 / 500) * width
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return img

def _unnormalize(img):
    return img / 2 + 0.5
class RolloutVideo:
    def __init__(self, log_to_file: bool, save_dir: str, resolution_scale: int=1, log_to_wandb: bool=False):
        self.videos = []
        self.video_paths = {}
        self.tags = []
        self.captions = []
        self.log_to_file = log_to_file
        self.save_dir = Path(save_dir)
        self.sub_task_beginning = 0
        self.step_counter = 0
        self.resolution_scale = resolution_scale
        self.log_to_wandb = log_to_wandb
        if self.log_to_file:
            os.makedirs(self.save_dir, exist_ok=True)

    def new_video(self, tag: str, caption: str) -> None:
        """
        Begin a new video with the first frame of a rollout.
        Args:
             tag: name of the video
             caption: caption of the video
        """
        # (1, 1, channels, height, width)
        self.videos.append(torch.Tensor())
        self.tags.append(tag)
        self.captions.append(caption)
        self.step_counter = 0
        self.sub_task_beginning = 0

    def draw_outcome(self, successful):
        """
        Draw red or green border around video depening on successful execution
        and repeat last frames.
        Args:
            successful: bool
        """
        c = 1 if successful else 0
        not_c = list({0, 1, 2} - {c})
        border = 3
        frames = 5
        self.videos[-1][:, -1:, c, :, :border] = 1
        self.videos[-1][:, -1:, not_c, :, :border] = 0
        self.videos[-1][:, -1:, c, :, -border:] = 1
        self.videos[-1][:, -1:, not_c, :, -border:] = 0
        self.videos[-1][:, -1:, c, :border, :] = 1
        self.videos[-1][:, -1:, not_c, :border, :] = 0
        self.videos[-1][:, -1:, c, -border:, :] = 1
        self.videos[-1][:, -1:, not_c, -border:, :] = 0
        repeat_frames = torch.repeat_interleave(self.videos[-1][:, -1:], repeats=frames, dim=1)
        self.videos[-1] = torch.cat([self.videos[-1], repeat_frames], dim=1)
        self.step_counter += frames

    def new_subtask(self):
        self.sub_task_beginning = self.step_counter

    def update(self, rgb_obs: torch.Tensor) -> None:
        """
        Add new frame to video.
        Args:
            rgb_obs: static camera RGB images
        """
        img = rgb_obs.detach().cpu()
        self.videos[-1] = torch.cat([self.videos[-1], _unnormalize(img)], dim=1)  # shape 1, t, c, h, w
        self.step_counter += 1


    def add_language_instruction(self, instruction: str) -> None:
        img_text = np.zeros(self.videos[-1].shape[2:][::-1], dtype=np.uint8) + 127
        img_text = add_text(img_text, instruction)
        img_text = ((img_text.transpose(2, 0, 1).astype(float) / 255.0) * 2) - 1
        self.videos[-1][:, self.sub_task_beginning :, ...] += torch.from_numpy(img_text)
        self.videos[-1] = torch.clip(self.videos[-1], -1, 1)

    def write_to_tmp(self):
        """
        In case of logging with WandB, save the videos as GIF in tmp directory,
        then log them at the end of the validation epoch from rank 0 process.
        """
        for video, tag in zip(self.videos, self.tags):
            video = np.clip(video.numpy() * 255, 0, 255).astype(np.uint8)
            wandb_vid = wandb.Video(video, fps=10, format="gif")
            self.video_paths[tag] = wandb_vid._path
        self.videos = []
        self.tags = []

    def log(self, global_step: int) -> None:
        """
        Call this method at the end of a validation epoch to log videos to tensorboard, wandb or filesystem.
        Args:
            global_step: global step of the training
        """
        if self.log_to_file:
            self._log_videos_to_file(global_step)
        if self.log_to_wandb:
            self._log_videos_to_wandb()
        else:
            raise NotImplementedError
        self.videos = []
        self.tags = []
        self.captions = []
        self.video_paths = {}

    def _log_videos_to_wandb(self):
        video_paths = self.video_paths
        captions = self.captions
        for (task, path), caption in zip(video_paths.items(), captions):
            wandb.log({f"video{task}": wandb.Video(path, fps=20, format="gif", caption=caption)})
            delete_tmp_video(path)
    

    def _log_videos_to_file(self, global_step, save_as_video=True):
        """
        Mostly taken from WandB
        """
        for video, tag in zip(self.videos, self.tags):
            if len(video.shape) == 4:
                video = video.unsqueeze(0)
            video = np.clip(video.numpy() * 255, 0, 255).astype(np.uint8)

            mpy = wandb.util.get_module(
                "moviepy.editor",
                required='wandb.Video requires moviepy and imageio when passing raw data.  Install with "pip install moviepy imageio"',
            )
            tensor = self._prepare_video(video)
            # Resize tensor if resolution scale is not 1.0
            _, _height, _width, _channels = tensor.shape
            
            tag = tag.replace("/", "_")
            if save_as_video:
            # encode sequence of images into gif string
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                filename = str(self.save_dir / f"{tag}_{global_step}.mp4")
                clip.write_videofile(filename, codec='libx264', bitrate="5000k")  # You can adjust the bitrate as needed
            else:
                clip = mpy.ImageSequenceClip(list(tensor), fps=20)
                filename = self.save_dir / f"{tag}_{global_step}.gif"
                clip.write_gif(filename, logger=None)
    
    def save_frames_to_subfolder(self, n, rollout_index):
        # Ensure n is a valid number
        if n <= 0 or not isinstance(n, int):
            raise ValueError("n must be a positive integer.")

        # Create a new subfolder for the rollout
        subfolder_path = self.save_dir / f'rollout_{rollout_index}'
        os.makedirs(subfolder_path, exist_ok=True)

        # Iterate through all videos in self.videos
        for video_idx, video_tensor in enumerate(self.videos):
            # Assuming video_tensor shape is (1, t, c, h, w)
            _, total_frames, channels, height, width = video_tensor.shape

            # Create a sub-subfolder for each video
            video_subfolder_path = subfolder_path / f'video_{video_idx}'
            os.makedirs(video_subfolder_path, exist_ok=True)

            # Iterate through the video tensor and save every nth frame to the subfolder
            for frame_index in range(0, total_frames, n):
                frame = video_tensor[0, frame_index].permute(1, 2, 0).cpu().numpy()
                if channels == 1:  # If grayscale, remove the color dimension
                    frame = frame.squeeze(-1)
                frame_image = Image.fromarray((frame * 255).astype('uint8'))  # Assuming frame values are normalized
                frame_image.save(video_subfolder_path / f'frame_{frame_index}.png')

        print(f'Saved frames from {len(self.videos)} videos to {subfolder_path}')

    @staticmethod
    def _prepare_video(video):
        """This logic was mostly taken from tensorboardX"""
        if video.ndim < 4:
            raise ValueError("Video must be atleast 4 dimensions: time, channels, height, width")
        if video.ndim == 4:
            video = video.reshape(1, *video.shape)
        b, t, c, h, w = video.shape

        if video.dtype != np.uint8:
            video = video.astype(np.uint8)

        def is_power2(num):
            return num != 0 and ((num & (num - 1)) == 0)

        # pad to nearest power of 2, all at once
        if not is_power2(video.shape[0]):
            len_addition = int(2 ** video.shape[0].bit_length() - video.shape[0])
            video = np.concatenate((video, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

        n_rows = 2 ** ((b.bit_length() - 1) // 2)
        n_cols = video.shape[0] // n_rows

        video = np.reshape(video, newshape=(n_rows, n_cols, t, c, h, w))
        video = np.transpose(video, axes=(2, 0, 4, 1, 5, 3))
        video = np.reshape(video, newshape=(t, n_rows * h, n_cols * w, c))
        return video
