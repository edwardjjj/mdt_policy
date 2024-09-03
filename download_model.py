
"""Download datasets using rclone or gdown."""
import os
import argparse
import subprocess


def download_folder_rclone(model, out_dir):
    """Use rclone to download a folder from the Google Drive."""
    path = f"Pretrained_Models/{model}/"
    target_path = f"{out_dir}/{model}/"

    if os.path.exists(target_path):
        print(f"Skipped {model}: file already exists at {target_path}")
        return

    print(f"Start downloading Google Drive folder {path}")
    command = f"rclone copy -P furniture:{path} {out_dir}/{model}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print(f"Finished downloading Google Drive folder {path} in {out_dir}/{model}")


def main():
    out_dir = "pretrained_models"
    download_list = [
        "CALVIN ABCD",
        "CALVIN D",
    ]
    for model in download_list:
        download_folder_rclone(model, out_dir)

if __name__ == "__main__":
    main()
