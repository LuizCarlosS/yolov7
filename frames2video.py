import cv2
import numpy as np
import glob
import click
import os
from tqdm import tqdm

@click.command()
@click.option('-t', '--target', type=str)
def main(target, frame_rate=30):

    image_folder = target
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, frame_rate, (width,height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()