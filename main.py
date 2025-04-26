import numpy as np
import ffmpeg
from glob import glob
import os

#иниициализация пути для библиотеки и директории с входным видео
ffmpeg_path = r'FFmpeg/bin/ffmpeg.exe'
video_file = glob(os.path.join('input', '*mp4'))

#проверка на наличие файла в директории и наличие директории
if (os.path.exists('input') == False or len(video_file) == 0):
    print("There are no directory or directory is empty!")
    raise SystemExit
if (os.path.exists('output') == False):
    os.mkdir('output')

video_path = video_file[0]

#декодирование видео в кадры, которые затем превращаются в массив данных numpy
def video_to_array(input_video):
    probe = ffmpeg.probe(input_video)
    video_info = next(i for i in probe ['streams'] if i ['codec_type'] == 'video')
    width = int(video_info ['width'])
    height = int(video_info ['height'])
    out,_ = (
        ffmpeg
        .input (input_video)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout = True)
   )
    video_array = (
        np
        .frombuffer(out, np.uint8)
       .reshape([-1, height, width, 3])
   )
    return video_array


#псевдо обработка
array = video_to_array(video_path)
array.flags.writeable = False
processed_video = array.copy()

# Цикл для обработки (работает крайне медленно)
for n in range(processed_video.shape[0]): # Цикл по кадрам
    for i in range(processed_video.shape[1]): # Цикл по высоте (H)
        for j in range(processed_video.shape[2]): # Цикл по ширине (W)
            for c in range(processed_video.shape[3]): # Цикл по каналам (C)
                if 50 <= processed_video[n, i, j, c] <= 75:
                    processed_video[n, i, j, c] = 125

#массив обратно в видео
def array_to_video(frames, output_path, fps):
    height, width = frames.shape[1], frames.shape[2]
    out = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps)
        .output(output_path, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in frames:
        out.stdin.write(frame.tobytes())
    
    out.stdin.close()
    out.wait()

array_to_video(processed_video, "output/output.mp4", 25)