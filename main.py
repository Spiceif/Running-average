import numpy as np
import ffmpeg
from glob import glob
import os
from scipy.ndimage import gaussian_filter, median_filter, binary_dilation, binary_erosion, label
import time

# Константы
ALPHA = 0.7  # Коэффициент для running average
THRESHOLD = 25  # Порог для обнаружения движения

MIN_CONTOUR_AREA = 529 # Минимальная площадь контура
GAUSSIAN_SIGMA = 0 # Параметр размытия Гаусса
PRE_MEDIAN_SIZE = 0 # Размер медианного фильтра (предобработка)
POST_MEDIAN_SIZE = 0 # Размер медианного фильтра (постобработка)
MORPHOLOGY_ITER = 20 # Количество итераций дилатации
FPS = 25 # Количество кадров в секунду
BOX_COLOR = (255, 0, 0)  # Цвет для прямоугольников
BOX_THICKNESS = 2  # Толщина линий прямоугольника

# Путь к FFmpeg
ffmpeg_path = r'FFmpeg/bin/ffmpeg.exe'

# Глобальная переменная для фона
background = None

# Проверка наличий директорий
def setup_directories():
    if not os.path.exists('input') or not glob(os.path.join('input', '*.mp4')):
        raise FileNotFoundError("Directory 'input' does not exist or is empty!")
    if not os.path.exists('output'):
        os.mkdir('output')

# Декодирование видео в массив данных numpy
def video_to_array(video):
    probe = ffmpeg.probe(video)
    video_info = next(i for i in probe['streams'] if i['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    out,_ = ffmpeg.input(video).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
    result = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return result

# Предобработка
def preprocess(frame):
    # Преобразование в полутон
    frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
    # Векторизованная обработка всех каналов сразу
    if PRE_MEDIAN_SIZE != 0:
       frame = median_filter(frame, PRE_MEDIAN_SIZE)
    if GAUSSIAN_SIGMA != 0 :
        frame = gaussian_filter(frame, GAUSSIAN_SIGMA)
    return frame

# Определение движения
def motion_detector(current_frame):
    global background
    # Инициализация фона
    if background is None:
        background = current_frame.copy()
        return np.zeros_like(current_frame, dtype=np.uint8)
    # Обновление фона
    background = (1 - ALPHA) * background + ALPHA * current_frame
    # Разница между текущим кадром и фоном
    diff = np.abs(current_frame - background)
    # Пороговая обработка
    motion_mask = (diff > THRESHOLD).astype(np.uint8) * 255
    return motion_mask

# Постобработка
def postprocess(frame):
    if POST_MEDIAN_SIZE != 0:
       frame = median_filter(frame, POST_MEDIAN_SIZE)
    if MORPHOLOGY_ITER != 0:
        #frame = binary_erosion(frame, iterations=1).astype(np.uint8) * 255
        frame = binary_dilation(frame, iterations=MORPHOLOGY_ITER).astype(np.uint8) * 255
    return frame

# Отрисовка границ объектов
def draw_bounding_boxes(mask_frame, original_frame):
    labeled, n_features = label(mask_frame > 0)
    result = original_frame.copy()
    # Обработка каждого объекта
    for i in range(1, n_features+1):
        # Нахождение координат всех пикселей i-го объекта
        y_indices, x_indices = np.where(labeled == i)
        # Определение границы объекта
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Пропуск слишком маленьких объектов
        if (x_max - x_min) * (y_max - y_min) < MIN_CONTOUR_AREA:
            continue
        # Отрисовка прямоугольника
        result[y_min:y_min+BOX_THICKNESS, x_min:x_max+1] = BOX_COLOR
        result[y_max-BOX_THICKNESS+1:y_max+1, x_min:x_max+1] = BOX_COLOR
        result[y_min:y_max+1, x_min:x_min+BOX_THICKNESS] = BOX_COLOR
        result[y_min:y_max+1, x_max-BOX_THICKNESS+1:x_max+1] = BOX_COLOR
    return result

def process_video_frames(unprocessed_frames):
    processed_frames = []
    # Исходные кадры
    for frame in unprocessed_frames:
        # Применение фильтров
        filtered = preprocess(frame)
        # Получение маски движения
        motion_mask = motion_detector(filtered)
        # Улучшение маски
        motion_mask = postprocess(motion_mask)
        # Отрисовка прямоугольников на оригинальном кадре
        result_frame = draw_bounding_boxes(motion_mask, frame)
        processed_frames.append(result_frame)
    return np.array(processed_frames)

# Кодирование массивов обратно в видео
def array_to_video(frames, output_path, fps):
    height, width = frames.shape[1], frames.shape[2]
    out = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps).output(output_path, pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
    for frame in frames:
        out.stdin.write(frame.tobytes())
    out.stdin.close()
    out.wait()

def main():
    start = time.time()
    setup_directories()
    video_path = glob(os.path.join('input', '*.mp4'))[0]
    unprocessed_frames = video_to_array(video_path)
    processed_frames = process_video_frames(unprocessed_frames)
    output_path = os.path.join('output', 'output.mp4')
    array_to_video(processed_frames, output_path, FPS)
    end = time.time()
    print(end - start)
    print("Обработка завершена! Результат сохранен в output/output.mp4")

if __name__ == "__main__":
    main()