import numpy as np
import ffmpeg
from glob import glob
import os
from scipy.ndimage import gaussian_filter, median_filter, binary_dilation
import cv2  # только для findContours и drawContours (альтернатив нет)

# Константы
ALPHA = 0.95  # Коэффициент для running average
THRESHOLD = 3  # Порог для обнаружения движения
MIN_CONTOUR_AREA = 200 # Минимальная площадь контура
GAUSSIAN_SIGMA = 1 # Параметр размытия Гаусса
MEDIAN_SIZE = 3 # Размер медианного фильтра
DILATION_ITER = 20 # Количество итераций дилатации
FPS = 25 # Количество кадров в секунду
BOX_COLOR = (0, 0, 255)  # Цвет для прямоугольников
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

# Декодирование видео в кадры, которые затем превращаются в массив данных numpy
def video_to_array(video):
    probe = ffmpeg.probe(video)
    video_info = next(i for i in probe['streams'] if i['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    out,_ = ffmpeg.input(video).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
    result = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return result

# Применение фильтров по 3м каналам отдельно
def apply_filters(frame):
    filtered_frame = np.zeros_like(frame)
    for channel in range(3):
        channel_data = frame[..., channel]
        channel_data = gaussian_filter(channel_data, sigma=GAUSSIAN_SIGMA)
        channel_data = median_filter(channel_data, size=MEDIAN_SIZE)
        filtered_frame[..., channel] = channel_data
    return filtered_frame

# Определение движения
def motion_detector(current_frame):
    global background
    gray = np.dot(current_frame[...,:3], [0.2989, 0.5870, 0.1140])                                                      # Преобразуем в grayscale
    if background is None:                                                                                                                    # Инициализация фона
        background = gray.copy()
        return np.zeros_like(gray, dtype=np.uint8)
    background = (1 - ALPHA) * background + ALPHA * gray                                                             # Running average для фона
    diff = np.abs(gray - background)                                                                                                   # Разница между текущим кадром и фоном
    motion_mask = (diff > THRESHOLD).astype(np.uint8) * 255                                                        # Пороговая обработка
    return motion_mask

# Отрисовка контуров
def draw_bounding_boxes(mask_frame, original_frame):
    contours,_ = cv2.findContours(mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Находим контуры
    result_frame = original_frame.copy()                                                                                           # Создаем копию оригинального кадра
    for cnt in contours:                                                                                                                       # Рисуем прямоугольники вокруг объектов
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)                                                                                      # Получаем координаты ограничивающего прямоугольника
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), BOX_COLOR, BOX_THICKNESS)                 # Рисуем прямоугольник (используем OpenCV только здесь)
    return result_frame

def process_video_frames(unprocessed_frames):
    processed_frames = []
    for frame in unprocessed_frames:                                                                                               # Исходные кадры
        filtered = apply_filters(frame)                                                                                                 # Применяем фильтры
        motion_mask = motion_detector(filtered)                                                                              # Получаем маску движения
        motion_mask = binary_dilation(motion_mask, iterations=DILATION_ITER).astype(np.uint8) * 255 # Морфологическое улучшение маски
        result_frame = draw_bounding_boxes(motion_mask, frame)                                                 # Рисуем прямоугольники на оригинальном кадре
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
    setup_directories()
    video_path = glob(os.path.join('input', '*.mp4'))[0]
    unprocessed_frames = video_to_array(video_path)
    
    # Обработка кадров
    processed_frames = process_video_frames(unprocessed_frames)
    
    # Сохранение результата
    output_path = os.path.join('output', 'result.mp4')
    array_to_video(processed_frames, output_path, FPS)
    
    print("Обработка завершена! Результат сохранен в output/result.mp4")
if __name__ == "__main__":
    main()