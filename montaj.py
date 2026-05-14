import os
import random
import numpy as np
import librosa
from moviepy import VideoFileClip, concatenate_videoclips

class MontageEngine:
    def __init__(self, audio_path, footage_dir, output_file="final_montage.mp4", bpm=116.9):
        self.audio_path = audio_path
        self.footage_dir = footage_dir
        self.output_file = output_file
        self.bpm = bpm
        self.beat_duration = 60.0 / bpm  # Длительность одной доли (четверти)
        
    def get_audio_rms_envelope(self):
        """
        Анализирует аудио и возвращает нормализованную огибающую RMS, 
        квантованную точно под 25 кадров в секунду.
        """
        print(f"Анализ RMS (25 FPS): {self.audio_path}")
        # Загружаем аудио
        y, sr = librosa.load(self.audio_path, sr=None)
        
        # Чтобы RMS идеально ложился на 25 кадров в секунду,
        # hop_length должен быть равен sr / 25
        hop_len = int(sr / 25)
        
        # Считаем RMS
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_len)[0]
        
        # Нормализуем значения в диапазон [0, 1]
        if rms.max() > 0:
            rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
            
        return rms, sr, hop_len

    def get_kick_peaks(self):
        """
        Агрессивный анализатор транзиентов. Ищет резкие всплески энергии (Kick/Snare).
        """
        print(f"Запуск агрессивного поиска пиков для бобслей-монтажа...")
        y, sr = librosa.load(self.audio_path, sr=None)
        
        # Разделяем на гармоническую и перкуссивную составляющие
        # Нам нужна только перкуссия (удары), чтобы не реагировать на гул баса
        y_perc = librosa.effects.percussive(y)
        
        # Вычисляем огибающую силы ударов
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, aggregate=np.median)
        
        # Используем onset_detect — он лучше находит музыкальные события
        # backtrack=True помогает точнее найти начало удара, чтобы склейка не "мазала"
        # wait=4 означает минимальный интервал между склейками ~0.1 секунды
        peaks = librosa.onset.onset_detect(onset_envelope=onset_env,
                                           sr=sr,
                                           wait=1,
                                           pre_avg=3,
                                           post_avg=3,
                                           pre_max=3,
                                           post_max=3,
                                           delta=0.07, # Низкий порог чувствительности (ловим всё!)
                                           backtrack=True)
        
        # Переводим кадры анализа в секунды
        peak_times = librosa.frames_to_time(peaks, sr=sr)
        
        # Добавляем границы файла
        audio_dur = librosa.get_duration(y=y, sr=sr)
        peak_times = np.sort(np.unique(np.concatenate(([0.0], peak_times, [audio_dur]))))
        
        print(f"Анализ завершен: Найдено {len(peak_times)-1} динамических точек для склейки.")
        return peak_times

    def slice_footage(self):
        # Метод нарезки, адаптированный под огромное количество микро-клипов
        
        abs_footage_dir = os.path.abspath(self.footage_dir)
        print(f"--- Дебаг: Сканирую папку {abs_footage_dir} ---")
        
        if not os.path.isdir(abs_footage_dir):
            print(f"Ошибка: Директория не найдена: {abs_footage_dir}")
            return []

        video_files = [
            os.path.join(abs_footage_dir, f)
            for f in os.listdir(abs_footage_dir)
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        ]
        
        if not video_files:
            return []
            
        self.audio_path = self._ensure_audio_source()
        
        # Получаем наши "взрывные" пики
        peak_times = self.get_kick_peaks()
        
        # Грузим метаданные исходников
        source_clips = [VideoFileClip(f) for f in video_files]
        clips_data = []
        last_source_idx = -1
        
        print(f"Начинаем скоростную шинковку видео...")
        
        for i in range(len(peak_times) - 1):
            t_start, t_end = peak_times[i], peak_times[i+1]
            duration_needed = t_end - t_start
            
            # Если кусок слишком короткий (меньше 0.04 сек), пропускаем, чтобы не было мерцания
            if duration_needed < 0.04:
                continue
            
            # Выбор случайного видео
            random_idx = random.randint(0, len(source_clips) - 1)
            while random_idx == last_source_idx and len(source_clips) > 1:
                random_idx = random.randint(0, len(source_clips) - 1)
            last_source_idx = random_idx
            
            src_clip = source_clips[random_idx]
            
            # Рандомизируем место в исходнике
            max_start = max(0, src_clip.duration - duration_needed)
            cut_start = random.uniform(0, max_start)
            
            # Нарезаем. В MoviePy 2.0 subclipped работает быстро
            cut_clip = src_clip.subclipped(cut_start, cut_start + duration_needed)
            
            clips_data.append({
                'clip': cut_clip,
                'source': video_files[random_idx],
                'duration': duration_needed
            })
            
            if i % 100 == 0 and i > 0:
                print(f"Нарезано {i} клипов...")
            
        print(f"Итого: Готово {len(clips_data)} микро-клипов для бобслей-монтажа.")
        return clips_data
        
    def _ensure_audio_source(self):
        """
        Если нет отдельного аудио-файла, извлекает аудио из первого доступного футажа.
        """
        if self.audio_path and os.path.exists(self.audio_path):
            return self.audio_path
        
        # Если аудио нет, берем звук из первого найденного футажа
        files =[os.path.join(self.footage_dir, f) for f in os.listdir(self.footage_dir)
                 if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        
        if not files:
            raise FileNotFoundError("Нет ни аудио-трека, ни футажей с аудио!")
        
        print(f"Предупреждение: Аудио-трек не найден. Извлекаю звук из {files[0]}...")
        temp_audio = "extracted_audio.wav"
        video = VideoFileClip(files[0])
        video.audio.write_audiofile(temp_audio, codec='pcm_s16le')
        video.close()
        return temp_audio

    def build_montage(self):
        # Основной метод сборки, теперь полностью завязанный на динамику аудио
        
        # 1. Сначала подготавливаем данные
        clips_data = self.slice_footage()
        if not clips_data:
            return

        # 2. Получаем RMS, который теперь строго синхронен с 25 FPS видео
        rms, sr, hop_len = self.get_audio_rms_envelope()
        
        # Перемешивание (уже сделано в slice_footage по пикам, но здесь подтверждаем структуру)
        final_sequence = [d['clip'] for d in clips_data]
        
        processed_clips = []
        current_time_offset = 0
        target_w, target_h = final_sequence[0].size
        
        print(f"Сборка финала (Zoom-and-Crop + Kick-Sync)...")
        
        for clip in final_sequence:
            def make_zoom_frame(get_frame, t):
                total_t = current_time_offset + t
                # Так как RMS квантован 1/25, индекс — это просто номер кадра
                # 25 кадров в секунду: индекс = время * 25
                idx = int(total_t * 25)
                
                val = rms[min(idx, len(rms)-1)]
                
                # Делаем зум чуть более агрессивным на пиках бочки
                zoom_factor = 1.0 + (val * 1.3)
                
                frame = get_frame(t)
                import cv2
                h, w = frame.shape[:2]
                new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
                
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                sx = (new_w - w) // 2
                sy = (new_h - h) // 2
                return resized[sy:sy+h, sx:sx+w]
            
            processed_clips.append(clip.transform(make_zoom_frame))
            current_time_offset += clip.duration

        final_video = concatenate_videoclips(processed_clips, method="chain")
        
        from moviepy import AudioFileClip
        full_audio = AudioFileClip(self.audio_path)
        
        safe_dur = min(final_video.duration, full_audio.duration)
        final_video = final_video.subclipped(0, safe_dur).with_audio(full_audio.subclipped(0, safe_dur))
        
        print(f"Рендеринг финального файла...")
        final_video.write_videofile(
            self.output_file,
            fps=25,
            codec='libx264',
            audio_codec='aac',
            preset='slow', # Для максимального качества на M2
            threads=13
        )
        
        for c in final_sequence: c.close()
        full_audio.close()
        final_video.close()

if __name__ == "__main__":
    # Умный поиск аудио: ищем любой mp3 или wav в текущей папке
    audio_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
    found_audio = audio_files[0] if audio_files else None
    
    ENGINE = MontageEngine(
        audio_path=found_audio, # Если None, выдернет из видео сам
        footage_dir="footage_folder",
        output_file="montage_final.mp4"
    )
    ENGINE.build_montage()
