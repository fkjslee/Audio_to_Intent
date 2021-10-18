import argparse
from src.stream_analyzer import Stream_Analyzer
import time


# reference: git@github.com:aiXander/Realtime_PyAudio_FFT.git

class AudioGuiConfig:
    def __init__(self):
        self.device = None  # pyaudio (portaudio) device index
        self.height = 450
        self.frequency_bins = 400  # The FFT features are grouped in bins
        self.verbose = False
        self.window_ratio = "24/9"
        self.sleep_between_frames = True


class AudioController:
    def __init__(self):
        self.hidden = True
        self.fps_msg = ""
        self.pred_res = []


def parse_args():
    return AudioGuiConfig()


def convert_window_ratio(window_ratio):
    if '/' in window_ratio:
        dividend, divisor = window_ratio.split('/')
        try:
            float_ratio = float(dividend) / float(divisor)
        except Exception:
            raise ValueError('window_ratio should be in the format: float/float')
        return float_ratio
    raise ValueError('window_ratio should be in the format: float/float')


def run_FFT_analyzer(controller: AudioController = None):
    args = parse_args()
    window_ratio = convert_window_ratio(args.window_ratio)

    ear = Stream_Analyzer(
        device=args.device,  # Pyaudio (portaudio) device index, defaults to first mic input
        rate=None,  # Audio samplerate, None uses the default source settings
        FFT_window_size_ms=60,  # Window size used for the FFT transform
        updates_per_second=1000,  # How often to read the audio stream for new data
        smoothing_length_ms=50,  # Apply some temporal smoothing to reduce noisy features
        n_frequency_bins=args.frequency_bins,  # The FFT features are grouped in bins
        visualize=1,  # Visualize the FFT features with PyGame
        verbose=args.verbose,  # Print running statistics (latency, fps, ...)
        height=args.height,  # Height, in pixels, of the visualizer window,
        window_ratio=window_ratio  # Float ratio of the visualizer window. e.g. 24/9
    )

    fps = 60  # How often to update the FFT features + display
    last_update = time.time()
    while True:
        if (time.time() - last_update) > (1. / fps):
            last_update = time.time()
            raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
            if controller and ear.visualize:
                ear.visualizer.hidden = controller.hidden
                ear.visualizer.other_fps_msg = controller.fps_msg
                ear.visualizer.pred_res = controller.pred_res
        elif args.sleep_between_frames:
            time.sleep(max(((1. / fps) - (time.time() - last_update)) * 0.99, 0))


if __name__ == '__main__':
    run_FFT_analyzer()
