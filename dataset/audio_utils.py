"""Defines routines to compute mel spectrogram features from audio waveform."""

import numpy as np
import resampy
import matplotlib.pyplot as plt
import librosa

SAMPLE_RATE = 16000
LOG_OFFSET = 0.01
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):

  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz < 0.0:
    raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > nyquist_hertz:
    raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)

  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)

  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]

    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix

def frame(data, window_length, hop_length):

  num_samples = data.shape[0]
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
  shape = (num_frames, window_length) + data.shape[1:]
  strides = (data.strides[0] * hop_length,) + data.strides
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def periodic_hann(window_length):
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):

  frames = frame(signal, window_length, hop_length)
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


def get_spectrogram(data,
                        audio_sample_rate=8000,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):

  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)

  return spectrogram


def log_mel_spectrogram(data,
                        audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):

  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)
  mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
      num_spectrogram_bins=spectrogram.shape[1],
      audio_sample_rate=audio_sample_rate, **kwargs))
  return np.log(mel_spectrogram + log_offset)

STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010

NUM_BANDS = 64
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
EXAMPLE_WINDOW_SECONDS = 0.96
EXAMPLE_HOP_SECONDS = 0.96

def waveform_to_examples(data, sample_rate):
  
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  
  if sample_rate != SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, SAMPLE_RATE)
  
  log_mel = log_mel_spectrogram(
      data,
      audio_sample_rate=SAMPLE_RATE,
      log_offset=LOG_OFFSET,
      window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=NUM_MEL_BINS,
      lower_edge_hertz=MEL_MIN_HZ,
      upper_edge_hertz=MEL_MAX_HZ)

  features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

# def waveform_to_spectogram(data, sample_rate):
#   if len(data.shape) > 1:
#     data = np.mean(data, axis=1)
#   # Resample to the rate assumed by VGGish.
#   if sample_rate != SAMPLE_RATE:
#     data = resampy.resample(data, sample_rate, SAMPLE_RATE)

#   # Compute log mel spectrogram features.
#   spec = get_spectrogram(
#       data,
#       audio_sample_rate=SAMPLE_RATE,
#       log_offset=LOG_OFFSET,
#       window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
#       hop_length_secs=STFT_HOP_LENGTH_SECONDS,
#       num_mel_bins=NUM_MEL_BINS,
#       lower_edge_hertz=MEL_MIN_HZ,
#       upper_edge_hertz=MEL_MAX_HZ)
#   return spec