from .encoder import inference as encoder
from .synthesizer.inference import Synthesizer
from .vocoder import inference as rnn_vocoder
from pathlib import Path

import numpy as np
import torch
import re
import os
from pathlib import Path
from scipy.io.wavfile import write
from typing import Dict

import matplotlib.pyplot as plt

ROOT = "rtvc/"
# Constants
ENC_MODELS_DIR = ROOT+"model/encoder"
SYN_MODELS_DIR = ROOT+"model/synthesizer"
VOC_MODELS_DIR = ROOT+"model/vocoder"
# Path of inference wav
TEMP_FOLDER = "static/temp"
TEMP_SOURCE_AUDIO = "static/temp/temp_source.wav"
TEMP_RESULT_AUDIO = "static/temp/temp_result.wav"


# Pre-Load models
if os.path.isdir(SYN_MODELS_DIR):
    synthesizers = {}
    for file in Path(SYN_MODELS_DIR).glob("**/*.pt"):
        synthesizers[file.name] = file
    print("Loaded synthesizer models: " + str(len(synthesizers.keys())))
else:
    raise Exception(f"Model folder {SYN_MODELS_DIR} doesn't exist.")

if os.path.isdir(ENC_MODELS_DIR):
    encoders = {}
    for file in Path(ENC_MODELS_DIR).glob("**/*.pt"):
        encoders[file.name] = file
    print("Loaded encoders models: " + str(len(encoders.keys())))
else:
    raise Exception(f"Model folder {ENC_MODELS_DIR} doesn't exist.")

if os.path.isdir(VOC_MODELS_DIR):
    vocoders = {}
    for file in Path(VOC_MODELS_DIR).glob("**/*.pt"):
        vocoders[file.name] = file
    print("Loaded vocoders models: " + str(len(synthesizers.keys())))
else:
    raise Exception(f"Model folder {VOC_MODELS_DIR} doesn't exist.")


def embed_extract(input_path: Path, encoder_path: Path):
    """
    args:
        input_path: the input person voice path
        encoder_path: the encoder model_path
    return:
        (wav, spec, embed) -> Tuple
        wav: the input voice path wav numpy array
        spec: 2D np.array of voice spectrogram
        embed: (256,) 1D array of embed image
    """
    wav = Synthesizer.load_preprocess_wav(Path(input_path))
    spec = Synthesizer.make_spectrogram(wav)

    encoder.load_model(Path(encoder_path))
    encoder_wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(encoder_wav)

    return wav, spec, embed


def plot_spec(spec, name="noName"):
    """
    args:
        spec: spectrogram
    description:
        image_size: 10, 2 -> (width, height)
        axis_number: hidden
    """
    plt.figure(figsize=(10, 2))
    plt.axis("off")
    plt.imshow(spec, aspect="auto", interpolation="none")
    plt.savefig(f'{TEMP_FOLDER}/spec-{name}.png', transparent=True)
    plt.show()


def plot_embed(embed, name="noName"):
    """
    args:
        embed: embed image
    description:
        image_size: reshape to 16 * 16
        axis_number: hidden
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(embed.reshape(16, 16), aspect="auto", interpolation="none")
    plt.axis("off")
    plt.colorbar().set_ticks([])
    plt.savefig(f'{TEMP_FOLDER}/embed-{name}.png', transparent=True)
    plt.show()


def get_encoders() -> Dict[str, Path]:
    return encoders


def get_synthesizers() -> Dict[str, Path]:
    return synthesizers


def get_vocoder() -> Dict[str, Path]:
    return vocoders


def synthesize_voice(speaker_path: Path,
                     encoder: str,
                     synthesizer: str,
                     vocoder: str,
                     text: str = 'Hello, I am a robot'
                     ):
    encoder_path = get_encoders()[encoder]
    synthesizer_path = get_synthesizers()[synthesizer]
    vocoder_path = get_vocoder()[vocoder]
    fname = speaker_path.name.split('.')[0]

    wav, spec, embed = embed_extract(speaker_path, encoder_path)
    plot_spec(spec, "temp_input")
    plot_embed(embed, "temp_input")
    write(TEMP_SOURCE_AUDIO, 16000, wav.astype(np.float32))

    generate = RTVC()
    generate.init_model(encoder_path, synthesizer_path, vocoder_path)
    embed_wav, spec, embed = generate.synthesize(
        synthesizer_path, vocoder_path, text, embed)
    plot_spec(spec, "temp_output")
    plot_embed(embed, "temp_output")

    write(
        TEMP_RESULT_AUDIO, generate.sample_rate, embed_wav.astype(
            np.float32)
    )  # Make sure we get the correct wav


class RTVC:
    """
    parameters:
        sythesizer: the synthesizer model to generate spectrogram
        sample_rate: sample_rate of synthesizer
        voder: the vocoder model used to generate wav
        seed: the random seed of sythesizer and vocoder torch model
    methods:
        init_encoder: load encoder (model_path)
        init_synthesizer: load synthesizer (model_path, model_config_path)
        init_vocoder: load vocoder (model_path, model_config_path)
        ==========================================================
        synthesize: load text and target embed to generate fake voice of target
    """

    def __init__(self, vocoder=rnn_vocoder, seed=4234) -> None:
        self.synthesizer = None
        self.sample_rate = None
        self.vocoder = vocoder
        self.seed = seed

    def init_encoder(self, model_fpath: Path) -> None:
        encoder.load_model(model_fpath)

    def init_synthesizer(self, model_fpath: Path) -> None:
        self.synthesizer = Synthesizer(model_fpath)
        self.sample_rate = Synthesizer.sample_rate

    def init_vocoder(self, model_fpath: Path) -> None:
        self.vocoder.load_model(model_fpath)

    def init_model(self, encoder_path: Path, synthesizer_path: Path, vocoder_path: Path) -> None:
        self.init_encoder(encoder_path)
        self.init_synthesizer(synthesizer_path)
        self.init_vocoder(vocoder_path)

    def synthesize(
        self,
        synthesizer_path: Path,
        vocoder_path: Path,
        texts: str,
        embed,
    ):
        """
        args:
            synthesizer_path: the synthesizer_model path
            vocoder_path: the vocoder_model path
            texts: the text you want voice to generate
            embed: the target embed
            max_length: the max of each partition of sentece split by punctuation
            style: the speech style
            steps: the evaluation steps
        return:
            (encoder_wav, spec, embed) -> Tuple
            encoder_wav: generated wav process by encoder
            spec: 2D np.array of voice spectrogram
            embed: (256,) 1D array of embed image
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Synthesize the spectrogram
        if self.synthesizer is None or self.seed is not None:
            self.init_synthesizer(synthesizer_path)

        punctuation = "[!,.?\n]"  # punctuate and split/clean text
        texts = re.split(punctuation, texts)
        while '' in texts:
            texts.remove('')
        embed = embed
        embeds = [embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        # =============== #
        # below is vocode #
        # ================#
        if not self.vocoder.is_loaded() or self.seed is not None:
            self.init_vocoder(vocoder_path)

        wav = self.vocoder.infer_waveform(spec)

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * self.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        wav = encoder.preprocess_wav(wav)
        wav = wav / np.abs(wav).max() * 0.97

        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed = encoder.embed_utterance(encoder_wav)

        return encoder_wav, spec, embed


# TODO: TEST model
# default_text = "Welcome to use the SemiTrue, and we are pleasure to give you nice experience"
# sentence_length = 2
# fpath = Path(AUDIO_SAMPLES_DIR + "/han_input.wav")
# encoder_path = Path(ENC_MODELS_DIR + "/encoder.pt")
# synthesizer_path = Path(SYN_MODELS_DIR + "/synthesizer.pt")
# vocoder_path = Path(VOC_MODELS_DIR + "/vocoder.pt")

# wav, spec, embed = embed_extract(fpath, encoder_path)
# plot_spec(spec, fpath.name.split('.')[0])
# plot_embed(embed, fpath.name.split('.')[0])


# generate = SemiTrue()
# generate.init_model(encoder_path, synthesizer_path, vocoder_path)
# embed_wav, spec, embed = generate.synthesize(
#     synthesizer_path, vocoder_path, default_text, embed)
# plot_spec(spec, "output-han")
# plot_embed(embed, "output-han")

# write(
#     TEMP_RESULT_AUDIO, generate.sample_rate, embed_wav.astype(np.float32)
# )  # Make sure we get the correct wav
