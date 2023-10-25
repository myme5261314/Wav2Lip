from hparams import hparams
import os
import numpy as np
import audio


def get_mel_cache(wavpath: str, cache: dict):
    cache_mel_path = "%s.mel_%d.npy" % (wavpath, hparams.sample_rate)
    orig_mel = cache.get(cache_mel_path, None)
    if orig_mel is None:
        if os.path.exists(cache_mel_path):
            orig_mel = np.load(cache_mel_path)
            cache[cache_mel_path] = orig_mel
        else:
            wav = audio.load_wav(wavpath, hparams.sample_rate)
            orig_mel = audio.melspectrogram(wav).T

            np.save(cache_mel_path, orig_mel)
            cache[cache_mel_path] = orig_mel
    return orig_mel
