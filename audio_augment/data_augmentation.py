import librosa
import numpy as np
import crepe
from pydub import AudioSegment
import pyrubberband as pyrb

SAMPLE_RATE = 22050


def pitch_sync_mixing(segment1, segment2):
    # Estimate pitch for both segments
    f1, _ = crepe.predict(segment1, sr=SAMPLE_RATE, viterbi=True)
    f2, _ = crepe.predict(segment2, sr=SAMPLE_RATE, viterbi=True)

    # Calculate the required frequency shift in semitones
    s = 12 * np.log2(f1 / f2)

    # Shift the second segment by the computed semitone shifts
    segment2_shifted = pyrb.pitch_shift(segment2, sr=SAMPLE_RATE, n_steps=s)

    # Mix the two segments
    mixed_segment = segment1 + segment2_shifted

    return mixed_segment


def tempo_sync_mixing(segment1, segment2):
    # Detect tempo for both segments
    tempo1, _ = librosa.beat.beat_track(segment1, sr=SAMPLE_RATE)
    tempo2, _ = librosa.beat.beat_track(segment2, sr=SAMPLE_RATE)

    # Calculate the required time stretching factor
    rate = tempo2 / tempo1

    # Stretch the second segment
    segment2_stretched = pyrb.time_stretch(segment2, sr=SAMPLE_RATE, rate=rate)

    # Mix the two segments
    mixed_segment = segment1 + segment2_stretched

    return mixed_segment


def similar_genre_mixing(segment1, segment2):
    # Mix the two segments
    mixed_segment = segment1 + segment2

    return mixed_segment


def combined_augmentation(segment1, segment2):
    # Apply pitch-sync mixing
    pitch_synced = pitch_sync_mixing(segment1, segment2)

    # Apply tempo-sync mixing
    tempo_synced = tempo_sync_mixing(pitch_synced, segment2)

    # Apply similar genre mixing
    mixed_segment = similar_genre_mixing(tempo_synced, segment2)

    return mixed_segment
