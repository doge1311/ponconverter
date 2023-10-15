from tkinter import filedialog
import tkinter as tk
from operator import indexOf
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import time
from dataclasses import dataclass
import math
import sys
import os

mono = False

# Define a function to get the WAV file path using a file dialog
def get_wav_file_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    wav_file_path = filedialog.askopenfilename(title="Select a WAV file")
    if not wav_file_path:
        print("No WAV file selected. Exiting.")
        sys.exit(1)
    return wav_file_path

# Define a function to get the output MIDI file path using a file dialog
def get_output_midi_file_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    output = filedialog.asksaveasfilename(
        title="Save the output MIDI file",
        defaultextension=".mid",
        filetypes=[("MIDI files", "*.mid")]
    )
    if not output:
        print("No output file selected. Exiting.")
        sys.exit(1)
    return output

# Get user input for main variables
windowSize = 8192
stepSize = int(input("Enter the step size (default: 1024): ") or 1024)
fac = 1
wav_file_path = get_wav_file_path()  # Use the file dialog for WAV input
output = get_output_midi_file_path()  # Use the file dialog for MIDI output

@dataclass
class NoteTone:
    key: int
    tick: int
    channel: int
    vel: int
    tickLen: int

@dataclass
class UnendedNote:
    delta: int
    channel: int
    vel: int
    key: int
    tick: int
    tickLen: int



def BPMtoMicroseconds(n):
    return stepSize * 58593.75 / n

def midiNoteFromPitch(n):
    return 12.0 * (math.log(n / 220.0) / math.log(2.0)) + 57.01

def toVQL(n):
    b = [0, 0, 0, 0, 0]
    l = 4
    added = 0x00
    while True:
        v = (n & 0x7F)
        n = n >> 7
        v = v | added
        b[l] = v
        l -= 1
        added = 0x80
        if n == 0:
            break
    return b[l + 1:]

def convToRawData(notes):
    tmpArr = []
    prevTime = 0
    noteOffs = []
    for i in range(len(notes)):
        currNote = notes[i]
        while len(noteOffs) != 0 and noteOffs[0].tick <= currNote.tick:
            e = noteOffs.pop(0)
            e.delta = e.tick - prevTime
            tmpArr += toVQL(e.delta)
            tmpArr += [0x80 | e.channel, e.key, e.vel]
            prevTime = e.tick
        tmpArr += toVQL(currNote.tick - prevTime)
        tmpArr += [0x90 | currNote.channel, currNote.key, currNote.vel]
        prevTime = currNote.tick
        t = currNote.tick + currNote.tickLen
        off = UnendedNote(0, currNote.channel, currNote.vel, currNote.key, t, currNote.tickLen)
        pos = len(noteOffs) // 2
        if len(noteOffs) == 0:
            noteOffs.append(off)
        else:
            j = len(noteOffs) // 4
            while True:
                if j <= 0:
                    j = 1
                if pos < 0:
                    pos = 0
                if pos >= len(noteOffs):
                    pos = len(noteOffs) - 1
                u = noteOffs[pos]
                if u.tick >= t:
                    if pos == 0 or noteOffs[pos - 1].tick < t:
                        noteOffs.insert(pos + 1, off)
                        break
                    else:
                        pos -= j
                else:
                    if pos == len(noteOffs) - 1:
                        noteOffs.append(off)
                        break
                    else:
                        pos += j
                j = j // 2
    for nf in range(len(noteOffs)):
        noteoff = noteOffs[nf]
        noteoff.delta = noteoff.tick - prevTime
        tmpArr += toVQL(noteoff.delta)
        tmpArr += [0x80 | noteoff.channel, noteoff.key, noteoff.vel]
        print(f"Note Event {len(tmpArr)}/{len(notes) * 3} added")
        prevTime = noteoff.tick

    return tmpArr

noteTonesL = []
noteTonesR = []

# Get Sample rate and Signal data

rate, sound = wav.read(wav_file_path)

if "-ws" in sys.argv:
    windowSize = int(sys.argv[indexOf(sys.argv, "-ws") + 1])

if "-mono" in sys.argv:
    mono = True

sound = sound / (2 ** 15)

# Split into two separate arrays
if len(sound.shape) == 1:
    signalL = sound
    mono = True
else:
    signalL = sound.sum(axis=1) / 2 if mono else sound[:, 0]
    signalR = sound[:, 1]

dirname = os.path.dirname(__file__)
f = open(os.path.join(output), "wb")
byte_arr = []
byte_arr += [ord("M"), ord("T"), ord("h"), ord("d")]
byte_arr += [0x00, 0x00, 0x00, 0x06]
byte_arr += [0x00, 0x01, 0x00, 0x02 if mono else 0x03, 0x03, 0xC0]
track0 = []
track0 += [0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]
track0 += [0x00, 0xFF, 0x51, 0x03]
convertedBPM = BPMtoMicroseconds(160)
convertedBPM_int = int(convertedBPM)
track0 += [(convertedBPM_int & 0xFF0000) >> 16, (convertedBPM_int & 0xFF00) >> 8, convertedBPM_int & 0xFF]
track0 += [0x00, 0xFF, 0x2F, 0x00]
byte_arr += [ord("M"), ord("T"), ord("r"), ord("k")]
track0_len = len(track0)
byte_arr += [(track0_len & 0xFF000000) >> 24, (track0_len & 0xFF0000) >> 16, (track0_len & 0xFF00) >> 8,
             (track0_len & 0xFF)]
byte_arr += track0

byte_arr += [ord("M"), ord("T"), ord("r"), ord("k")]
track1 = []

if not mono:
    for i in range(8):
        track1 += [0x00, 0xB0 | i, 0x0A, 0x00]

lastChunkL = False

print(f"Processing L Channel...")

hasPrintedOnce = False

progress = 0

# Goes on until an error lol
for i in range(160000000):
    # Where to put tfr_spec()????
    try:
        chunk = signalL[stepSize * progress:stepSize * progress + windowSize]
        window = signal.windows.blackmanharris(chunk.size)
        chunk = chunk * window
        fft_spec = np.fft.rfft(chunk)
        freq = np.fft.rfftfreq(chunk.size, d=1 / rate)
        spec_abs = np.abs(fft_spec)
        peakFreqs, _ = signal.find_peaks(spec_abs, threshold=-1)
        for p in range(len(peakFreqs)):
            j = peakFreqs[p]
            fr = spec_abs[j]
            if freq[j] > 0.0:
                notePitch = round(midiNoteFromPitch(freq[j]))
                if 1 <= notePitch < 128 and fr / fac > 1:
                    tmpVel = fr / fac
                    vel = 127 if tmpVel > 127 else 1 if tmpVel < 1 else int(tmpVel)
                    noteTonesL.append(NoteTone(notePitch, 58 * i, math.floor(vel / 16), vel, 58))
        progress += 1
        print(f"Chunk {progress} done")
    except:
        progress = 0
        print("Channel L Done")
        break

track1 += convToRawData(noteTonesL)
track1 += [0x00, 0xFF, 0x2F, 0x00]
track1_len = len(track1)
byte_arr += [(track1_len & 0xFF000000) >> 24, (track1_len & 0xFF0000) >> 16, (track1_len & 0xFF00) >> 8,
             (track1_len & 0xFF)]
byte_arr += track1

if not mono:
    byte_arr += [ord("M"), ord("T"), ord("r"), ord("k")]
    track2 = []
    for i in range(8):
        track2 += [0x00, 0xB0 | (i + 8), 0x08, 0x7F]

    lastChunkR = False

    progress = 0

    print("Processing R Channel...")
    for i in range(160000000):
        try:
            chunk = signalR[stepSize * progress:stepSize * progress + windowSize]
            if stepSize * progress + windowSize >= len(signalR) - 1:
                chunk = signalR[stepSize * progress:len(signalR) - 1]
                lastChunkR = True
            window = signal.windows.blackmanharris(chunk.size)
            chunk = chunk * window
            fft_spec = np.fft.rfft(chunk)
            freq = np.fft.rfftfreq(chunk.size, d=1 / rate)
            spec_abs = np.abs(fft_spec)
            peakFreqs, _ = signal.find_peaks(spec_abs, threshold=-1)
            for p in range(len(peakFreqs)):
                j = peakFreqs[p]
                fr = spec_abs[j]
                if freq[j] > 0.0:
                    notePitch = round(midiNoteFromPitch(freq[j]))
                    if 1 <= notePitch < 128 and fr / fac > 1:
                        tmpVel = fr / fac
                        vel = 127 if tmpVel > 127 else 1 if tmpVel < 1 else int(tmpVel)
                        mappedVel = math.floor(vel / 16) + 8
                        if mappedVel == 9:
                            mappedVel = 10
                        noteTonesR.append(NoteTone(notePitch, 58 * i, mappedVel, vel, 58))
            if lastChunkR:
                break
            progress += 1
            print(f"Chunk {progress} done")
        except:
            break

    track2 += convToRawData(noteTonesR)
    track2 += [0x00, 0xFF, 0x2F, 0x00]
    track2_len = len(track2)
    byte_arr += [(track2_len & 0xFF000000) >> 24, (track2_len & 0xFF0000) >> 16, (track2_len & 0xFF00) >> 8,
                 (track2_len & 0xFF)]
    byte_arr += track2

f.write(bytes(byte_arr))
f.close()
