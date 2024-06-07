import numpy as np
import librosa
import sys
from midiutil import MIDIFile
import matplotlib.pylab as plt

file = sys.argv[-1]
assert file.endswith(".mp3")

y, sr = librosa.load(file, sr=22050)
press_time = 0.01

notes = [(119, 7902.133), (118, 7458.62), (117, 7040), (116, 6644.875), (115, 6271.927), (114, 5919.911), (113, 5587.652), (112, 5274.041), (111, 4978.032), (110, 4698.636), (109, 4434.922), (108, 4186.009), (107, 3951.066), (106, 3729.31), (105, 3520), (104, 3322.438), (103, 3135.963), (102, 2959.955), (101, 2793.826), (100, 2637.02), (99, 2489.016), (98, 2349.318), (97, 2217.461), (96, 2093.005), (95, 1975.533), (94, 1864.655), (93, 1760), (92, 1661.219), (91, 1567.982), (90, 1479.978), (89, 1396.913), (88, 1318.51), (87, 1244.508), (86, 1174.659), (85, 1108.731), (84, 1046.502), (83, 987.7666), (82, 932.3275), (81, 880), (80, 830.6094), (79, 783.9909), (78, 739.9888), (77, 698.4565), (76, 659.2551), (75, 622.254), (74, 587.3295), (73, 554.3653), (72, 523.2511), (71, 493.8833), (70, 466.1638), (69, 440), (68, 415.3047), (67, 391.9954), (66, 369.9944), (65, 349.2282), (64, 329.6276), (63, 311.127), (62, 293.6648), (61, 277.1826), (60, 261.6256), (59, 246.9417), (58, 233.0819), (57, 220), (56, 207.6523), (55, 195.9977), (54, 184.9972), (53, 174.6141), (52, 164.8138), (51, 155.5635), (50, 146.8324), (49, 138.5913), (48, 130.8128), (47, 123.4708), (46, 116.5409), (45, 110), (44, 103.8262), (43, 97.99886), (42, 92.49861), (41, 87.30706), (40, 82.40689), (39, 77.78175), (38, 73.41619), (37, 69.29566), (36, 65.40639), (35, 61.73541), (34, 58.27047), (33, 55), (32, 51.91309), (31, 48.99943), (30, 46.2493), (29, 43.65353), (28, 41.20344), (27, 38.89087), (26, 36.7081), (25, 34.64783), (24, 32.7032), (23, 30.86771), (22, 29.13524), (21, 27.5), (20, 25.95654), (19, 24.49971), (18, 23.12465), (17, 21.82676), (16, 20.60172), (15, 19.44544), (14, 18.35405), (13, 17.32391), (12, 16.3516)]
notes = notes[::-1]

uppers = [(None, 0)]
for n, m in zip(notes, notes[1:]):
    mid = (n[1] + m[1]) / 2
    uppers.append((n[0], mid))
uppers += [(notes[-1][0], notes[-1][1] * 1.01)]

ranges = []
for n, m in zip(uppers, uppers[1:]):
    ranges.append((m[0], (n[1], m[1])))

samples = int(sr * press_time)

midi = MIDIFile(1)
midi.addTempo(0, 0, 60 * sr // samples)

std = 200
sample_length = std * 8
mean = sample_length / 2
gaussian = np.array([
    1/(std * np.sqrt(2 * np.pi)) * np.exp( - (i - mean)**2 / (2 * std**2))
    for i in range(sample_length)])

weights = []
for start in range(0, len(y)-sample_length, samples):
    print(start // samples,"/",(len(y)-sample_length) // samples)
    bit = y[start:start+sample_length] * gaussian
    fft = np.fft.rfft(bit)
    freq = np.fft.rfftfreq(len(bit)) * sr
    ws = {i: 0.0 for i, _ in ranges}
    for i, j in zip(freq, fft):
        for note, r in ranges:
            if r[0] <= i < r[1]:
                ws[note] = max(abs(j), ws[note])
                break
    weights.append(ws)

max_weight = max(max(i.values()) for i in weights)

playing = {}
for i, w in enumerate(weights):
    max_weight = max(w.values())
    for note, ww in w.items():
        if ww > max_weight / 100:
            www = int(ww / max_weight * 127)
            if note in playing:
                if abs(www - playing[note][0]) > 5:
                    midi.addNote(0, 0, note, playing[note][1], i - playing[note][1], playing[note][0])
                    del playing[note]
            else:
                playing[note] = (www, i)

for note in playing:
    midi.addNote(0, 0, note, playing[note][1], len(weights) - playing[note][1], playing[note][0])

with open(f"{file}.mid", "wb") as output_file:
    midi.writeFile(output_file)
