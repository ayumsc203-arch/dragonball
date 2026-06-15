# Dragon Ball Z Kamehameha Effect

Uses your webcam and pose detection to overlay DBZ effects when you do a Kamehameha pose.

---

## What it does

- Detects your body pose in real time using MediaPipe
- Shows an **energy ball** when you bring your wrists together (charging pose)
- Shows a **Kamehameha beam** when you extend your arms forward (firing pose)

---

## Requirements

```bash
pip install opencv-python mediapipe numpy
```

You also need:
- `pose_landmarker_heavy.task` — download from [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) and place in the project root
- `assets/energy.mp4` — energy charge effect video
- `assets/kamehameha.mp4` — beam effect video (black background works best)

---

## Project structure

```
project/
├── main.py
├── pose_landmarker_heavy.task
└── assets/
    ├── energy.mp4
    └── kamehameha.mp4
```

---

## How to run

```bash
python main.py
```

Press `Q` or `ESC` to quit.

---

## How to trigger effects

| Pose | Effect |
|---|---|
| Wrists close together, arms bent | Charging energy ball |
| Wrists close together, arms extended | Kamehameha beam |

---

## Tips

- Good lighting improves detection accuracy
- Effect videos should have a black background for clean blending
- Swap `pose_landmarker_heavy.task` with the lite version if it runs slow
