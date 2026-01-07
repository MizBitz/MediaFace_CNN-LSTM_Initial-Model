# Pose Categories Guide - DataGen_Side.py

## Overview
The enhanced `DataGen_Side.py` now automatically categorizes head poses and saves data to separate folders with pose-specific EAR thresholds.

## Pose Categories

### 1. **LEFT** - Head Turned Left
- **Yaw Range**: 30° to 90° (turning left)
- **Pitch Range**: -25° to 25° (slight up/down)
- **EAR Thresholds**: Open: 0.22, Closed: 0.10
- **Instructions**: Turn your head to the left while keeping eyes level
- **Output Path**: `dataset_angled/left/open/` and `dataset_angled/left/closed/`

### 2. **RIGHT** - Head Turned Right
- **Yaw Range**: -90° to -30° (turning right)
- **Pitch Range**: -25° to 25°
- **EAR Thresholds**: Open: 0.22, Closed: 0.10
- **Instructions**: Turn your head to the right while keeping eyes level
- **Output Path**: `dataset_angled/right/open/` and `dataset_angled/right/closed/`

### 3. **DOWN** - Looking Downward
- **Yaw Range**: -30° to 30° (head relatively centered)
- **Pitch Range**: 25° to 75° (looking down significantly)
- **EAR Thresholds**: Open: 0.25, Closed: 0.12 *(More relaxed because EAR inflates when looking down)*
- **Instructions**: Tilt your head down to look at your phone/desk
- **Output Path**: `dataset_angled/down/open/` and `dataset_angled/down/closed/`

## Understanding the Angles

- **Yaw (Side-to-Side)**: 
  - Positive = Looking left
  - Negative = Looking right
  - 0° = Looking straight ahead

- **Pitch (Up-Down)**:
  - Positive = Looking down
  - Negative = Looking up
  - 0° = Looking straight ahead

## Why Different EAR Thresholds?

EAR (Eye Aspect Ratio) changes based on head pose:
- When looking **down**, eyes appear more open (higher EAR value)
- When looking **left/right**, EAR stays relatively stable
- Thresholds are tuned to each pose for consistent labeling

## Output Structure

```
dataset_angled/
├── left/
│   ├── open/
│   │   ├── open_0_64x64_L.jpg
│   │   ├── open_0_64x64_R.jpg
│   │   └── open_0_64x64_Combined.jpg
│   └── closed/
├── right/
│   ├── open/
│   └── closed/
└── down/
    ├── open/
    └── closed/
```

## Running the Script

```bash
# Activate your environment
conda activate gpu_env

# Run the script
python DataGen_Side.py
```

## Tips for Best Results

1. **Lighting**: Ensure good, even lighting on your face
2. **Camera Position**: Keep camera at eye level for better pose detection
3. **Movement**: Make smooth, continuous head movements between poses
4. **Hold Position**: Once a pose is detected, hold it steady for several frames to collect samples
5. **Variety**: Collect data from different distances and angles within each category

## Adjusting Thresholds

Edit `POSE_CATEGORIES` in `DataGen_Side.py` to fine-tune:
- **Angle ranges**: Expand or contract to capture more/fewer poses
- **EAR thresholds**: Adjust if you're getting wrong labels for a pose

Example:
```python
"down": {
    "yaw_range": (-30, 30),
    "pitch_range": (25, 75),
    "ear_open": 0.25,      # Increase if missing open eyes
    "ear_closed": 0.12     # Decrease if missing closed eyes
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pose not detected | Check you're in the right angle range for the pose |
| Wrong labels | Adjust EAR thresholds for that specific pose |
| Poor face detection | Improve lighting or move closer to camera |
| All frames rejected | Increase blur threshold or check camera quality |
