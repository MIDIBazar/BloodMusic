# Blood Music Live Sonification (OpenCV)

A live MIDI sonification engine that maps blood cell detection data to musical events and visualizes a ghost particle flow in OpenCV.

## Overview

`live_sonification_opencv.py` streams MIDI from a hardware controller to a real-time engine that triggers notes based on blood cell annotations. It supports:

- BCCD dataset image + annotation loading from a ZIP archive
- MIDI CC parameter mapping and pad / action learning
- Several musical scales and harmony modes
- OpenCV visualization of the particle flow and active triggers
- Real-time control over flow, trigger, and musical parameters

## Features

- Live MIDI controller mapping for parameters such as speed, wave amplitude, trigger radius, and harmony mode
- MIDI pad actions for cycling scales, changing harmony, freezing the simulation, resetting the particle, and silencing notes
- Harmony patterns for cell labels: RBC, WBC, Platelets
- Visual overlay of cells, trigger glow, particle trail, and control HUD

## Requirements

- Python 3.10+ recommended
- `numpy`
- `pandas`
- `opencv-python`
- `Pillow`
- `mido`
- `python-rtmidi`

## Installation

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python live_sonification_opencv.py
```

### Configuration

At the top of `live_sonification_opencv.py`, update the following values if needed:

- `CONTROLLER_INPUT_NAME`: partial or full name of your MIDI controller input
- `MIDI_OUTPUT_NAME`: partial or full name of the MIDI output port
- `BCCD_ZIP_PATH`: path to the `blood_dataset.zip` archive
- `BCCD_IMAGE_ID`: image identifier to load from the BCCD dataset
- `USE_BCCD`: set to `False` to load `events_BloodImage_00000_basic.csv` instead of the ZIP dataset
- `LEARN_MODE`: set to `True` to enter MIDI learn mode and create a mapping file

### MIDI mapping

The script stores MIDI mappings in `midi_mapping.json`. If the file is missing or `LEARN_MODE` is enabled, the script runs an interactive MIDI learn process.