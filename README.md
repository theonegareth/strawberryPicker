<a id="readme-top"></a>

# Strawberry Picker

A five-servo robotic arm that finds ripe strawberries with a YOLOv8n detector, locates them in 3D with two webcams, and cuts the stem with a razor-blade scissor end effector.

<details>
<summary>Table of contents</summary>

- [About the project](#about-the-project)
- [Hardware](#hardware)
- [How it works](#how-it-works)
- [Kinematics and control](#kinematics-and-control)
- [Model and dataset](#model-and-dataset)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

</details>

## About the project

Strawberries are still mostly picked by hand. The work is slow, and the fruit bruises if it is squeezed too hard. This project is a robotic arm that does the whole job in one system: it finds a ripe strawberry, moves to it, cuts the stem with a scissor end effector, and catches the fruit for a basket.

Two webcams act as a stereo pair. A laptop runs the vision: a YOLOv8n model boxes the ripe strawberries in each frame, and the two views give the fruit's position in centimetres. The laptop sends that target over USB serial to an Arduino Uno, which drives the five servos through a PCA9685 servo driver. An early draft of the design had a Raspberry Pi computing the coordinates; the final design does that work on the laptop.

The arm was built by Group 2 (Kelompok 2) for the Robotics Kinematics and Dynamics course at BINUS ASO School of Engineering, from September 2025 to January 2026. A write-up with photos is on the [project page][project-page].

### Team

- **Bryan Wilberd Lo** ([@bryan-wilberd][gh-bryan]): camera calibration, integration of Python and Arduino
- **Enrico Ethan**: control system, transfer functions, PID tuning
- **Gareth Aurelius Harrison** ([@theonegareth][gh-gareth]): machine-learning model, dataset, labelling, training
- **Jancent**: assembly, setting up the environment
- **Samuel Justus Tan** ([@SamJuss][gh-samuel]): forward kinematics, inverse kinematics, Arduino code

### Built with

- Python, with [Ultralytics YOLOv8][ultralytics] on PyTorch for detection
- OpenCV and NumPy for capture, undistortion and stereo triangulation
- TensorFlow / Keras for a separate MobileNetV2 image classifier
- Arduino (C++) with the [Adafruit PWM Servo Driver Library][adafruit-pwm], built with the Arduino IDE or PlatformIO
- SolidWorks for the arm's parts, exported to STL and STEP
- MATLAB for checking the kinematics, Simulink for the servo controller design
- Google Colab for GPU training

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Hardware

| Joint | Servo | PCA9685 channel |
|---|---|---|
| Base (yaw) | MG996R | 0 |
| Shoulder | DS3240 | 1 |
| Elbow | MG996R | 2 |
| Wrist | SG90s | 3 |
| Scissor | MG90s | 4 |

The servo models come from the team's design slides and poster. The channel numbers come from the firmware on the [`commit_fixes`][branch-commit-fixes] branch.

Other parts:

- Arduino Uno
- PCA9685 16-channel servo driver, on I2C at the Adafruit library's default address
- 5 V 10 A power supply for the driver and servos
- two USB webcams, used as a stereo pair
- a razor blade as the cutting edge of the scissor end effector
- jumper wires and a breadboard
- a laptop for the vision, preferably with a GPU

The first version of the arm was a cardboard mock-up. The parts are modelled in SolidWorks, and their files are in [`assets/`](assets). The file names are Indonesian:

| File | Part |
|---|---|
| `baase putar` | rotating base |
| `base` | base |
| `LENGAN` | arm (lengan) |
| `lengan depan1`, `lengan depanZ` | forearm (lengan depan), two versions |
| `Arm 03` | arm part |
| `gripper` | gripper |
| `gripper silet 1` | gripper with the razor blade (silet) |
| `keranjang strawberry` | strawberry basket (keranjang) |
| `CLS6336 35kg servo`, `SG90 - Micro Servo 9g - Tower Pro.1` | servo models for fitting |
| `Assem1.SLDASM` | assembly |
| `arm tanpa hasur download semua.STEP` | STEP export of the arm |

Each part has a `.SLDPRT` file in [`assets/solidworks/`](assets/solidworks) and an STL export in [`assets/stl/`](assets/stl). A newer basket export is at [`assets/keranjang strawberry new.STL`](<assets/keranjang strawberry new.STL>).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How it works

One pick runs as a loop:

1. Both webcams capture a frame.
2. YOLOv8n runs on each frame. The model is trained to box only ripe strawberries, so ripeness is decided at this step.
3. Each left detection is paired with a right detection, and the pair is triangulated into X, Y and Z in centimetres.
4. The axes are remapped from camera to robot. The camera calls depth Z, but the arm calls height Z, so robot X = camera X, robot Y = camera Z (depth becomes horizontal reach), and robot Z = camera Y (height).
5. The laptop sends the target to the Arduino over serial at 9600 baud as `x,z,y` in centimetres, for example `12.50,25.30,5.20`. It sends a new target only after the Arduino prints `READY`, so the arm never gets a new target in the middle of a move.
6. The Arduino solves the inverse kinematics, checks the joint limits, and moves the servos through the PCA9685.
7. The scissor servo cuts the stem, and the gripper catches the strawberry for the basket.

If no strawberry is in view, the cameras take another photo.

Where each step lives in this repository:

- Steps 1 to 3: [`combine3.py`](combine3.py) captures one frame from each camera, undistorts both, runs YOLO, matches the detections, and triangulates them with `cv2.triangulatePoints`. It shows the result on screen and does not open a serial port.
- Steps 5 and 6: the firmware in [`ArduinoCode/inverse kinematics serial`][fw-ik-serial] on the `commit_fixes` branch. It reads `x,z,y`, prints `BUSY` while it works and `DONE` then `READY` when it finishes, and rejects targets that are out of reach or past a joint limit.
- Step 7: the interactive firmware in [`ArduinoCode/inverse kinematics`][fw-ik] on the same branch has a `c` command that opens and closes the scissor servo eight times. The serial firmware defines the scissor moves but its serial loop does not call them.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Kinematics and control

The arm is modelled as a planar two-link arm (shoulder and elbow) with a tool offset, swung round by a yaw joint at the base. The joint variables are q = [θ0, θ1, θ2, θ3] for base, shoulder, elbow and wrist.

Forward kinematics:

- r = L1 cos θ1 + L2 cos θ2 + L3
- z = L1 sin θ1 + L2 sin θ2 + z0
- x = r cos θ0 + x0, y = r sin θ0 + y0

Inverse kinematics runs the other way. The base angle is θ0 = atan2(y, x). The remaining reach r = √(x² + y²) − L3 and the height z form a triangle with the two links, so the law of cosines gives the elbow: cos θ2 = (r² + z² − L1² − L2²) / (2 L1 L2). A target is reachable only when |L1 − L2| ≤ √(r² + z²) ≤ L1 + L2.

The serial firmware on `commit_fixes` implements this on the Uno with these constants:

| Constant | Value |
|---|---|
| `L1` | 20.0 cm |
| `L2` | 13.2 cm |
| `L3` | 7.0 cm |
| Height offset in inverse kinematics | z − 9.0 cm |
| Height offset in forward kinematics | z + 7.5 cm (ground to shoulder) |
| Base offset in forward kinematics | 1.4 cm |
| Shoulder limits | 10° to 160° |
| Elbow limit | 140° maximum |
| Wrist limit when the elbow is at 140° | 90° minimum |

After each move it runs forward kinematics on the angles it reached and prints the result, so the inverse solution can be checked against the target.

The dynamics are τ = M(q)q̈ + C(q, q̇)q̇ + g(q) + τf. For slow servo motion the gravity term dominates, which is why the firmware enforces joint limits and moves in small interpolated steps: no sudden force spikes that could bruise the fruit. The serial firmware moves all five servos over 2 s with a 10 ms (100 Hz) update. Its PID loop runs on an estimated servo position from a first-order model (time constant 0.15 s), with Kp = 1.0, Ki = 0 and Kd = 0.

The servo controllers were also designed in Simulink, on transfer-function models of the servos, as discrete controllers with a sample time of 0.01 s. These gains are simulation results:

| Servo | Role | Controller | P | I | D |
|---|---|---|---|---|---|
| MG996R | base, elbow | PID | 4.5023 | 17.4680 | 0.2680 |
| DS3240 | shoulder | PI | 0.0556 | 11.1175 | 0 |
| SG90s | wrist | PI | 0.0698 | 13.9675 | 0 |
| MG90s | scissor | PI | 0.0278 | 5.5564 | 0 |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model and dataset

The first detector was trained on strawberry images from Kaggle and Roboflow. On the bench it detected pictures of strawberries but not real strawberries in front of the webcam.

The fix was the data:

- **Mix:** 50% Kaggle images, for the volume to learn the shape, and 50% home-made webcam images, for real lighting.
- **Negative samples:** images of leaves, shelves and clothing with no strawberries in them, so a red shirt does not count as fruit.
- **Labels:** bounding boxes in YOLO format, drawn only on ripe strawberries, made with a small Flask labelling app served at `http://localhost:5000`.
- **Size:** 232 images plus 56 negative images. The labels hold 198 strawberry instances. Their box centres cluster in the upper and middle part of the frame, so the set has some positional bias.
- **Augmentation:** mosaic (100%), HSV shifts for lighting, horizontal flips (50%), scaling (50%), and translation (10 to 20%).

The team compared YOLOv8 with YOLOv11 and chose YOLOv8n for real-time detection.

[`train_yolov8.py`](train_yolov8.py) fine-tunes the pretrained `yolov8n.pt` on a single-class YOLO dataset: 100 epochs, 640 px images, batch 16, early stopping after 20 epochs without improvement, a checkpoint every 10 epochs, and image caching. It uses the GPU when PyTorch finds one. [`train_yolov8_colab.ipynb`][colab] runs the same training on Google Colab, then exports ONNX and TensorFlow Lite copies and downloads them. Neither applies INT8 quantization. The live system ran the PyTorch weights (`best.pt`) on the laptop.

The dataset and trained weights are not on `main`: [`.gitignore`](.gitignore) excludes `model/dataset/`, `model/weights/`, and `*.pt`. The `commit_fixes` branch keeps the December 2025 training runs, with their weights, under [`model/detection/`][model-detection].

The repository also holds a separate image classifier. [`train_model.py`](train_model.py) (the same code is in [`strawberry_training.ipynb`](strawberry_training.ipynb)) trains a two-class good/bad head on a frozen, ImageNet-pretrained MobileNetV2 at 224×224. [`collect_dataset.py`](collect_dataset.py) gathers its images from a webcam, [`convert_tflite.py`](convert_tflite.py) converts the Keras model to TensorFlow Lite, and [`detect_realtime.py`](detect_realtime.py) labels whole webcam frames with it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

Detector, on the validation set:

| Metric | Value |
|---|---|
| mAP@0.5 | about 0.976 |
| Best F1 | about 0.94, at a confidence threshold of about 0.33 |
| Precision | close to 1.0 above a confidence of about 0.62 |
| Recall | about 0.98 at very low confidence |
| Confusion matrix | 101 ripe strawberries detected, 3 false positives, 7 missed |

Run live on the laptop with the two webcams, the detector showed 8.7 and 9.4 FPS.

The inverse kinematics was checked in MATLAB. For a target at (0.1, 0.1, 0.1) m it returned joint angles of 45.00°, 109.02°, −132.35° and 0.00°. Forward kinematics on those angles landed on the target with an error norm of 5.72 × 10⁻¹⁷ m.

Source for all figures in this section: the team's final course presentation and poster.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository layout

```text
strawberryPicker/
├── ArduinoCode/
│   └── codingservoarm.ino       # servo test: moves channels 0 and 1 to a typed angle
├── assets/
│   ├── solidworks/              # SolidWorks parts, assembly, STEP export
│   ├── stl/                     # STL exports of the parts
│   └── keranjang strawberry new.STL
├── combine3.py                  # two-camera capture, YOLO, stereo triangulation
├── train_yolov8.py              # YOLOv8n training script
├── train_yolov8_colab.ipynb     # the same training on Google Colab
├── setup_training.py            # installs requirements and checks the training setup
├── requirements.txt             # Python dependencies
├── collect_dataset.py           # webcam capture into dataset/good and dataset/bad
├── train_model.py               # MobileNetV2 good/bad classifier
├── strawberry_training.ipynb    # the same classifier as a notebook
├── convert_tflite.py            # Keras classifier to TensorFlow Lite
├── detect_realtime.py           # live webcam classification with the TFLite model
├── webcam_capture.py            # saves a frame from camera 2 every second
├── download_dataset.py          # Roboflow download template
├── get-pip.py                   # pip bootstrap script
├── TRAINING_README.md           # training guide
└── GITHUB_SETUP.md              # git and GitHub setup notes
```

`main` is a clean re-initialization made on 2026-01-20. The full December 2025 working tree is on other branches:

| Branch | Contents |
|---|---|
| [`commit_fixes`][branch-commit-fixes] | Latest full tree (2025-12-25): forward and inverse kinematics firmware, camera calibration, the stereo-to-serial scripts, datasets, training runs with weights, and documentation |
| [`backup`][branch-backup] | Tree from 2025-12-16, including the first inverse kinematics firmware |
| [`clean-branch`][branch-clean] | Restored tree from 2025-12-16 |
| [`clean-new-branch`][branch-clean-new] | Same commit as `main` |

Some figures in the older branches' documents, such as their READMEs, came before the final results and are superseded by the [Results](#results) section.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

### Prerequisites

- [Python][python] 3.8 or newer (`setup_training.py` stops on older versions) with pip
- Git
- for the stereo script: two USB webcams and Windows, because [`combine3.py`](combine3.py) opens the cameras through DirectShow (`cv2.CAP_DSHOW`)
- for the arm: the [Arduino IDE][arduino-ide] (or PlatformIO for the `commit_fixes` firmware), an Arduino Uno, a PCA9685 board, the servos and a 5 V supply
- optional: an NVIDIA GPU with CUDA, or a Google account for Colab

### Installation

1. Clone the repository.

   ```sh
   git clone https://github.com/theonegareth/strawberryPicker.git
   cd strawberryPicker
   ```

2. Create a virtual environment and activate it.

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with `.venv\Scripts\activate`.

3. Install the dependencies. `train_yolov8.py` and `setup_training.py` import PyYAML, which `requirements.txt` does not list, so install it too.

   ```sh
   pip install -r requirements.txt pyyaml
   ```

4. Check the setup. This checks the Python version, PyTorch, Ultralytics, the GPU and the dataset without installing anything.

   ```sh
   python setup_training.py --validate-only
   ```

5. Put a YOLO-format dataset at `model/dataset/straw-detect.v1-straw-detect.yolov8/`, with a `data.yaml` like this:

   ```yml
   train: ../train/images
   val: ../valid/images
   test: ../test/images

   nc: 1
   names: ['strawberry']
   ```

   `train_yolov8.py --dataset` takes another location.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Train the detector

```sh
python train_yolov8.py --validate-only
python train_yolov8.py --epochs 100 --batch-size 16 --img-size 640 --export-onnx
```

| Flag | Default | Effect |
|---|---|---|
| `--dataset PATH` | `model/dataset/straw-detect.v1-straw-detect.yolov8` | dataset directory holding `data.yaml` |
| `--epochs N` | 100 | training epochs |
| `--batch-size N` | 16 | batch size |
| `--img-size N` | 640 | training image size |
| `--export-onnx` | off | export ONNX after training |
| `--validate-only` | off | check the dataset and stop |

The Ultralytics run goes to `model/results/strawberry_detection/`, and the final model is saved as `model/weights/strawberry_yolov8n.pt`.

To train on Colab instead, open [`train_yolov8_colab.ipynb`][colab], switch to a GPU runtime, set `DATASET_PATH` to the dataset in your Google Drive, and run the cells in order.

### Locate a strawberry in 3D

1. Copy the trained weights into the repository root as `strawberry.pt`.
2. Mount the two webcams 15 cm apart, each turned about 14° inward so their axes meet 30 cm ahead.
3. Set `CAM_A_ID` (left) and `CAM_B_ID` (right) in `combine3.py` to your camera indexes.
4. Run the script:

   ```sh
   python combine3.py
   ```

It prints the configuration and the number of left-right matches, then one line per strawberry, such as `Target (strawberry): Z=30.0cm (X=..., Y=...)`. A window shows both undistorted frames side by side with the boxes and match lines. Press any key to close it.

The camera matrices and distortion coefficients in the script belong to the team's two webcams. Other cameras need their own calibration; the checkerboard capture and calibration scripts are in [`cam_callibration/IPM`][calibration] on the `commit_fixes` branch.

### Test the servos

1. Open [`ArduinoCode/codingservoarm.ino`](ArduinoCode/codingservoarm.ino) in the Arduino IDE. The IDE asks to move it into a folder named `codingservoarm`; accept.
2. Install the Adafruit PWM Servo Driver Library from the Library Manager.
3. Select the Arduino Uno board and its port, and upload.
4. Open the Serial Monitor at 9600 baud with a newline line ending, and type an angle from 0 to 180.

The servos on channels 0 and 1 move together to that angle in 0.1° steps and print `Done!`.

### Run the inverse kinematics firmware

The firmware that drives the whole arm is on the `commit_fixes` branch, as PlatformIO projects for the Uno:

```sh
git switch commit_fixes
cd "ArduinoCode/inverse kinematics serial"
pio run -e uno -t upload
pio device monitor
```

`platformio.ini` sets the serial monitor to `COM5` at 9600 baud; change `monitor_port` to your port. Send a target as `x,z,y` in centimetres, for example `12.50,25.30,5.20`.

The interactive version in [`ArduinoCode/inverse kinematics`][fw-ik] takes typed commands instead:

| Command | Action |
|---|---|
| `i x z y` | inverse kinematics to a target in centimetres; the second value is the height (the start-up prompt says `i x y z`, but the code reads x, z, y) |
| `f t0 t1 t2` | forward kinematics from joint angles in degrees |
| `r` | reset pose |
| `9` | 90° pose |
| `e` | extended pose |
| `c` | cut: open and close the scissor servo eight times |

### Train the good/bad classifier

```sh
python collect_dataset.py
python train_model.py
python convert_tflite.py
python detect_realtime.py
```

In `collect_dataset.py`, press `g` to save the frame as good, `b` to save it as bad, and `q` to quit. `train_model.py` trains for 10 epochs on `dataset/` with an 80/20 split and saves `strawberry_model.h5`. `convert_tflite.py` writes `strawberry_model.tflite`, and `detect_realtime.py` shows the label and confidence on the live feed until you press `q`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration

The project has no environment variables. Settings are constants at the top of each file.

`combine3.py`:

| Constant | Value | Meaning |
|---|---|---|
| `CAM_A_ID` | 2 | left camera index |
| `CAM_B_ID` | 1 | right camera index |
| `FRAME_W`, `FRAME_H` | 640, 408 | capture size |
| `YOLO_MODEL_PATH` | `strawberry.pt` | detector weights |
| `BASELINE_CM` | 15.0 | distance between the cameras |
| `FOCUS_DIST_CM` | 30.0 | distance where the camera axes meet; sets the inward angle of ±14.04° |
| `DEPTH_SCALAR` | 0.82 | depth correction from a bench check: a raw reading of about 36.6 cm for a strawberry at 30 cm |
| `K_A`, `dist_A`, `K_B`, `dist_B` | calibrated values | camera matrices and distortion coefficients |

Left and right detections are matched only when they have the same class and their centres are within 60 px vertically. The script's notes give a working range of 20 to 40 cm.

`ArduinoCode/codingservoarm.ino`:

| Constant | Value | Meaning |
|---|---|---|
| `SERVOMIN` | 150 | PCA9685 pulse count for 0° |
| `SERVOMAX` | 600 | PCA9685 pulse count for 180° |
| PWM frequency | 50 Hz | servo frame rate |
| `step`, `delayTime` | 0.10°, 5 ms | smoothing: smaller steps or longer delays move slower |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

The course project ended in January 2026. During it, the team merged its work through [pull requests][pulls]. To propose a change:

1. Fork the repository.
2. Create a branch: `git switch -c fix/short-description`.
3. Commit your changes with a message that says what changed and why.
4. Push the branch and open a pull request against `main`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

The `main` branch has no licence file. The `commit_fixes` branch carries an [MIT License][license] (copyright 2025 Gareth).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Gareth Aurelius Harrison, [@theonegareth][gh-gareth]

Project page: [theonegareth.com/projects/strawberry-picking-robotic-arm][project-page]

Repository: [github.com/theonegareth/strawberryPicker][repo]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Ultralytics YOLOv8][ultralytics], the detector and training framework
- [Adafruit PWM Servo Driver Library][adafruit-pwm], used by the Arduino code for the PCA9685
- [Fruit Ripeness Dataset][kaggle-dataset] by dudinurdiyansah on Kaggle, the source of the Kaggle images in the `commit_fixes` dataset scripts
- [LearnOpenCV: Stereo Vision and Depth Estimation][learnopencv], the reference the team used for the pinhole depth model

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[repo]: https://github.com/theonegareth/strawberryPicker
[project-page]: https://theonegareth.com/projects/strawberry-picking-robotic-arm/
[gh-gareth]: https://github.com/theonegareth
[gh-samuel]: https://github.com/SamJuss
[gh-bryan]: https://github.com/bryan-wilberd
[pulls]: https://github.com/theonegareth/strawberryPicker/pulls?q=is%3Apr
[license]: https://github.com/theonegareth/strawberryPicker/blob/commit_fixes/LICENSE
[branch-commit-fixes]: https://github.com/theonegareth/strawberryPicker/tree/commit_fixes
[branch-backup]: https://github.com/theonegareth/strawberryPicker/tree/backup
[branch-clean]: https://github.com/theonegareth/strawberryPicker/tree/clean-branch
[branch-clean-new]: https://github.com/theonegareth/strawberryPicker/tree/clean-new-branch
[fw-ik-serial]: https://github.com/theonegareth/strawberryPicker/tree/commit_fixes/ArduinoCode/inverse%20kinematics%20serial
[fw-ik]: https://github.com/theonegareth/strawberryPicker/tree/commit_fixes/ArduinoCode/inverse%20kinematics
[calibration]: https://github.com/theonegareth/strawberryPicker/tree/commit_fixes/cam_callibration/IPM
[model-detection]: https://github.com/theonegareth/strawberryPicker/tree/commit_fixes/model/detection
[colab]: https://colab.research.google.com/github/theonegareth/strawberryPicker/blob/main/train_yolov8_colab.ipynb
[ultralytics]: https://github.com/ultralytics/ultralytics
[adafruit-pwm]: https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library
[kaggle-dataset]: https://www.kaggle.com/datasets/dudinurdiyansah/fruit-ripeness-dataset
[learnopencv]: https://learnopencv.com/stereo-vision-and-depth-estimation-using-opencv-ai-kit/
[python]: https://www.python.org/downloads/
[arduino-ide]: https://www.arduino.cc/en/software
