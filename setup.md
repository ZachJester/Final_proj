# Setup and Inference Guide 

Welcome! This guide will hold your hand through setting up this project and running your first Thermal-RGB Object Detection inference using the provided YOLO model. 

Don't worry if you aren't an expert programmer—just follow these steps in order.

---

## 💻 Step 1: Open Your Terminal (Command Line)
First, you need a terminal window open in the folder where your files are located (presumably `e:\NII_CU_MAPD_dataset`).

If you are using **VS Code**, you can open the terminal by clicking `Terminal` -> `New Terminal` in the top menu.

## 🐍 Step 2: Ensure Python is Installed & Create a Virtual Environment (venv)
You need Python to run this code. If you don't have it, download it from [python.org](https://www.python.org/downloads/). 
*(Make sure to check the box that says "Add Python to PATH" during installation).*

We highly recommend using a Python Virtual Environment to keep your packages isolated. In your terminal, run the following commands:
```bash
# Create the virtual environment called "env"
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```
*(Your terminal should now show `(.venv)` at the beginning of the prompt indicating it's active!)*

## 📦 Step 3: Install the Required Packages
We have provided a file called `requirements.txt` containing the list of libraries the code needs to run (like OpenCV, PyTorch, and Ultralytics).

In your terminal, copy and paste this command and press **Enter**:
```bash
pip install -r requirements.txt
```
*Note: Wait a few minutes for everything to install. It might download a few gigabytes of data if it's installing PyTorch for the first time.*

## 🚀 Step 4: Run Your First Inference!
Now for the fun part. You are going to use the `inference.py` script to fuse a regular camera picture with a thermal picture, and then our AI model (`best.pt`) will try to find the humans in it.

To do this, we've provided two sample images in your folder: `sample_rgb.jpg` and `sample_thermal.jpg`.

Copy and paste this exact command into your terminal and press **Enter**:
```bash
python inference.py --rgb sample_rgb.jpg --thermal sample_thermal.jpg
```

**What is happening?**
1. The script reads `sample_rgb.jpg` and `sample_thermal.jpg`.
2. It blends them together (70% standard color, 30% thermal heat).
3. It loads the AI brain (`best.pt`).
4. It spits out a brand new image called `inference_result.jpg` (or in some cases `result_sample.jpg` if you specify it).

## 🖼️ Step 5: View the Result
Look in your folder! You should see a newly created file called `inference_result.jpg`. 
Open it up. You will see the weirdly-colored fused image, and if the AI worked correctly, there will be colored boxes drawn around any people it found!

---

## 🛠️ Advanced Usage
Do you want to run this on your own images? No problem! Just change the names of the files in the command.

For example, if you have `my_camera.jpg` and `my_thermal.jpg`, you would run:
```bash
python inference.py --rgb my_camera.jpg --thermal my_thermal.jpg
```

If you want to save the final result under a specific catchy name, add `--out`:
```bash
python inference.py --rgb my_camera.jpg --thermal my_thermal.jpg --out "my_cool_result.jpg"
```
