
<!--DeepMethy
=========
DeepMethy: Prediction of Protein Methylation Sites with Deep Learning

<!--Developer: XinCheng  from  Data Science and Big Data Technology, College of Software, Jilin University

Requirement
=========
    keras==2.0.0
    numpy>=1.8.0
    backend==tensorflow
    
Related data information need to first load
=========
The related data is stored in '/dataset/test_file.csv'.

The input file is an csv file, which includes , postion, sequences and labels

Predict for your test data
=========
To use the model for prediction on your test data, prepare a CSV file with two columns: position and sequence. 

You can then run predict.py to generate predictions. The output will be a text file with results in the format:

"21"	"0.9999963"
"21"	"0.95067513"
"21"	"1.0669616e-24"
"21"	"3.7860446e-30"
"21"	"0.72186846"
"21"	"1.16561736e-07"
"21"	"1.8712221e-07"
"21"	"1.2668259e-24"

You can modify parameters in the predict.py main function to customize the prediction process according to your needs.

Train with your own data
=====
If you want to train your own network,your input file is an csv fie, while contains 3 columns:
label,  postion, sequence
label is 1 or 0 represents methylation and non-methylation site.

If you are interested in adding more function to the code, welcome to show your talent!

Project Structure
==============

The methods folder contains train.py, predict.py.

The detailed model structure is shown in the train.py



Contact
=========
Please feel free to contact us if you need any help: xinhku168@gmail.com -->



⸻

DeepMethy

DeepMethy: A Deep Learning Model for Protein Methylation Site Prediction

This project provides a deep neural network framework for the prediction of arginine methylation sites in protein sequences using evolutionary features (BLOSUM62 & PSSM), convolutional layers, residual blocks, and attention mechanisms.

⸻

🧠 Requirements

Ensure the following packages are installed before running the code:
=========
	keras==2.0.0
	numpy>=1.8.0
	tensorflow (backend)


⸻

📁 Data Preparation

The input data should be stored in:

/dataset/test_file.csv

Data Format

	•	For Prediction: The CSV file should contain the following columns: position, sequence


	•	For Training: The CSV file should contain: label, position, sequence

	•	label = 1 for methylated sites
	•	label = 0 for non-methylated sites

Missing residues should be padded with "0" for sequences shorter than the required length.

⸻

🔍 Prediction

To perform predictions on your test dataset:

	1.	Prepare a CSV file with: position, sequence


	2.	Run: python predict.py



The output will be a .txt file with prediction scores:

"21"	"0.9999963"
"21"	"0.95067513"
"21"	"1.0669616e-24"
...

🔧 Customization

You can modify parameters in predict.py under the main() function to adjust:

	•	Input file path
	•	Output file name
	•	Threshold or model settings

⸻

🏋️‍♂️ Training with Your Own Data

To train the DeepMethy model from scratch:

	1.	Prepare a CSV file with: label, position, sequence


	2.	Run: python train.py



You can fine-tune hyperparameters and model architecture in train.py.

⸻

🧬 Model Architecture

	•	Evolutionary Features:
	•	BLOSUM62 Matrix
	•	PSSM Matrix
	•	Deep Learning Backbone:
	•	1D Convolutional Layers
	•	Residual Blocks
	•	Dense Connections
	•	Attention Mechanism
	•	Multi-window feature extraction
	•	Weighted loss for class imbalance

The complete structure is implemented in train.py under the /methods/ directory.

⸻

📂 Project Structure

	DeepMethy/
	│
	├── dataset/
	│   └── test_file.csv
	│
	├── methods/
	│   ├── train.py
	│   └── predict.py
	│
	├── README.md
	└── requirements.txt (optional)


⸻

📫 Contact

If you have any questions or suggestions, feel free to reach out:

<!--Xin Cheng
Email: xinhku168@gmail.com
-->
⸻


