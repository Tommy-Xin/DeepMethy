
DeepMethy
=========
DeepMethy: Prediction of Protein Methylation Sites with Deep Learning

Developer: XinCheng  from  Data Science and Big Data Technology, College of Software, Jilin University 

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

The methods folder contains train_BP(length-concat).py, predict_BP(length-concat).py.
The detailed model structure is shown in the train_BP(length-concat).py



Contact
=========
Please feel free to contact us if you need any help: xinhku168@gmail.com
