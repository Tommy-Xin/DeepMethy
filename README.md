
DeepMeths
=========
DeepMeths: prediction of protein methylation sites with deep learning
Developer: XinCheng  from  Data Science and Big Data Technology, College of Software, Jilin University 

Requirement
=========
    keras==2.0.0
    numpy>=1.8.0
    backend==tensorflow

Related data information need to first load
=========
it is stored in the "./dataset/test_file.csv" 

The input file is an csv file, which includes , postion, sequences and labels

Predict for your test data
=========
If you want to use the model to predict your test data, you must prepared the test data as an csv file, the first col: postion, the second col: sequences 

The you can run the predict.py 

The results is an txt file,like:
"21"	"0.9999963"
"21"	"0.95067513"
"21"	"1.0669616e-24"
"21"	"3.7860446e-30"
"21"	"0.72186846"
"21"	"1.16561736e-07"
"21"	"1.8712221e-07"
"21"	"1.2668259e-24"

You can change the corresponding parameters in  main function prdict.py to choose to use the model to predict for general 
prediction

Train with your own data
=====
If you want to train your own network,your input file is an csv fie, while contains 3 columns:
label,  postion, sequence
label is 1 or 0 represents methylation and non-methylation site

Structure
==============

The methods folder contains dataprocess_predict.py, dataprocess_train.py, model_n.py, 



Contact
=========
Please feel free to contact us if you need any help: 2270315952@qq.com
