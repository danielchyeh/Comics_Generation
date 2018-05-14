# Comics_Generation
MLDS2017 Project 4

Project Link: https://www.csie.ntu.edu.tw/~yvchen/f106-adl/A4

Conditional Generative Adversarial Network (CGAN) is implemented to automatically generate anime images based on the given constraints (ex: green hair, blue eyes).
## Dataset
Download Link: https://drive.google.com/drive/folders/1bXXeEzARYWsvUwbW3SA0meulCR3nIhDb. tag_clean.csv includes multiple tags (hair color, eyes color, etc..) for each anime image. 
## Quick start
1. Download Dataset from link above.

2. Run the shell script!
```
./run.sh [test file]
```
[test file] should be <testing_text.txt> in the main folder. You can modify content if want different result.
## Training
In image_generationv1.py, change mode = 1 (line 20) to be mode = 0, then do the Quick Start!
## Demo Results
Generated Samples will be like

![image](https://github.com/danielchyeh/Comics_Generation/blob/master/assets/samples.png)

