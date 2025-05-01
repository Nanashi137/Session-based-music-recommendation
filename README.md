**Setup enviroment:**
```
conda create -n <enviroment_name> python=3.11 #or use -p with a specific path
conda activate <enviroment_name> 
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 #install torch with cuda independently 
```
