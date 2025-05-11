up: 
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 #install torch with cuda independently 
	pip3 install -r requirements.txt
	mkdir data
	mkdir experiments