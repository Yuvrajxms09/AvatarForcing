pip install gdown

gdown 1a0ZqX9GMrkb3JFQgOBmUB0V-Bpwtv4Hp -O pretrained_dir
gdown 1lchvHiDka0bBQ29bdY6lHxCyNGbGzvS_ -O pretrained_dir

pip install huggingface_hub
hf download facebook/wav2vec2-base-960h --local-dir pretrained_dir/wav2vec2-base-960h