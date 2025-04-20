# Zasoby
- https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- https://medium.com/@MalikSaadAhmed/text-generation-using-lstms-and-gpt-2-in-pytorch-8097c948ccd8
- Dataset https://huggingface.co/datasets/chirunder/text_messages

# Uruchamianie (mac)
```
brew install git-lfs

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/datasets/chirunder/text_messages

python3 src/1_prepare_data.py
python3 src/2_train.py
python3 src/3_predict.py
```
