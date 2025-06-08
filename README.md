# Zastosowanie sieci neuronowej do modelowania sekwencji w zadaniu przetwarzania języka naturalnego

## Uruchamianie (mac)
```
brew install git-lfs

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

git lfs install
git clone https://huggingface.co/datasets/chirunder/text_messages

source venv/bin/activate
python3 src/1_prepare_data.py
python3 src/2_train.py
python3 src/3_predict.py
```
