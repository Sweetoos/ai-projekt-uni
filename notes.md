# Zasoby
- https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- https://medium.com/@MalikSaadAhmed/text-generation-using-lstms-and-gpt-2-in-pytorch-8097c948ccd8
- Dataset https://huggingface.co/datasets/chirunder/text_messages
- https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Uruchamianie (mac)
```
brew install git-lfs

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/datasets/chirunder/text_messages

source venv/bin/activate
python3 src/1_prepare_data.py
python3 src/2_train.py
python3 src/3_predict.py
```

# Treść

## Wstęp

- Klasyczne podejścia w przetwarzaniu tekstu
- Podejścia AI
- Wstęp teoretyczny LSTM

## Dane

- Opis danych
- Lematyzacja
- Reprezentacja danych
- Sposób kodowania

# Klasyczne podejścia w przetwarzaniu tekstu

W zależności od modelu, który piszemy, dane będą się różnić, ale cel pozostaje ten sam. Ważne jest rozszerzenie plików danych, z którego importujemy nasze dane, ponieważ dane importowane z internetu są zazwyczaj w skompresowanej formie (np .warc.gz, tar.gz czy .zip). Te pliki są konwertowane do rozszerzeń bardziej przyjaznych do przetwarzania tekstu jak .jsonl czy .parquet. Dane należy oczyścić, gdyż dane mogą zawierać znaki bardziej złożone niż w alfabecie łacińskim jak np. `ł`, `ó`, `ź`, czy też mogą mieć różną wielkość liter i trzeba to poprawić. W importowanych zasobach kluczowe jest pozbycie się wszelkich duplikatów dla poprawy szybkości modelu językowego oraz zapewnienia różnorodności generowanego tekstu. Pomaga to zapobieganiu nadmiernego dopasowania modelu dla powtarzalnej treści. Ten proces można zaimplementować trzema podejściami: dokładnym, rozmytym oraz semantycznej deduplikacji. 

# Podejścia AI

**Dokładne** - skupia się na zidentyfikowaniu i usunięciu kompletnie identycznych dokumentów. To podejście generuje klucz dla każdego dokumentu oraz grupuje te dokumenty przez ich klucze do kubełków, tak by trzymać jeden dokument na kubełek. Zaletą takiego podejścia jest efektywność, szybkość oraz niezawodność, a wadą jest ograniczenie do wykrywania idealnego dopasowania do treści, co może spowodować ominięcie semantycznie porównywalnych dokumentów z drobnymi wariacjami. 

**Rozmyte** - adresuje prawie zduplikowane treści przy użyciu sygnatur MinHash i Locality-Sensitive Hashing (LSH). Proces wpierw wylicza klucze MinHash dla dokumentów, po czym używa LSH do grupowania podobnych dokumentów do kubełków. 1 dokument może należeć do więcej niż jednego kubełka. Następnie trzeba wyliczyć podobieństwo Jaccarda, czyli takie, które porównuje podobieństwo między dokumentami w tych samych kubełkach, porównując stopień wspólności tych elementów na przykład zbioru słów względem wszystkich unikalnych elementów w obu dokumentach. Bazując na tym podobieństwie przekształcamy macierz podobieństwa do grafu i identyfikujemy połączone komponenty w grafie. Dokumenty w połączonym komponencie są rozpatrywane jako rozmyte duplikaty, a następnie usuwane z datasetu. 

**Semantyczne** - reprezentuje najbardziej zaawansowane podejście wykorzystujące nowoczesne modele osadzania (embedding), które uchwytują znaczenie semantyczne danych, w połączeniu z technikami klasteryzacji do grupowania semantycznie podobnych treści. Badania wykazały, że deduplikacja semantyczna skutecznie zmniejsza rozmiar zbioru danych, jednocześnie utrzymując lub nawet poprawiając wydajność modelu. Jest szczególnie przydatna w wykrywaniu parafraz, tłumaczeń tego samego materiału oraz treści o identycznym znaczeniu. Aby dokonać deduplikacji semantycznej wpierw trzeba przekształcić każdy punkt danych na wektor za pomocą wstępnie wytrenowanego modelu. Grupujemy te wektory w k klastrów przy użyciu algorytmu k-średnich (k-means). Wewnątrz każdego takiego klastra obliczane są pary podobieństw cosinusowych. Każdej parze danych, której podobieństwo cosinusowe przekracza ustalony próg, przypisuje się status semantycznych duplikatów. Z każdej grupy semantycznych duplikatów w klastrze zachowuje się tylko jeden reprezentatywny punkt danych, reszta jest usuwana. 

# Wstęp do LSTM

LSTM jest ulepszoną wersją RNN (Recurrent Neural Network), która może utrzymywać zależności na długi okres czasu w danych sekwencyjnych.
