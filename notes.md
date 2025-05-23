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

# Wstęp o Danych

W zależności od modelu, który piszemy, dane będą się różnić, ale cel pozostaje ten sam. Ważne jest rozszerzenie plików danych, z którego importujemy nasze dane, ponieważ dane importowane z internetu są zazwyczaj w skompresowanej formie (np .warc.gz, tar.gz czy .zip). Te pliki są konwertowane do rozszerzeń bardziej przyjaznych do przetwarzania tekstu jak .jsonl czy .parquet. Dane należy oczyścić, gdyż dane mogą zawierać znaki bardziej złożone niż w alfabecie łacińskim jak np. `ł`, `ó`, `ź`, czy też mogą mieć różną wielkość liter i trzeba to poprawić. W importowanych zasobach kluczowe jest pozbycie się wszelkich duplikatów dla poprawy szybkości modelu językowego oraz zapewnienia różnorodności generowanego tekstu. Pomaga to zapobieganiu nadmiernego dopasowania modelu dla powtarzalnej treści. Ten proces można zaimplementować trzema podejściami: dokładnym, rozmytym oraz semantycznej deduplikacji. 

## Podejścia AI

**Dokładne** - skupia się na zidentyfikowaniu i usunięciu kompletnie identycznych dokumentów. To podejście generuje klucz dla każdego dokumentu oraz grupuje te dokumenty przez ich klucze do kubełków, tak by trzymać jeden dokument na kubełek. Zaletą takiego podejścia jest efektywność, szybkość oraz niezawodność, a wadą jest ograniczenie do wykrywania idealnego dopasowania do treści, co może spowodować ominięcie semantycznie porównywalnych dokumentów z drobnymi wariacjami. 

**Rozmyte** - adresuje prawie zduplikowane treści przy użyciu sygnatur MinHash i Locality-Sensitive Hashing (LSH). Proces wpierw wylicza klucze MinHash dla dokumentów, po czym używa LSH do grupowania podobnych dokumentów do kubełków. 1 dokument może należeć do więcej niż jednego kubełka. Następnie trzeba wyliczyć podobieństwo Jaccarda, czyli takie, które porównuje podobieństwo między dokumentami w tych samych kubełkach, porównując stopień wspólności tych elementów na przykład zbioru słów względem wszystkich unikalnych elementów w obu dokumentach. Bazując na tym podobieństwie przekształcamy macierz podobieństwa do grafu i identyfikujemy połączone komponenty w grafie. Dokumenty w połączonym komponencie są rozpatrywane jako rozmyte duplikaty, a następnie usuwane z datasetu. 

**Semantyczne** - reprezentuje najbardziej zaawansowane podejście wykorzystujące nowoczesne modele osadzania (embedding), które uchwytują znaczenie semantyczne danych, w połączeniu z technikami klasteryzacji do grupowania semantycznie podobnych treści. Badania wykazały, że deduplikacja semantyczna skutecznie zmniejsza rozmiar zbioru danych, jednocześnie utrzymując lub nawet poprawiając wydajność modelu. Jest szczególnie przydatna w wykrywaniu parafraz, tłumaczeń tego samego materiału oraz treści o identycznym znaczeniu. Aby dokonać deduplikacji semantycznej wpierw trzeba przekształcić każdy punkt danych na wektor za pomocą wstępnie wytrenowanego modelu. Grupujemy te wektory w k klastrów przy użyciu algorytmu k-średnich (k-means). Wewnątrz każdego takiego klastra obliczane są pary podobieństw cosinusowych. Każdej parze danych, której podobieństwo cosinusowe przekracza ustalony próg, przypisuje się status semantycznych duplikatów. Z każdej grupy semantycznych duplikatów w klastrze zachowuje się tylko jeden reprezentatywny punkt danych, reszta jest usuwana.

# Przygotowanie danych

Do projektu wykorzystujemy dataset [chirunder/text_messages z Huggingface](https://huggingface.co/datasets/chirunder/text_messages). Dane przygotowujemy w następujący sposób:
1. Losowa część rekordów jest usuwana (w przypadku, gdy potrzebujemy mniejszy dataset do testów)
2. Tekst jest zamieniany na wyłącznie małe litery (lowercase)
3. Usuwane są znaki interpunkcyjne oraz liczby
4. Tekst jest "trymowany", usuwane są początkowe i końcowe spacje
5. Tekst jest dzielony na słowa
6. Wybierane jest 10 000 najpopularniejszych słów, które zostają. Pozostałe są zamieniane na token `<UNK>`
7. Tworzony jest słownik, tablica z 10 001 elementami, w której kazde słowo występuje tylko raz. Dzięki temu mozna przypisać kazdemu słowu liczbę będącą indeksem tego słowa w tablicy.
8. Ze zdań tworzone są dane do trenowania - ciąg trzech słów mapowany jest do następnego wyrazu w zdaniu. Dane są zapisywane jako liczby, korzystając ze słownika.
9. Przypadki są zapisywane do pliku w formacie parquet, który umozliwia ładowanie danych do pamięci operacyjnej w częściach.
10. Metadane, czyli informacje o słowniku oraz lookup table słownika są zapisywane w formacie pickle, gdyz mogą być załadowane w całości ze względu na mały rozmiar.

# Wstęp do LSTM

LSTM jest ulepszoną wersją RNN (Recurrent Neural Network), która może utrzymywać zależności na długi okres czasu w danych sekwencyjnych.

# Model LSTM przewidujący następne słowo

## Prototyp #1

Prototyp #1 ma ponizsze parametry:
- 10% danych do treningu
- 6 epok
- Wszystkie słowa, bez filtra najpopularniejszych
- learning_rate = 0.01, embedding_dim = 50, hidden_dim = 100

Trenowanie jednej epoki trwało około 5 minut. Wyniki testów są następujące:
```
TODO
```

## Prototyp #2
Prototyp #2 ma ponizsze parametry:
- 100% danych do treningu
- 1 epoka
- 10 tysięcy najpopularniejszych słów + token `<UNK>`
- learning_rate = 0.01, embedding_dim = 50, hidden_dim = 100

Trenowaie jednej epoki trwało około 43 minuty. Wyniki testów są następujące:
```
Input: we are going to --> be
Input: we are --> we are not a fan of the car and i have a
Input: the iphone --> the iphone and i have a question about the car and i
Input: the --> the same thing is the same thing is that the same
Input: i believe --> i believe that the car is a good thing to do with
Input: not --> not a problem with the car and i have a few
Input: i --> i have a few questions and i have been looking for
Input: what were --> what were you looking for a good idea to do it and
```

## Wersja końcowa
Wersja końcowa ma ponizsze parametry:
- 100% danych do treningu
- 5 epok
- 10 tysięcy najpopularniejszych słów + token `<UNK>`
- learning_rate = 0.01, embedding_dim = 50, hidden_dim = 100

Trenowanie zajęło 5x 43 min = 3h 35 min. Wyniki testów są następujące:
```
Initializing model with vocab_size=10001, embedding_dim=50, hidden_dim=100
Input: we are going to --> be
Input: we are --> we are going to be a good one for the first time
Input: the iphone --> the iphone is a great idea of how much of a lot
Input: the --> the same thing as the other one is a good idea
Input: i believe --> i believe it is a good idea to me and i have
Input: not --> not a problem with the stock rom and the phone is
Input: i --> i have a few more pics of the new ones and
Input: what were --> what were you doing with the new one and the other one
```
Efekty są juz zadowalające, przewidywany tekst ma poprawną składnię, a sens jest porównywalny do
autouzupełniania dostępnego w nowoczesnych urządzeniach mobilnych.
