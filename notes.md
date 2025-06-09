# Zasoby
- https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- https://medium.com/@MalikSaadAhmed/text-generation-using-lstms-and-gpt-2-in-pytorch-8097c948ccd8
- Dataset https://huggingface.co/datasets/chirunder/text_messages
- https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- https://pl.eitca.org/sztuczna-inteligencja/eitc-ai-dltf-g%C5%82%C4%99bokie-uczenie-z-tensorflow/rekurencyjne-sieci-neuronowe-w-tensorflow/Przyk%C5%82ad-rnn-w-tensorflow/egzamin-przegl%C4%85d-rnn-przyk%C5%82ad-w-tensorflow/co-to-jest-kom%C3%B3rka-lstm-i-dlaczego-jest-u%C5%BCywana-w-implementacji-rnn/
- https://web.stanford.edu/~jurafsky/slp3/3.pdf

# Uruchamianie (mac)
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

# Treść

1. Cel projektu
2. Wstęp teoretyczny
  - Wstęp do Danych
  - Klasyczne podejścia w przetwarzaniu tekstu
  - Podejścia AI w przetwarzaniu tekstu
  - Wstęp teoretyczny do LSTM
3. Przygotowanie danych
  - Wybór zbioru gotowych do trenowania
  - Format danych zbioru
  - Przygotowanie zbioru do uczenia
  - Lematyzacja
  - Sposób kodowania
  - Reprezentacja danych treningowych
4. Model LSTM przewidujący następne słowo
  - Zasada działania
5. Trenowanie i testowanie modelu
  - Prototyp #1
  - Prototyp #2
  - Wersja końcowa
6. Podsumowanie
7. Bibliografia

# Cel projektu

TODO KACPER: przewidywanie tekstu, autouzupełnianie, nauka LSTM
finished

Głównym celem projektu było zbadanie i zaimplementowanie modelu do przewidywania następnego słowa w sekwencji tekstowej, z potencjalnym zastosowaniem w systemach autouzupełniania. Projekt skupiał się na wykorzystaniu architektury sieci neuronowej typu long Short-Term Memory (LSTM). Dodatkowym clem było praktyczne zapoznanie się z działaniem, trenowaniem oraz ewaluacją modeli LSTM w kontekście przetwarzania języka naturalnego.

# Wstęp teoretyczny

## Wstęp do Danych

W zależności od modelu, który piszemy, dane będą się różnić, ale cel pozostaje ten sam. Ważne jest rozszerzenie plików danych, z którego importujemy nasze dane, ponieważ dane importowane z internetu są zazwyczaj w skompresowanej formie (np .warc.gz, tar.gz czy .zip). Te pliki są konwertowane do rozszerzeń bardziej przyjaznych do przetwarzania tekstu jak .jsonl czy .parquet. Dane należy oczyścić, gdyż dane mogą zawierać znaki bardziej złożone niż w alfabecie łacińskim jak np. `ł`, `ó`, `ź`, czy też mogą mieć różną wielkość liter i trzeba to poprawić. W importowanych zasobach kluczowe jest pozbycie się wszelkich duplikatów dla poprawy szybkości modelu językowego oraz zapewnienia różnorodności generowanego tekstu. Pomaga to zapobieganiu nadmiernego dopasowania modelu dla powtarzalnej treści. Ten proces można zaimplementować trzema podejściami: dokładnym, rozmytym oraz semantycznej deduplikacji. 

## Klasyczne podejścia w przetwarzaniu tekstu

TODO KACPER finished

Przed erą zaawansowanych modeli głębokiego nauczania, przetwarzanie tekstu było oparte o metody statystyczne i algorytmiczne. Kluczowymi podejściami, szczególnie w kontekście modelowania języka i przewidywania tekstu, należą:

**N-gramy** - Sekwencje N kolejnych słów (lub znaków) w tekście. Modele oparte na N-gramach przewidują następne słowo na podstawie prawdopodobieństwa jego wystąpienia po sekwencji N-1 poprzedzających słów, obliczanego na podstawie częstotliwości w korpusie treningowym. Są proste obliczeniowo, ale mają trudności z uchwyceniem zależności długoterminowych i rzadkimi słowami.
**Modele Markowa** - Uogólnienie N-gramów, gdzie zakłada się, że prawdopodobieństwo wystąpienia danego stanu (np. słowa) zależy tylko od ograniczonej liczby poprzednich stanów (własność Markowa).
**Bag-of-Words (BoW)** - Reprezentacja tekstu jako nieuporządkowanego zbioru słów, zliczająca ich wystąpienia. Choć nie służy bezpośrednio do przewidywania sekwencji, jest podstawą wielu klasycznych technik klasyfikacji tekstu.
**TF-IDF (Term Frequency-Inverse Document Frequency)** - Metoda ważenia słów, która ocenia ich znaczenie w dokumencie w kontekście całego korpusu. Przydatna w wyszukiwaniu informacji i kategoryzacji, ale mniej bezpośrednio w generowaniu sekwencji.

Mimo swoich ograniczeń, te metody stanowiły fundament dla rozwoju bardziej zaawansowanych technik i wciąż znajdują zastosowanie w jakichś prostszych zadaniach.

## Podejścia AI w przetwarzaniu tekstu

**Dokładne** - skupia się na zidentyfikowaniu i usunięciu kompletnie identycznych dokumentów. To podejście generuje klucz dla każdego dokumentu oraz grupuje te dokumenty przez ich klucze do kubełków, tak by trzymać jeden dokument na kubełek. Zaletą takiego podejścia jest efektywność, szybkość oraz niezawodność, a wadą jest ograniczenie do wykrywania idealnego dopasowania do treści, co może spowodować ominięcie semantycznie porównywalnych dokumentów z drobnymi wariacjami. 

**Rozmyte** - adresuje prawie zduplikowane treści przy użyciu sygnatur MinHash i Locality-Sensitive Hashing (LSH). Proces wpierw wylicza klucze MinHash dla dokumentów, po czym używa LSH do grupowania podobnych dokumentów do kubełków. 1 dokument może należeć do więcej niż jednego kubełka. Następnie trzeba wyliczyć podobieństwo Jaccarda, czyli takie, które porównuje podobieństwo między dokumentami w tych samych kubełkach, porównując stopień wspólności tych elementów na przykład zbioru słów względem wszystkich unikalnych elementów w obu dokumentach. Bazując na tym podobieństwie przekształcamy macierz podobieństwa do grafu i identyfikujemy połączone komponenty w grafie. Dokumenty w połączonym komponencie są rozpatrywane jako rozmyte duplikaty, a następnie usuwane z datasetu. 

**Semantyczne** - reprezentuje najbardziej zaawansowane podejście wykorzystujące nowoczesne modele osadzania (embedding), które uchwytują znaczenie semantyczne danych, w połączeniu z technikami klasteryzacji do grupowania semantycznie podobnych treści. Badania wykazały, że deduplikacja semantyczna skutecznie zmniejsza rozmiar zbioru danych, jednocześnie utrzymując lub nawet poprawiając wydajność modelu. Jest szczególnie przydatna w wykrywaniu parafraz, tłumaczeń tego samego materiału oraz treści o identycznym znaczeniu. Aby dokonać deduplikacji semantycznej wpierw trzeba przekształcić każdy punkt danych na wektor za pomocą wstępnie wytrenowanego modelu. Grupujemy te wektory w k klastrów przy użyciu algorytmu k-średnich (k-means). Wewnątrz każdego takiego klastra obliczane są pary podobieństw cosinusowych. Każdej parze danych, której podobieństwo cosinusowe przekracza ustalony próg, przypisuje się status semantycznych duplikatów. Z każdej grupy semantycznych duplikatów w klastrze zachowuje się tylko jeden reprezentatywny punkt danych, reszta jest usuwana.

## Wstęp teoretyczny do LSTM

TODO Kacper dokończyć to

Sieci LSTM są specjalnym rodzajem RNN (Recurrent Neural Network), które zostały opracowane, aby znacznie lepiej radzić sobie z zapamiętywaniem przez długi czas. W standardowych sieciach problemem jest tendencja do "zapominania" informacji z wcześniejszych etapów sekwencji, gdy sekwencja stanie się długa. LSTM został stworzony po to, by poradzić sobie z tym problemem. Mają wbudowaną taką wewnętrzną "pamięć" zwaną stanem komórki (cell state) oraz specjalnych struktur kontrolujących przepływ informacji, zwykle zwanych bramkami.

Stan komórki jest główną linią pamięci, biegnącą przez całą sieć LSTM od jednego kroku przetwarzania sekwencji do następnego. Informacje na tej linii mogą być przechowywane, modyfikowane lub ususuwane w kontrolowany sposób. Mogą również przebiegać w dużej mierze niezmienione, co pozwala sieci "pamiętać" istotne rzeczy z odległej przeszłości sekwencji. Jest to istotna różnica pomiędzy LSTM a prostszymi RNN. 

Struktury kontrolujące (bramki) można podzielić na 3 typy: 
1. Mechanizm Zapominania (Forget Gate)
2. Mechanizm Wejściowy (Input Gate)
3. Mechanizm Wyjściowy (Output Gate)

Te struktury działają w powyższej kolejności. W bramce zapominania jest podejmowana decyzja, które informacje z poprzedniego stanu komórki powinny zostać zapomniane lub odrzucone. Następnie w bramce wejściowej LSTM decyduje, jakie nowe informacje z bieżącego fragmentu danych (np. aktualizowanie danego słowa) są na tyle ważne, by je zapisać w stanie komórki. Ten mechanizm składa się z dwóch części - wpierw identyfikuje, które wartości z nowych danych warto zaktualizować, a potem tworzy listę potencjalnych nowych informacji, które mogłyby zostać dodane. Łącząc te kroki pozwala nam selektywnie zaktualizować stan komórki o nowe istniejące dane. Na końcu za pomocą bramki wyjściowej na podstawie zaktualizowanego stanu komórki (która jest mieszanką starych i nowych informacji), LSTM decyduje, co powinno być wynikiem przetwarzania w bieżącym kroku. Ten wynik staje się również tzw. stanem ukrytym, który jest formą krótkoterminowej pamięci przekazywanej do następnego kroku przetwarzania i używanej przez mechanizmy kontrolne w kolejnym etapie. Mechanizm wyjściowy filtruje informacje ze stanu komórki, aby wygenerować użyteczne wyjście. 

# Przygotowanie danych

## Wybór zbioru

Do projektu wykorzystujemy dataset [chirunder/text_messages z Huggingface](https://huggingface.co/datasets/chirunder/text_messages).

## Format danych zbioru

Zbiór danych składa się z dwóch plików parquet, stworzonych z jednej kolumny `text`.
Zbiór zawiera 11.6 miliona zdań, kazdy jako osobny wiersz o długości od 2 do 3010 znaków.
Zdania są w języku angielskim. Zaczynają się wielką literą, a kończą kropką. Zdania są wiadomościami
tekstowymi pochodzącymi z komunikacji pomiędzy ludźmi. Do trenowania wykorzystujemy jeden z plików,
o wadze 281.7 MB.

## Przygotowanie zbioru do uczenia

Aby model mógł efektywnie się uczyć i sugerować poprawne podpowiedzi, surowe dane nalezy
przetworzyć. W tym celu usuwamy interpunkcję, liczby, zamieniamy słowa na małe litery. Czyścimy
zbiór z niechcianych podpowiedzi. Zbiór przetwarzamy za pomocą skryptu w napisanego w języku python.
Jest on zapisany w pliku `src/1_prepare_data.py`.

Dane przygotowujemy w następujący sposób:
1. Losowa część rekordów jest usuwana (w przypadku, gdy potrzebujemy mniejszy dataset do testów)
2. Tekst jest zamieniany na wyłącznie małe litery (lowercase)
3. Usuwane są znaki interpunkcyjne oraz liczby
4. Tekst jest "trymowany", usuwane są początkowe i końcowe spacje
5. Tekst jest dzielony na słowa w słowniku
6. Dokonujemy lematyzacji
8. Ze zdań tworzone są dane do trenowania - ciąg trzech słów mapowany jest do następnego wyrazu w zdaniu. Dane są zapisywane jako liczby, korzystając ze słownika.
9. Przypadki są zapisywane do pliku w formacie parquet, który umozliwia ładowanie danych do pamięci operacyjnej w częściach.
10. Metadane, czyli informacje o słowniku oraz lookup table słownika są zapisywane w formacie pickle, gdyz mogą być załadowane w całości ze względu na mały rozmiar.

## Lematyzacja, sposób kodowania

Nie chcemy, by model zajmował się rzadkimi słowami, gdyz zwiększa to złozoność modelu dając minimalne korzyści. Dlatego wybierane jest 10 000 najpopularniejszych słów, które chcemy podpowiadać. Pozostałe są zamieniane na token `<UNK>`
Tworzony jest słownik, tablica z 10 001 elementami, w której kazde słowo występuje tylko raz. Dzięki temu mozna przypisać kazdemu słowu liczbę będącą indeksem tego słowa w tablicy.

## Reprezentacja danych

Dane dzielimy na kilka plików:
- Plik słownika (vocab.pkl)
- Plik z danymi treningowymi

Plik słownika zawiera prostą strukturę z trzema danymi:
- Lookup table word to index - słownik mapujący słowo do liczby
- Lookup table index to word - słownik mapujący liczbę do słowa
- vocab_size - liczbę słów w słowniku

Dane są zapisywane w formacie pickle (pkl) za pomocą biblioteki słownika.

Plik z danymi treningowymi jest zapisywany w formacie parquet, który umozliwia ładowanie danych
do pamięci operacyjnej w kawałkach. Format danych jest następujący:
- Zbiór składa się z listy słowników
- Kazdy słownik zawiera 2 pola, x i y
- Pole x składa się z krotki (tuple) zawierającej 1-3 wyrazy poprzedzające następne słowo, jako liczba
- Pole y jest liczbą reprezentującą słowo następujące po ciągu

Przykładowo, dla słownika `ale = 0, ala = 1, ma = 2, kota = 3` i jedynej danej treningowej
`ale ala ma -> kota`, plik będzie zawierał następującą strukturę:
```
[
  { "x": (0, 1, 2), "y": 3 }
]
```

# Model LSTM przewidujący następne słowo

TODO KACPER opisać ze uzywamy pytorch

W ramach projektu model LSTM został zaimplementowany za pomocą frameworka PyTorch. 

```py
import torch
import torch.nn as nn
import torch.optim as optim

embedding_dim = 50
hidden_dim = 100

class LSTMWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        print(f"Initializing model with vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.fc(out[:, -1, :])  # Use output from last timestep
        return out
```

TODO KACPER o co chodzi w tym kodzie
to chyba jest finished, nie wiem 

Mamy klasę LSTMWordPredictor z konstruktorem modelu (__init__) i parametramy `vocab_size`, `embedding_dim`, `hidden_dim`. `vocab_size` oznacza liczbę całkowitą unikalnych słów w słowniku, na podstawie którego model będzie operował, `embedding_dim` jest wymiarem wektorów osadzeń (embeddings), gdzie każde słowo będzie reprezentowane przez gęsty wektor o tej długości. `hidden_dim` jest liczbą jednostek w warstwie ukrytej LSTM, definiuje "pojemność" pamięci modelu. W konstruktorze definiujemy 3 warstwy. Pierwszą z nich, czyli warstwa osadzenia (self.embedding) odpowiada za transformację indeksów słów (będących liczbami całkowitymi) na gęste wektory liczbowe, czyli osadzenia. Jest to fundamentalny krok, pozwalający modelowi na naukę semantycznych reprezentacji poszczególnych słów. Kolejną inicjalizowaną warstwą jest główna warstwa LSTM (self.lstm), która przetwarza sekwencje wektorów osadzeń (embeddingów) dostarczonych przez poprzednią warstwę. Argument `batch_first` informuje warstwę, że dane wejściowe będą miały wymiarowość, gdzie pierwszy wymiar to rozmiar batcha. Ostatnią warstwą jest warstwa w pełni połączona (liniowa) (self.fc), która ma za zadanie zmapowanie wyjścia z warstwy LSTM (które ma `hidden_dim` wymiarów) na wektor o rozmiarze `vocab_size`. Ten finalny wektor reprezentuje logity, czyli nieznormalizowane prawdopodobieństwa, dla każdego słowa w słowniku, wskazując, które z nich jest najbardziej prawdopodobne jako następne w sekwencji.

Metoda `forward` definiuje sposób, w jaki dane przepływają przez model podczas predykcji (tzw. forward pass). Na wejściu (x) otrzymuje tensor zawierający sekwencje indeksów słów. Najpierw dane te są przekazywane przez warstwę LSTM. Warstwa LSTM zwraca pełne wyjście dla każdego kroku czasowego oraz ostatni stan ukryty i stan komórki (które w tym konkretnym przypadku nie są bezpośrednio wykorzystywane do dalszej predykcji, ale są kluczowe dla wewnętrznego działania LSTM). Do przewidzenia następnego słowa wykorzystujemy jedynie wyjście LSTM z ostatniego kroku czasowego analizowanej sekwencji wejściowej. To wyjście jest następnie przekazywane przez warstwę w pełni połączoną, aby uzyskać wspomniany wcześniej rozkład prawdopodobieństwa nad całym słownikiem. Ostatecznie model zwraca te logity jako wynik swojej predykcji. 

# Trenowanie i testowanie modelu

## Prototyp #1

Prototyp #1 ma ponizsze parametry:
- 10% danych do treningu
- 5, a następnie 10 epok
- Wszystkie słowa, bez filtra najpopularniejszych
- learning_rate = 0.01, embedding_dim = 50, hidden_dim = 100
- Rozmiar: 65.8 MB (plik pth) + 2.5 MB (vocab.pkl)

Trenowanie jednej epoki trwało około 5 minut. Wyniki testów dla 5 epok są następujące:
```
Initializing model with vocab_size=108514, embedding_dim=50, hidden_dim=100
Input: we are going to --> be
Input: we are --> we are not going to be a good idea to be a
Input: the iphone --> the iphone is a good idea to be a little more than
Input: the --> the first time i have a problem with the new one
Input: i believe --> i believe that the car is a little more than a few
Input: not --> not sure if i can get a new thread to the
Input: i --> i will be able to get a good deal with the
Input: what were --> what were going to be the same thing that i have been
```

Natomiast po 10 epokach model zachowuje się następująco:
```
Initializing model with vocab_size=108514, embedding_dim=50, hidden_dim=100
Input: we are going to --> be
Input: we are --> we are not going to be a lot of time to get
Input: the iphone --> the iphone is a little more expensive and i am not sure
Input: the --> the best way to do it with the other side of
Input: i believe --> i believe that the guy who is a good thing to do
Input: not --> not lobbing the forum and i was thinking about the same
Input: i --> i think it will be a good looking car and i
Input: what were --> what were talking about the same thing i have to do it
```

Model poprawnie przewiduje wyrazy, które mogą następować po sobie. Utworzone ciągi nie tworzą
jednak logicznych zdań. Aby poprawić model, w następnej iteracji wykorzystamy pełny zbiór danych.
Obecny model generuje wyłącznie krótkie, popularne słowa. Jest to w naszym przypadku duza zaleta.
Aby model nie uczył się rzadkich wyrazów, które nie będą chętnie wybierane przez uzytkownikow,
postanawiamy trenować zbiór 10 000 najpopularniejszymi wyrazami, a resztę zastępować tokenem `<UNK>`.
Ta zmiana powinna takze znacząco zmniejszyć rozmiar modelu.

## Prototyp #2
Prototyp #2 ma ponizsze parametry:
- 100% danych do treningu
- 1 epoka
- 10 tysięcy najpopularniejszych słów + token `<UNK>`
- learning_rate = 0.01, embedding_dim = 50, hidden_dim = 100
- Rozmiar: 6.3 MB (plik pth) + 201 KB (vocab.pkl)

Trenowanie jednej epoki trwało około 43 minuty. Wyniki testów są następujące:
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

Generowane ciągi słów mają gramatyczny sens, ale model szybko "gubi się" i zatraca sens wypowiedzi.
Zdarzają się pętle (np. `the same thing is the same thing is that the same`).
Duzym problemem zdaje się mały kontekst danych, będący jedynie trzema ostatnimi słowami.
Zdaje się on powodować zdania bez sensu (np. `what were you looking for a good idea to do it and`).
Zwiększenie kontekstu do 6 czy 10 wyrazów znacznie zwiększy złozoność i czas uczenia się, dlatego
decydujemy się sprawdzić większą ilość epok przy niezmienionej długości kontekstu. Mamy nadzieję,
ze uda się wygenerować satysfakcjonujące wyniki bez zwiększenia kontekstu.

## Wersja końcowa
Wersja końcowa ma ponizsze parametry:
- 100% danych do treningu
- 5 epok
- 10 tysięcy najpopularniejszych słów + token `<UNK>`
- learning_rate = 0.01, embedding_dim = 50, hidden_dim = 100
- Rozmiar: 6.3 MB (plik pth) + 201 KB (vocab.pkl)

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
Efekty są juz zadowalające, przewidywany tekst ma poprawną składnię, a zachowywanie sensu wypowiedzi
jest porównywalne do prostych modeli autouzupełniania dostępnych w nowoczesnych urządzeniach
mobilnych. Nadal widoczne jest szybkie zapominanie, ale model działa znacznie lepiej niz w
prototypie #2. Kolejnym widocznym problemem jest naduzywanie przedimków (the, a), spójników (and),
przyimków czy krótkich słów (is, it, for). Są to zwroty często uzywane w krótkich wiadomościach
tekstowych. Model zdaje się preferować te słowa, gdyz występują nader często w danych testowych.
Gdybyśmy chcieli generować długie teksty byłby to duzy problem, który nalezałoby mitygować poprzez
odpowiednie dobranie danych i zmniejszenie częstotliwości występowania tych wyrazów. Jako ze
celem projektu jest podpowiadanie słów wiadomości tekstowych, problem ten jest w rzeczywistości
zaletą - uzytkownicy często wybierają krótkie wiadomości zawierające wiele tego typu słów.
Z tego powodu uznajemy model za dostatecznie dobry dla naszego przypadku.

# Podsumowanie

TODO KACPER

Projekt miał na celu zbudowanie i przetestowanie modelu opartego na architekturze LSTM do przewidywania następnego słowa w sekwencji, z myślą o zastosowaniach takich jak autouzupełnianie tekstu. Realizacja składała się z kilku kroków takich jak oczyszczenie danych tekstowych (w tym lematyzację i filtrowanie słownika), implementację modelu LSTM przy użyciu biblioteki PyTorch oraz iteracyjne trenowanie i testowanie różnych konfiguracji modelu.

Eksperymenty wykazały, że:
TODO eksperymenty wykazały (?)

# Bibliografia

## Pozycje zwarte

Tadeusiewicz R., Sieci Neuronowe, Kraków, 2008

(ibidem, strona)

## Pozycje internetowe

Jan K., Tytuł, https://example.com
Dostęp 01.01.1970
