import math
import os
import pandas as pd
from graphviz import Digraph
dataset = pd.read_csv('dane.csv', sep=';')

# jaką nazwę w tabeli ma zmienna, którą będziemy określać
GOAL = "Paliwo"

#
# Oblicz entropie dla danej tabeli
# W tym konkretnym przypadku dana jest wzorem:
# E = -(a/n)log2(a/n) - (b/n)log2(b/n) - (c/n)log2(c/n) 
# gdzie:
# E - entropia dla danej tabeli
# a - liczba silników benzynowych
# b - liczba silników diesla
# c - liczba silników elektrycznych
# n - łączna liczba wszystkich obserwacji w tabeli
#
# Sposób zliczania:
# Zlicz liczbę poszczególnych typów silników i zapisz je w DataFrame
# Iteruj po tabeli i podstawiaj dane do wzrou podanego powyżej
# Zwróć zliczoną entropie
#

def calc_entropy_table(dataset):
  paliwa_poziom = dataset[GOAL].value_counts()
  paliwa_suma = paliwa_poziom.sum()
  entropy = 0
  for index in paliwa_poziom.index:
    entropy += -(paliwa_poziom[index]/paliwa_suma)*math.log2(paliwa_poziom[index]/paliwa_suma)
  return entropy


#
# Oblicz entropie w zależności od podanego atrybutu
# W tym konkretnym przypadku dana jest wzorem
# S = a/n * I+ + b/n * I-
# I+ = -(c1/a)log2(c1/a) - (c2/a)log2(c2/n) ... 
# 
# gdzie:
# a - liczba przypadków spełniających warunek
# b - liczba przypadków niespełniających warunku
# n - łączna liczba wszystkich przypadków
#

def calc_entropy_attribute(dataset, attribute):
  
  # zlicz ile jest obserwacji spełnia warunek a ile nie
  counted_values = dataset[attribute].value_counts()

  # ze wszystkich danych wybierz kolumny z omawianym atrybutem równym yes oraz wartością objaśnianą
  positive_cases = dataset[[attribute, GOAL]].loc[dataset[attribute] == 'yes']
  # analogicznie z no
  negative_cases = dataset[[attribute, GOAL]].loc[dataset[attribute] == 'no']

  def calc_entropy_condition(condition_set):
    
    # weź dataframe z pozliczanymi wartościami zmiennej objaśnianej pod określonym warunkiem: np osoba jest kobietą
    # i na jego podstawie oblicz entropie dla podanego warunku
    entropy = 0
    for index in condition_set.index:
     entropy += -(condition_set[index]/condition_set.sum())*math.log2(condition_set[index]/condition_set.sum())
    return entropy
  
  # suma spełniających warunek
  # wynika z prezentacji na wzór funckji
  if "yes" in counted_values:
    entropy_attribute = (counted_values["yes"]/counted_values.sum()) * calc_entropy_condition(positive_cases[GOAL].value_counts())
  else:
    entropy_attribute = 0

  #suma niespełniających warunku
  if "no" in counted_values:
    entropy_attribute += (counted_values["no"]/counted_values.sum()) * calc_entropy_condition(negative_cases[GOAL].value_counts())
  else:
    entropy_attribute += 0

  return entropy_attribute


class Node():

  idCounter = 65
  # funkcja służąca do budowy drzewa od danego węzła w dół
  # wywołuje się rekurencyjnie
  def build_tree(self):
    # dla każdego z węzłów oblicz jaka jest entropia na dla danej tabeli
    self.calc_self()
    # jeżeli entropia wynosi 0 oznacza to że w danej tabeli występują już tylko obiekty pojedynczej kategorii
    # w tym przypadku na przykład same silniki diesla
    # jest to warunek zakończenia rekurencji
    if self.entropy_table == 0:
      # doddaj do węzła wartość jaka została przewidziana
      self.prediction = self.dataset["Paliwo"].unique()[0]

      dot.node(chr(self.ID), self.prediction)
      return
    # jeżeli entropia nie jest zerem oznacza to, że tabela może zostać dalej podzielona, a węzeł nie jest węzłem ostatnim
    else:
      # stwórz węzłowi dzieci
      self.build_children()
      dot.node(chr(self.ID), self.attr_with_highest_value)
      # wywoładj tą samą metodę dla dziecka lewego
      self.childL.build_tree()
      # oraz prawego
      self.childR.build_tree()

  def calc_self(self):
    # oblicz entropię tabeli dla danego elementu
    self.entropy_table = calc_entropy_table(self.dataset)
    
    # dla każdego z możliwych warunków oblicz wartość wyrażenia
    # E - I
    # gdzie:
    # E jest entropią całej tabeli (tj bez warunku)
    # I jest entropią pod danym ograniczeniem
    #
    # i zapisz je w postać klucz, wartość (indeks, różnica E i I) jako tabela entropies
    #
    entropies = [(index ,self.entropy_table - calc_entropy_attribute(self.dataset, index)) for index in self.dataset.columns if not index == GOAL]

    #
    # z powyższej tabeli wyszukaj taki indeks dla którego różnica jest największa
    # wg algorytmu ID3 Quinna podział według tego indeksu będzie niósł ze sobą najwięcej informacji
    # zapisz ten indeks jako attr_with_highest_value
    #
    self.attr_with_highest_value = max(entropies, key = lambda i : i[1])[0]


  def build_children(self):
    # jeżeli entropia tablicy (E) była rózna od zera, będziemy dzielić dalej
    # podziel tabele według uprzednio obliczonego indeksu, który podzieli tabele na najbardziej informatywne podgrupy
    # wybierz pod tabele omawianego zbioru tak żeby zawierała tylko obserwacje spełniające warunek opisany przez indeks
    table_splitted_by_attr_1 = self.dataset.loc[self.dataset[self.attr_with_highest_value] == 'yes']
    # utwórz dziecko lewe, dla którego zbiorem odniesienia będzie utworzona wyżej tabela
    self.childL = Node(table_splitted_by_attr_1)

    #analogicznie z obserwacjami nie spełniającymi i dzieckiem prawym
    table_splitted_by_attr_2 = self.dataset.loc[self.dataset[self.attr_with_highest_value] == 'no']
    self.childR = Node(table_splitted_by_attr_2)

    dot.node(chr(self.childL.get_id()))
    dot.node(chr(self.childR.get_id()))
    dot.edge(chr(self.get_id()), chr(self.childL.get_id()), label="TAK")
    dot.edge(chr(self.get_id()), chr(self.childR.get_id()), label="NIE")


  def get_id(self):
    return self.ID

  def __init__(self, dataset):
    # konstruktor
    # przypisz nowoutworzonemu węzłowi jego tabelę odniesienia w tym przypadku:
    # tablicą odniesienia lewego dziecka węzła początkowego będzie tabela samych kobiet
    self.dataset = dataset
    
    # węzeł domyślnie nie ma dzieci, otrzyma je dopiero gdy po obliczeniu entropii będziemy dzielić tablicę
    self.childL = None
    self.childR = None
    
    self.ID = Node.idCounter
    Node.idCounter += 1



# utwórz węzeł początkowy
# w naszym przypadku tabelą odniesienia są wczytane dane z pliku dane.csv
root = Node(dataset)


dot = Digraph(comment="Binarne drzewo decyzyjne")

# wywołuje rekurencyjną budowę drzewa na węźle początkowym
# algorytm prawdopodobniej nie jest zaimplementowany w najlepszy możliwy sposób, a liczenie drzewa chwilę trwa
# prosimy o cierpliwość :)
root.build_tree()

# dziękuję za uwagę
# miłego wieczorku

#w celu wyświetlenia grafu proszę skopiować zawartość pliku lub wydruk konsoli i wkleić do okienka na stronie podanej poniżej
# https://dreampuf.github.io/GraphvizOnline/

with open("graph.txt", "w") as f:
  f.write(dot.source)
print(dot.source)
print("beep boop")