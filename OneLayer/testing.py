import csv
import random

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, stala_uczenia=0.01):
        # Im mniejsza stała uczenia, tym mniejsza konwergencja, ale dokładniejsze wyniki!
        self.stala_uczenia = stala_uczenia
        # Wagi na none, bo będą inicjalizowane podczas procesu uczenia
        self.wagi = None

    @staticmethod
    def wczytaj_dane(sciezka_pliku):
        # Do przechowywania danych z pliku
        dane = []
        # Zamieniamy nazwy języków na wartości liczbowe — łatwiej się na tym operuje
        jezyki_mapping = {"English": 0, "German": 1, "Polish": 2, "Spanish": 3}
        with open(sciezka_pliku, 'r', encoding='utf-8') as csvfile:
            # Czytanie z pliku
            reader = csv.reader(csvfile)
            # Iteracja po linijkach
            for linijka in reader:
                # Podział na język i tekst
                jezyk, tekst = linijka[0], linijka[1]
                # Pozbycie się niechcianych cudzysłowów
                tekst = tekst.strip('"')
                jezyk = jezyki_mapping[jezyk.strip().strip('"')]
                # Dodanie do danych do tablicy jako tuple
                dane.append((tekst, jezyk))
        return dane

    @staticmethod
    def wstepne_przetwarzanie(tekst):
        # Zamiana tekstu na małe litery
        tekst = tekst.lower()
        # Tablica, która będzie przechowywać ile razy dana litera wystąpiła
        liczebnosc_liter = np.zeros(26)
        for litera in tekst:
            # Patrzymy po kodach ASCII
            if 'a' <= litera <= 'z':
                # Dodawanie do liczby wystąpień
                liczebnosc_liter[ord(litera) - ord('a')] += 1

        # Chcemy uzyskać wektor, który zawsze będzie mieć długość - 1.
        # Pozwala to na zachowanie częstotliwości znaków, jednocześnie normalizując długość wektora.
        # Dzięki temu długość tekstu odbija się na wynikach.
        # Do tego używamy normy L2 - sqrt(suma_pierwiastków^2)
        normalizacja = liczebnosc_liter / np.linalg.norm(liczebnosc_liter)
        return normalizacja

    def aktywacja(self, wartosci):
        # dot — produkt wewnętrzny a1b1 + a2b2 ...
        return np.dot(self.wagi, wartosci)

    def zaklasyfikuj(self, wiersz):
        # Przygotowanie tablicy z wystąpieniami liter
        # Z dodaniem bias
        inputs = np.array([1] + list(self.wstepne_przetwarzanie(wiersz)))

        # Uzyskanie poziomu aktywacji
        aktywacje = self.aktywacja(inputs)

        # Aby klasyfikować język tekstu, wybieramy perceptron z maksymalną aktywacją.
        return np.argmax(aktywacje)

    def trenowanie(self, dane_treningowe, dane_testowe, liczba_epok=100, liczba_jezykow=4):

        # Inicjalizacja wag — dla liczby języków (w tym wypadku 4) po 26 liter

        # Ustawienie losowej wagi z zakresu od 0 do 1 - dla pierwszej z wag
        # Pozwala to na zachowanie względnej neutralności jednocześnie nie stojąc w lokalnym minimum
        self.wagi = np.array([[random.uniform(0, 1)] + [0.0] * 26 for _ in range(liczba_jezykow)])

        dokladnosci = []

        for epoka in range(liczba_epok):
            for wiersz, oczekiwany_output in dane_treningowe:
                # Wstępne przetwarzanie — dodajemy na start 1 do inputu — jako bias
                inputs = np.array([1] + list(self.wstepne_przetwarzanie(wiersz)))

                # Liczenie funkcji aktywacji
                uzyskany_output = self.aktywacja(inputs)
                # Ustawienie błędów na 0
                blad = np.zeros(liczba_jezykow)
                # Przypisanie 1 do języka, który miał wystąpić
                blad[oczekiwany_output] = 1
                # Obliczenie różnicy między oczekiwanym a uzyskanym inputem
                blad -= uzyskany_output
                # Aktualizacja wag perceptronu
                self.wagi += self.stala_uczenia * np.outer(blad, inputs)

            # Poprawności przypisania
            prawidlowe_przypisania = sum(1 for rzad, jezyk in dane_testowe if self.zaklasyfikuj(rzad) == jezyk)
            dokladnosc = prawidlowe_przypisania / len(dane_testowe)
            dokladnosci.append(dokladnosc)

        return dokladnosci


def main():
    stala_uczenia = float(input("Wprowadź stałą uczenia (dla najlepszych efektów wybierz małą stałą - rzędu 0.01): "))
    liczba_epok = int(input("Wprowadź liczbę epok: "))

    perceptron = Perceptron(stala_uczenia=stala_uczenia)
    dane_treningowe = perceptron.wczytaj_dane("lang.train.csv")
    dane_testowe = perceptron.wczytaj_dane("lang.test.csv")

    dokladnosci = perceptron.trenowanie(dane_treningowe, dane_testowe, liczba_epok=liczba_epok)

    # Wykres
    plt.plot(range(1, liczba_epok + 1), dokladnosci)
    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.title("Dokładność od liczby epok")
    plt.show()

    prawidlowe_przypisania = sum(1 for rzad, jezyk in dane_testowe if perceptron.zaklasyfikuj(rzad) == jezyk)

    dokladnosc = prawidlowe_przypisania / len(dane_testowe)
    print(f'Dokładność klasyfikacji: {dokladnosc}')

    mapping = {0: "Angielski", 1: "Niemiecki", 2: "Polski", 3: "Hiszpański"}

    while True:
        input_uzytkownika = input("Wpisz tekst do zaklasyfikowania, lub wpisz 'q' by wyjść: ")
        if input_uzytkownika.lower() == "q":
            break
        rezultat = perceptron.zaklasyfikuj(input_uzytkownika)
        print(f'Zaklasyfikowany jako język: {mapping[rezultat]}')


if __name__ == '__main__':
    main()
