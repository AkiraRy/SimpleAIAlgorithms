import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wczytywanie pliku
data = pd.read_csv('data.csv', header=None)

# Dane bez ostatniej kolumny
input_data = data.iloc[:, :-1].values

def standaryzacja(dane):
    srednia = np.mean(dane, axis=0)
    odchylenie_standardowe = np.std(dane, axis=0)
    return (dane - srednia) / odchylenie_standardowe


# Standaryzacja
input_data = standaryzacja(input_data)

def algo(dane, k, max_l_iteracji):
    # Losowo wybieranie k centroidów
    centroidy = dane[np.random.choice(dane.shape[0], k, replace=False)]
    # print(centroidy)

    for i in range(max_l_iteracji):
        # Lista na nowe centroidy
        nowe_centroidy = []

        # Lista etykiet klastrów
        etykiety = []

        # Suma dystansów
        suma_dystansow = 0

        # Dla każdego przykładu — znajdź najbliższy centroid i dodaj do odpowiednich grup
        for przyklad in dane:
            # Nieskończoność na początek
            min_dystans = float('inf')
            # Wartość na początek
            index_najblizszy_centroid = -1

            # Znajdź najbliższy centroid
            for index, centroid in enumerate(centroidy):
                # Wyliczenie odległości między przykładem a centroidem
                dystans = np.linalg.norm(przyklad - centroid)

                # Zamieniamy aż znajdziemy najmniejszy
                if dystans < min_dystans:
                    min_dystans = dystans
                    index_najblizszy_centroid = index

            # Przypisz indeks najbliższego centroidu do etykiet
            etykiety.append(index_najblizszy_centroid)
            # Dodaj minimalny dystans do sumy dystansów
            suma_dystansow += min_dystans

        # Zamiana na NumPy array
        labels = np.array(etykiety)
        print(labels)

        # Kalkulacja nowego centroidu dla każdej grupy — średniej arytmetycznej wektorów w grupie
        for j in range(k):
            grupa = dane[labels == j]
            # Wylicz nowy centroid licząc średnią arytmetyczną
            nowy_centroid = grupa.mean(axis=0)
            # Dodanie do listy
            nowe_centroidy.append(nowy_centroid)

        # Zamiana na NumPy array
        nowe_centroidy = np.array(nowe_centroidy)

        print(f'Iteracja {i + 1}: {suma_dystansow:.2f}')

        # Sprawdzenie zbieżności
        if np.array_equal(nowe_centroidy, centroidy):
            break

        centroidy = nowe_centroidy

    return labels, centroidy


if __name__ == "__main__":
    ile_k = 3

    labels, centroidy = algo(input_data, ile_k, 100)

    # Składy grup
    grupy = pd.DataFrame({'labels': labels})
    procenty = grupy['labels'].value_counts().sort_index()

    print('\nSkłady grup:')
    print(procenty.to_string(header=None))

    # Wyciągnięcie nazw
    nazwy = data.iloc[:, -1].values
    print(nazwy)
    # Wyciągnięcie nazw i etykiet
    sklady_grup = pd.DataFrame({'Nazwy': nazwy, 'Labels': labels})
    print(sklady_grup)
    # Wyciągnięcie faktycznych składów grup
    czystosc_grup = sklady_grup.groupby('Labels')['Nazwy'].value_counts(normalize=True)

    print("\nMiary czystości grupy:")
    print(czystosc_grup.to_string(header=None))

    # WYKRES
    plt.scatter(input_data[:, 0], input_data[:, 1], c=labels, cmap='viridis', marker='o', s=50)
    plt.scatter(centroidy[:, 0], centroidy[:, 1], c='red', marker='x', s=100)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('k-means')

    plt.show()