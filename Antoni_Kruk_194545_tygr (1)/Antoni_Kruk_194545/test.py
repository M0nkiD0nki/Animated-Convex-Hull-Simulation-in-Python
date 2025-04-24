import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class Tygryski:
    def __init__(self, ilosc):
        self.ilosc = ilosc
        self.tygrysy = []
        self.parametry = []
        self.generuj_tygryski()

    def generuj_tygryski(self):
        self.tygrysy = np.random.uniform(-50, 50, size=(self.ilosc, 2))
        for _ in range(self.ilosc):
            alpha = np.random.uniform(0, 360)  # kierunek tygrysa
            beta = np.random.uniform(0, 90)  # kąt między małymi trójkątami
            gamma = np.random.uniform(10, 30)  # dodatkowe przesunięcie kątowe

            a = np.random.uniform(5, 10)  # długość przyprostokątnych małych trójkątów
            b = np.random.uniform(10, 20)  # długość ogona i skoku
            self.parametry.append((alpha, beta,gamma, a, b))


def iloczyn_wektorowy(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])


def algorytm_grahama(punkty):
    if len(punkty) < 3:
        return punkty
    punkt_O = min(punkty, key=lambda p: (p[1], p[0]))
    punkty = sorted(punkty, key=lambda p: (np.arctan2(p[1] - punkt_O[1], p[0] - punkt_O[0]),
                                           (p[0] - punkt_O[0]) ** 2 + (p[1] - punkt_O[1]) ** 2))
    stos = [punkt_O]
    for p in punkty:
        while len(stos) > 1 and iloczyn_wektorowy(stos[-2], stos[-1], p) <= 0:
            stos.pop()
        stos.append(p)
    return stos


def rysuj_trojkat(ax, punkt, alpha, beta, b):

    # Konwersja kątów na radiany
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta-15)

    # Górny wierzchołek (czubek trójkąta) – punkt tygryska
    x, y = punkt

    # Dolne wierzchołki podstawy trójkąta
    podstawa_srodek_x = x - b * np.cos(alpha_rad)
    podstawa_srodek_y = y - b * np.sin(alpha_rad)

    # Lewy i prawy dolny wierzchołek podstawy
    podstawa_lewa = (podstawa_srodek_x + b * np.tan(beta_rad) * np.cos(alpha_rad - np.pi / 2),
                     podstawa_srodek_y + b * np.tan(beta_rad) * np.sin(alpha_rad - np.pi / 2))

    podstawa_prawa = (podstawa_srodek_x + b * np.tan(beta_rad) * np.cos(alpha_rad + np.pi / 2),
                      podstawa_srodek_y + b * np.tan(beta_rad) * np.sin(alpha_rad + np.pi / 2))

    # Rysowanie trójkąta
    ax.fill([x, podstawa_lewa[0], podstawa_prawa[0]],
            [y, podstawa_lewa[1], podstawa_prawa[1]],
            color='blue', alpha=0.5)

    # Rysowanie linii wskazującej kierunek
    dlugosc_linii = 15  # Długość linii kierunkowej
    koniec_linii = (x + dlugosc_linii * np.cos(alpha_rad),
                    y + dlugosc_linii * np.sin(alpha_rad))
    ax.plot([x, koniec_linii[0]], [y, koniec_linii[1]], color='black', linestyle='--', linewidth=1)

    # Oznaczenie punktu tygryska
    ax.scatter([x], [y], color='red', zorder=5)

    # Zwrócenie współrzędnych wierzchołków
    return [punkt, podstawa_lewa, podstawa_prawa]


def rysuj_mniejsze_trojkaty(ax, punkt, alpha, beta, gamma, a):

    # Konwersja kątów na radiany
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Kierunki małych trójkątów
    kierunek1 = alpha_rad + gamma_rad + beta_rad / 2
    kierunek2 = alpha_rad - gamma_rad - beta_rad / 2

    # Wierzchołki pierwszego małego trójkąta
    x1 = punkt[0] + a * np.cos(kierunek1)
    y1 = punkt[1] + a * np.sin(kierunek1)
    x2 = punkt[0] + a * 0.6 * np.cos(kierunek1 + np.pi / 6)
    y2 = punkt[1] + a * 0.6 * np.sin(kierunek1 + np.pi / 6)

    # Wierzchołki drugiego małego trójkąta
    x3 = punkt[0] + a * np.cos(kierunek2)
    y3 = punkt[1] + a * np.sin(kierunek2)
    x4 = punkt[0] + a * 0.6 * np.cos(kierunek2 - np.pi / 6)
    y4 = punkt[1] + a * 0.6 * np.sin(kierunek2 - np.pi / 6)

    # Rysowanie małych trójkątów
    ax.fill([punkt[0], x1, x2], [punkt[1], y1, y2], color='green', alpha=0.5)
    ax.fill([punkt[0], x3, x4], [punkt[1], y3, y4], color='yellow', alpha=0.5)

    # Oznaczenie punktu tygryska (czubek małych trójkątów)
    ax.scatter([punkt[0]], [punkt[1]], color='red', zorder=5)

    return [punkt, (x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def rysuj_otoczke(ax, otoczka_punkty, frame):

    if len(otoczka_punkty) >= 3:
        # Obliczamy pełną otoczkę raz na podstawie zebranych punktów
        otoczka = algorytm_grahama(otoczka_punkty)

        # Rysujemy tylko te krawędzie, które już powstały
        for i in range(min(frame, len(otoczka))):
            p1 = otoczka[i]
            p2 = otoczka[(i + 1) % len(otoczka)]  # Kolejny punkt lub początek
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2)



def porusz_tygryskami(tygrysy, parametry, otoczka):

    def styczna_do_sciany(A, B):
        AB = B - A
        return np.arctan2(AB[1], AB[0])

    def odbicie(alpha, styczna):
        return (2 * styczna - alpha) % (2 * np.pi)

    epsilon = 1e-3  # Tolerancja dla błędów numerycznych
    granica_min, granica_max = -99.9, 99.9  # Bezpieczne granice

    for i in range(len(tygrysy)):
        x, y = tygrysy[i]
        alpha_rad = np.deg2rad(parametry[i][0])  # Kąt alfa w radianach
        b = parametry[i][4]  # Długość kroku

        # Wstępne przesunięcie
        dx = b / 2 * np.cos(alpha_rad)
        dy = b / 2 * np.sin(alpha_rad)
        nowe_x = x + dx
        nowe_y = y + dy

        kolizja = False

        # Sprawdzenie kolizji z granicami wykresu
        if nowe_x < granica_min or nowe_x > granica_max:
            alpha_rad = np.pi - alpha_rad  # Odbicie względem osi pionowej
            kolizja = True
        if nowe_y < granica_min or nowe_y > granica_max:
            alpha_rad = -alpha_rad  # Odbicie względem osi poziomej
            kolizja = True

        # Aktualizacja pozycji po odbiciu od granicy
        if kolizja:
            parametry[i] = (np.rad2deg(alpha_rad) % 360, *parametry[i][1:])
            nowe_x = x + b / 2 * np.cos(alpha_rad)
            nowe_y = y + b / 2 * np.sin(alpha_rad)

        # Sprawdzenie kolizji z otoczką
        for j in range(len(otoczka)):
            A = np.array(otoczka[j])
            B = np.array(otoczka[(j + 1) % len(otoczka)])
            AB = B - A
            normalna = np.array([-AB[1], AB[0]])
            styczna = styczna_do_sciany(A, B)

            # Obliczenie odległości od ściany
            punkt = np.array([nowe_x, nowe_y])
            odleglosc = np.dot(punkt - A, normalna) / np.linalg.norm(normalna)

            if odleglosc < epsilon:  # Kolizja z otoczką
                alpha_rad = odbicie(alpha_rad, styczna)
                parametry[i] = (np.rad2deg(alpha_rad) % 360, *parametry[i][1:])
                nowe_x = x + b / 2 * np.cos(alpha_rad)
                nowe_y = y + b / 2 * np.sin(alpha_rad)
                break  # Kolizja z otoczką ma priorytet

        # Finalna aktualizacja pozycji z ograniczeniem do granic
        nowe_x = np.clip(nowe_x, granica_min, granica_max)
        nowe_y = np.clip(nowe_y, granica_min, granica_max)
        tygrysy[i] = (nowe_x, nowe_y)


def update(frame, ax, tygryski, otoczka_punkty, otoczka_data):

    ax.clear()

    # Poruszanie tygryskami
    porusz_tygryskami(tygryski.tygrysy, tygryski.parametry, otoczka_data['punkty'])

    # Rysowanie otoczki
    if not otoczka_data['zamknieta']:
        # Zbieranie punktów z trójkątów
        nowe_punkty = []
        for i, tygrys in enumerate(tygryski.tygrysy):
            alpha, beta, gamma, a, b = tygryski.parametry[i]

            # Duży trójkąt
            punkty_trojkata = rysuj_trojkat(ax, tygrys, alpha, beta, b)
            for punkt in punkty_trojkata:
                if -100 <= punkt[0] <= 100 and -100 <= punkt[1] <= 100:
                    nowe_punkty.append(punkt)

            # Małe trójkąty
            punkty_malych = rysuj_mniejsze_trojkaty(ax, tygrys, alpha, beta, gamma, a)
            for punkt in punkty_malych:
                if -100 <= punkt[0] <= 100 and -100 <= punkt[1] <= 100:
                    nowe_punkty.append(punkt)

        # Punkty, które mieszczą się w granicach
        otoczka_punkty.extend(nowe_punkty)

        # Obliczanie otoczki wypukłej
        if len(otoczka_punkty) >= 3:
            otoczka = algorytm_grahama(otoczka_punkty)

            # Zamknięcie otoczki, jeśli jeszcze nie jest zamknięta
            if not np.allclose(otoczka[0], otoczka[-1]):
                otoczka.append(otoczka[0])

            otoczka_data['punkty'] = otoczka

            # Krokowa animacja otoczki
            max_krawedzi = min(frame, len(otoczka) - 1)
            for i in range(max_krawedzi):
                p1 = otoczka[i]
                p2 = otoczka[i + 1]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2)

            # Jeśli cała otoczka jest narysowana, to jest zamknięta
            if max_krawedzi == len(otoczka) - 1:
                otoczka_data['zamknieta'] = True
                otoczka_data['punkty'] = otoczka.copy()

    else:
        # Rysowanie stałej otoczki
        for i in range(len(otoczka_data['punkty']) - 1):
            p1 = otoczka_data['punkty'][i]
            p2 = otoczka_data['punkty'][i + 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2)

    # Rysowanie tygrysków
    for i, tygrys in enumerate(tygryski.tygrysy):
        alpha, beta, gamma, a, b = tygryski.parametry[i]
        rysuj_trojkat(ax, tygrys, alpha, beta, b)
        rysuj_mniejsze_trojkaty(ax, tygrys, alpha, beta, gamma, a)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect("equal")


def main():
    fig, ax = plt.subplots()
    tygryski = Tygryski(10)
    wszystkie_punkty = []
    otoczka_data = {'punkty': [], 'zamknieta': False}

    anim = FuncAnimation(fig, update, fargs=(ax, tygryski, wszystkie_punkty, otoczka_data), interval=50)

    plt.show()



if __name__ == "__main__":
    main()