import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Konfiguracja parametrów (wybierz jeden z zestawów) ---

# Zestaw 1: "Coral" (Koralowiec) - powolny wzrost labiryntu
# F = 0.0545
# k = 0.0620

# Zestaw 2: "Mitosis" (Mitoza) - podział komórek / kropki
# F = 0.0367
# k = 0.0640

# Zestaw 3: Standardowy z PDF (złożone wzory)
F = 0.025
k = 0.055

# --- Inicjalizacja układu ---
N = 100
dx = 0.02
dt = 1.0
Du, Dv = 2e-5, 1e-5

u = np.ones((N, N))
v = np.zeros((N, N))

# Warunki początkowe (zakłócenie w środku)
N1, N2 = N // 4, 3 * N // 4
u[N1:N2, N1:N2] = 0.5 + np.random.rand(N2-N1, N2-N1) * 0.02
v[N1:N2, N1:N2] = 0.25 + np.random.rand(N2-N1, N2-N1) * 0.02

# --- Przygotowanie wykresu ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
# cmap='hot' lub 'viridis' lub 'Spectral' wygląda efektownie
im = ax.imshow(u, cmap='Spectral', interpolation='bicubic', vmin=0, vmax=1)
ax.set_title(f"Gray-Scott: F={F}, k={k}")
ax.axis('off')

# Tekst z licznikiem kroków
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')

# Funkcja aktualizująca klatkę animacji
steps_per_frame = 40  # Tyle kroków symulacji na jedną klatkę GIF-a

def update(frame):
    global u, v
    for _ in range(steps_per_frame):
        # Laplasjan (szybka wersja z np.roll)
        lu = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
              np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx**2)
        lv = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
              np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx**2)
        
        uvv = u * v**2
        du = Du * lu - uvv + F * (1 - u)
        dv = Dv * lv + uvv - (F + k) * v
        
        u += du * dt
        v += dv * dt
        
    im.set_array(u)
    time_text.set_text(f'Kroki: {frame * steps_per_frame}')
    return [im, time_text]

# --- Generowanie i zapisywanie animacji ---
# frames=200 -> 200 klatek * 40 kroków = 8000 kroków całkowitych
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Aby zapisać jako GIF, potrzebujesz 'pillow' (zazwyczaj zainstalowane) lub 'imagemagick'
# Jeśli wyskoczy błąd, zmień writer na 'pillow'
print("Generowanie GIF-a... to może chwilę potrwać.")
ani.save('gray_scott_simulation.gif', writer='pillow', fps=20)
print("Gotowe! Zapisano jako 'gray_scott_simulation.gif'.")
plt.show()