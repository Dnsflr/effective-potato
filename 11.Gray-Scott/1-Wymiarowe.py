import numpy as np
import matplotlib.pyplot as plt

# --- Parametry symulacji [cite: 45, 47, 50] ---
L = 2.0
N = 100
dx = 0.02
dt = 1.0
T_max = 10000

# Współczynniki dyfuzji i reakcji [cite: 50]
Du = 2e-5
Dv = 1e-5
F = 0.025
k = 0.055

# --- Inicjalizacja tablic [cite: 52, 53] ---
u = np.ones(N)
v = np.zeros(N)

# Warunki początkowe (perturbacja w środku) [cite: 54, 55, 56, 57, 58]
N1 = N // 4
N2 = 3 * N // 4
# Dodajemy +1 do N2 w slicing, aby uwzględnić ostatni element (standard Pythona)
u[N1:N2+1] = 0.5 + np.random.rand(N2 - N1 + 1) * 0.02 # Uwaga: w PDF jest wzór 0.4 + ... * 0.2, tutaj lekko wygładzone dla stabilności lub wg wzoru:
u[N1:N2+1] = 0.4 + np.random.rand(N2 - N1 + 1) * 0.2
v[N1:N2+1] = 0.2 + np.random.rand(N2 - N1 + 1) * 0.2

# Tablica do przechowywania historii ewolucji u (do wykresu czasoprzestrzennego)
# Zapisujemy co 10 kroków, żeby wykres był czytelny i nie zajmował za dużo pamięci
history_u = []
save_interval = 10

# Momenty czasowe do narysowania przekrojów [cite: 63]
plot_times = [0, 500, 1500, 2500, 10000]
plots_data = {}

# --- Pętla symulacji ---
steps = int(T_max / dt)

for step in range(steps + 1):
    current_time = step * dt
    
    # Zapisz dane do wykresu liniowego w wybranych momentach
    if current_time in plot_times:
        plots_data[current_time] = (u.copy(), v.copy())
    
    # Zapisz historię do mapy ciepła
    if step % save_interval == 0:
        history_u.append(u.copy())

    # Obliczenie Laplasjanu 1D przy użyciu np.roll (periodyczne warunki brzegowe) [cite: 48, 137]
    # f''(x) ~ (f(x+h) - 2f(x) + f(x-h)) / h^2
    lap_u = (np.roll(u, 1) - 2*u + np.roll(u, -1)) / (dx**2)
    lap_v = (np.roll(v, 1) - 2*v + np.roll(v, -1)) / (dx**2)
    
    # Reakcja Graya-Scotta 
    # du/dt = Du*Lap_u - u*v^2 + F*(1-u)
    # dv/dt = Dv*Lap_v + u*v^2 - (F+k)*v
    uvv = u * v**2
    du = Du * lap_u - uvv + F * (1 - u)
    dv = Dv * lap_v + uvv - (F + k) * v
    
    # Aktualizacja metodą Eulera [cite: 47]
    u += du * dt
    v += dv * dt

# --- Wizualizacja ---

# 1. Wykresy u(x) i v(x) dla różnych chwil czasowych [cite: 63]
fig, axes = plt.subplots(len(plot_times), 1, figsize=(8, 12), sharex=True)
if len(plot_times) == 1: axes = [axes] # obsługa przypadku jednej klatki

for i, t in enumerate(plot_times):
    u_t, v_t = plots_data[t]
    axes[i].plot(u_t, label='u', color='blue')
    axes[i].plot(v_t, label='v', color='red')
    axes[i].set_title(f'Czas t = {t}')
    axes[i].set_ylabel('Stężenie')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

plt.xlabel('x (indeks)')
plt.tight_layout()
plt.show()

# 2. Wykres czasoprzestrzenny dla u 
plt.figure(figsize=(8, 6))
# history_u jest listą tablic, konwertujemy na macierz 2D
plt.imshow(history_u, aspect='auto', extent=[0, N, T_max, 0], cmap='viridis')
plt.colorbar(label='Stężenie u')
plt.xlabel('Położenie x')
plt.ylabel('Czas t')
plt.title('Ewolucja czasoprzestrzenna u(x,t)')
plt.show()