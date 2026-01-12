import numpy as np
import matplotlib.pyplot as plt

# --- Parametry symulacji 2D [cite: 70] ---
N = 100
dx = 0.02
dt = 1.0
T_max = 10000  # Czas symulacji

# Parametry modelu (można zmieniać wg )
# Przykładowy ciekawy zestaw (spot replication / coral):
Du = 2e-5
Dv = 1e-5
F = 0.025   # Możesz zmienić na np. 0.025, 0.04 etc.
k = 0.055   # Możesz zmienić na np. 0.055, 0.06 etc.

# --- Inicjalizacja tablic 2D [cite: 73, 74] ---
u = np.ones((N, N))
v = np.zeros((N, N))

# Warunki początkowe - kwadrat szumu w środku [cite: 75, 76, 77, 78, 79]
N1 = N // 4
N2 = 3 * N // 4
# Perturbacja
u[N1:N2+1, N1:N2+1] = 0.4 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
v[N1:N2+1, N1:N2+1] = 0.2 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2

# --- Pętla symulacji ---
steps = int(T_max / dt)
print(f"Symulacja 2D: F={F}, k={k}. Liczba kroków: {steps}")

for step in range(steps + 1):
    # Obliczenie Laplasjanu 2D przy użyciu np.roll [cite: 138, 139]
    # Suma sąsiadów (góra, dół, lewo, prawo) - 4*środek
    # roll(axis=0) przesuwa wiersze (góra/dół), roll(axis=1) przesuwa kolumny (lewo/prawo)
    lap_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx**2)
             
    lap_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
             np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx**2)
    
    # Reakcja 
    uvv = u * v**2
    du = Du * lap_u - uvv + F * (1 - u)
    dv = Dv * lap_v + uvv - (F + k) * v
    
    # Aktualizacja metodą Eulera
    u += du * dt
    v += dv * dt
    
    # Opcjonalnie: prosty pasek postępu
    if step % 1000 == 0:
        print(f"Krok {step}/{steps}...")

# --- Wizualizacja końcowego stanu [cite: 113-120] ---
plt.figure(figsize=(8, 8))
# Używamy interpolacji 'nearest' i limitów 0-1 zgodnie z instrukcją
im = plt.imshow(u, cmap='jet', interpolation='nearest', vmin=0, vmax=1) # 'jet' lub 'spectral' dobrze oddaje kolory z PDF
plt.colorbar(im, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
plt.title(f'Układ Graya-Scotta 2D (T={T_max})\nF={F}, k={k}')
plt.axis('off') # Ukrycie osi dla czystszego obrazu
plt.show()