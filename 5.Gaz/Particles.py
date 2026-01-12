import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio
import os
import glob
from tqdm import tqdm

particle_number = 16
box_size = 8.0
eps = 1.0
sigma = 1.0
radius = sigma / 2.0
dt = 0.0001
temp = 2.5
kB = 1.0
m = 1.0

box_half = box_size / 2.0
n_steps = 20000
plot_every = 100
img_dir = "frames"

rc = 2.5 * sigma
rc_sq = rc**2
sr6_c = (sigma / rc)**6
u_cutoff = 4.0 * eps * (sr6_c**2 - sr6_c)

class particle:
    def __init__(self, radius, position, velocity, mass):
        self.radius = radius
        self.r = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)
        self.m = mass
        self.f = np.zeros(2, dtype=float)

nx, ny = (4, 4)
x = np.linspace(1, 7, 4)
y = np.linspace(1, 7, 4)
xv, yv = np.meshgrid(x, y)
positions = np.column_stack((xv.ravel(), yv.ravel()))

list_of_particles = []
total_momentum = np.zeros(2)
total_ke = 0.0

for position in positions:
    velocity = np.random.uniform(-0.5, 0.5, 2)
    
    par = particle(radius, position, velocity, m)
    list_of_particles.append(par)
    
    total_momentum += par.v * par.m
    total_ke += 0.5 * par.m * np.dot(par.v, par.v)

print(f"Utworzono {len(list_of_particles)} cząstek.")

def remove_com_momentum(particles):
    global total_momentum
    v_mean = total_momentum / (len(particles) * m)
    print(f"Usuwanie prędkości środka masy: {v_mean}")
    for p in particles:
        p.v -= v_mean
    
    total_momentum = np.zeros(2)
    for p in particles:
        total_momentum += p.v * p.m

def scale_velocities(particles, target_temp):
    current_ke = 0.0
    for p in particles:
        current_ke += 0.5 * p.m * np.dot(p.v, p.v)
        
    target_ke = len(particles) * kB * target_temp
    
    fs = np.sqrt(target_ke / current_ke)
    print(f"Skalowanie prędkości, T_start = {current_ke / (len(particles) * kB):.2f}, fs = {fs:.3f}")
    
    for p in particles:
        p.v *= fs

remove_com_momentum(list_of_particles)
scale_velocities(list_of_particles, temp)

def calculate_forces(particles):
    total_pe = 0.0
    
    for p in particles:
        p.f = np.zeros(2)
        
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            p_i = particles[i]
            p_j = particles[j]
            
            dr = p_i.r - p_j.r
            
            if dr[0] > box_half: dr[0] -= box_size
            elif dr[0] < -box_half: dr[0] += box_size
            if dr[1] > box_half: dr[1] -= box_size
            elif dr[1] < -box_half: dr[1] += box_size
            
            r_sq = np.dot(dr, dr)
            
            if r_sq < rc_sq:
                sr2 = (sigma**2) / r_sq
                sr6 = sr2**3
                sr12 = sr6**2

                total_pe += (4.0 * eps * (sr12 - sr6)) - u_cutoff
                
                force_scalar_part = (48.0 * eps / r_sq) * (sr12 - 0.5 * sr6)
                force_vec = force_scalar_part * dr
                
                p_i.f += force_vec
                p_j.f -= force_vec
    
    return total_pe

def save_frame(particles, step, filename):
    plt.clf()
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlim((0, box_size))
    plt.ylim((0, box_size))
    
    for p in particles:
        cir = Circle((p.r[0], p.r[1]), radius=p.radius)
        ax.add_patch(cir)
        
    plt.title(f'Symulacja gazu Lennarda-Jonesa, krok {step:06d}')
    plt.savefig(filename)
    plt.close(fig)

def create_animation(img_dir, gif_name):
    if not os.path.exists(img_dir) or not os.listdir(img_dir):
        print("Nie znaleziono klatek do stworzenia animacji.")
        return
        
    filenames = sorted(glob.glob(os.path.join(img_dir, 'img*.png')))
    
    print(f"\nTworzenie animacji {gif_name} z {len(filenames)} klatek...")
    with imageio.get_writer(gif_name, mode='I', duration=0.05) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print(f"Animacja zapisana. Czyszczenie klatek PNG...")
    for f in filenames: os.remove(f)
    os.rmdir(img_dir)
    print("Gotowe.")

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

times = []
kinetic_energies = []
potential_energies = []
total_energies = []

print("Rozpoczynanie symulacji...")
print("Obliczanie początkowych sił...")

pe = calculate_forces(list_of_particles)

for p in list_of_particles:
    p.v -= (p.f / p.m) * (dt / 2.0)

for step in tqdm(range(n_steps)):
    
    total_ke = 0.0
    for p in list_of_particles:
        p.v += (p.f / p.m) * dt
        p.r += p.v * dt
        p.r = p.r % box_size
        total_ke += 0.5 * p.m * np.dot(p.v, p.v)

    pe = calculate_forces(list_of_particles)
    
    if step % plot_every == 0:
        frame_filename = os.path.join(img_dir, f'img{step:06d}.png')
        save_frame(list_of_particles, step, frame_filename)
        
        times.append(step * dt)
        kinetic_energies.append(total_ke)
        potential_energies.append(pe)
        total_energies.append(total_ke + pe)

print("Symulacja zakończona.")
print("Tworzenie wykresów energii...")

plt.figure(figsize=(10, 6))
plt.plot(times, kinetic_energies, label='Energia Kinetyczna (K)', color='orange')
plt.xlabel('Czas (t)')
plt.ylabel('Energia (E)')
plt.title('Ewolucja Energii Kinetycznej')
plt.legend()
plt.grid(True)
plt.savefig("energia_kinetyczna.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(times, potential_energies, label='Energia Potencjalna (U)', color='blue')
plt.xlabel('Czas (t)')
plt.ylabel('Energia (E)')
plt.title('Ewolucja Energii Potencjalnej')
plt.legend()
plt.grid(True)
plt.savefig("energia_potencjalna.png")
plt.close()

plt.figure(figsize=(12, 7))
plt.plot(times, total_energies, label='Energia Całkowita (E = K + U)', linewidth=2, color='red')
plt.plot(times, kinetic_energies, label='Energia Kinetyczna (K)', linestyle=':', alpha=0.7, color='orange')
plt.plot(times, potential_energies, label='Energia Potencjalna (U)', linestyle=':', alpha=0.7, color='blue')
plt.xlabel('Czas (t)')
plt.ylabel('Energia (E)')
plt.title('Ewolucja Energii Układu')
plt.legend()
plt.grid(True)
plt.savefig("energia_calkowita_ze_skladowymi.png")
plt.close()

create_animation(img_dir, "symulacja_z_kodu_uzytkownika.gif")