import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


EPSILON = 1.0
SIGMA = 1.0
MASS = 1.0
CUTOFF = 2.5 * SIGMA

class LJSimulation:
    def __init__(self, N, density, T_target, dt):
        self.N = N
        self.L = np.sqrt(N / density)  
        self.dt = dt
        self.T_target = T_target
        

        side = int(np.ceil(np.sqrt(N)))
        spacing = self.L / side
        x = np.linspace(spacing/2, self.L - spacing/2, side)
        xv, yv = np.meshgrid(x, x)
        self.pos = np.column_stack((xv.flatten()[:N], yv.flatten()[:N]))
        

        self.vel = np.random.rand(N, 2) - 0.5

        current_T = np.sum(self.vel**2 * MASS) / (2 * self.N)
        scale_factor = np.sqrt(T_target / current_T)
        self.vel *= scale_factor
        

        self.forces = np.zeros((N, 2))
        self.virial = 0.0
        self.calc_forces()

    def apply_pbc(self, positions):

        return np.mod(positions, self.L)

    def min_image(self, dr):

        return dr - self.L * np.round(dr / self.L)

    def calc_forces(self):

        self.forces[:] = 0
        self.virial = 0.0
        pot_energy = 0.0
        
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = self.min_image(self.pos[i] - self.pos[j])
                r2 = np.sum(dr**2)
                
                if r2 < CUTOFF**2:
                    r2_inv = 1.0 / r2
                    r6_inv = r2_inv**3
                    
                    f_over_r = 48 * EPSILON * (r6_inv**2 * r2_inv - 0.5 * r6_inv * r2_inv)
                    
                    f_vec = f_over_r * dr
                    self.forces[i] += f_vec
                    self.forces[j] -= f_vec
                    
                    
                    self.virial += np.sum(f_vec * dr)
                    pot_energy += 4 * EPSILON * (r6_inv**2 - r6_inv)
                    
        return pot_energy

    def step_brown_clarke(self):
        v_u = self.vel + (self.forces / MASS) * (self.dt / 2)
        
        kin_energy_u = 0.5 * MASS * np.sum(v_u**2)
        T_inst = kin_energy_u / self.N  # Dla 2D (kB=1)
        
        if T_inst == 0: T_inst = 1e-6 # Zabezpieczenie
        eta = np.sqrt(self.T_target / T_inst)

        v_next_half = (2 * eta - 1) * self.vel + eta * (self.forces / MASS) * self.dt
        
        self.vel = v_next_half
        
        self.pos = self.pos + self.vel * self.dt
        self.pos = self.apply_pbc(self.pos)
        

        pe = self.calc_forces()
        

        ke = 0.5 * MASS * np.sum(self.vel**2)
        temp = ke / self.N 

        area = self.L ** 2
        pressure = (self.N / area) * temp + self.virial / (2 * area)
        
        return temp, pressure, ke, pe



def run_simulation(mode='graph'):

    
    if mode == 'solid':
        rho = 0.25   
        temp = 0.1    
        title = "Ciało Stałe (Solid)"
    elif mode == 'fluid':
        rho = 0.25     
        temp = 0.7    
        title = "Płyn (Fluid)"
    else:
        rho = 0.4
        temp = 0.5
        title = "Symulacja"

    N = 36          
    dt = 0.01       
    steps = 1000    
    
    sim = LJSimulation(N, rho, temp, dt)
    

    t_vals, T_vals, P_vals, E_tot_vals = [], [], [], []
    
    if mode == 'graph':
 
        print(f"Symulacja: {title}, N={N}, Rho={rho}, T={temp}")
        sim.dt = 0.025 
        
        for step in range(2000):
            T, P, K, U = sim.step_brown_clarke()
            t_vals.append(step * sim.dt)
            T_vals.append(T)
            P_vals.append(P)
            E_tot_vals.append((K + U)/N)


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        

        ax1.plot(t_vals, T_vals, label='Temperatura')
        ax1.plot(t_vals, P_vals, label='Ciśnienie', alpha=0.7)
        ax1.axhline(y=temp, color='r', linestyle='--', label='T zadana')
        ax1.set_ylabel('Wartość')
        ax1.set_xlabel('Czas')
        ax1.set_title('Zad. 1: Kontrola Temperatury i Ciśnienia')
        ax1.legend()
        

        ax2.plot(t_vals, E_tot_vals, color='green', label='Energia Całkowita / N')
        ax2.set_ylabel('Energia')
        ax2.set_xlabel('Czas')
        ax2.set_title('Zad. 3: Dochodzenie do stanu równowagi')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    else:

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, sim.L)
        ax.set_ylim(0, sim.L)
        ax.set_aspect('equal')
        ax.set_title(f"Animacja: {title}")
        
        particles, = ax.plot([], [], 'bo', ms=30) # ms=MarkerSize

        def init():
            particles.set_data([], [])
            return particles,

        def animate(i):
            for _ in range(5): 
                sim.step_brown_clarke()
            particles.set_data(sim.pos[:, 0], sim.pos[:, 1])
            return particles,

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
        plt.show()


run_simulation(mode='graph')


run_simulation(mode='fluid')


run_simulation(mode='solid')