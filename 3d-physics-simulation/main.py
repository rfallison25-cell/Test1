import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

# Physics constants
G = 9.81  # gravity (m/s^2)
DRAG_COEFFICIENT = 0.47  # unitless (sphere-like)
AIR_DENSITY = 1.225  # kg/m^3
CUBE_MASS = 1.0  # kg
CUBE_SIZE = 0.5  # meters (side length)
CONTAINER_SIZE = 10.0  # meters
COEFFICIENT_OF_RESTITUTION = 0.75  # bounciness (0-1)
FRICTION_COEFFICIENT = 0.1  # damping on collision
DT = 0.016  # time step (seconds) - ~60 FPS
SIMULATION_TIME = 30  # seconds

class CubePhysicsSimulator:
    def __init__(self):
        # Initial conditions
        self.position = np.array([0.0, 5.0, 0.0])  # Start in middle, elevated
        self.velocity = np.array([2.0, 0.0, 1.5])  # Initial velocity
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        # Forces (in Newtons)
        self.gravity_force = np.array([0.0, -CUBE_MASS * G, 0.0])
        self.collision_force = np.array([0.0, 0.0, 0.0])
        self.drag_force = np.array([0.0, 0.0, 0.0])
        self.net_force = np.array([0.0, 0.0, 0.0])
        
        # Energy tracking
        self.kinetic_energy = 0.0
        self.potential_energy = 0.0
        self.total_energy = 0.0
        
        # Data recording for analysis
        self.time_data = []
        self.position_data = []
        self.velocity_data = []
        self.force_data = []
        self.energy_data = []
        
    def calculate_drag_force(self):
        """Calculate air resistance using drag equation: F = 0.5 * ρ * v² * Cd * A"""
        speed = np.linalg.norm(self.velocity)
        if speed < 0.01:  # Avoid division by near-zero
            self.drag_force = np.array([0.0, 0.0, 0.0])
            return
        
        # Cross-sectional area of cube (simplified as square)
        area = CUBE_SIZE * CUBE_SIZE
        drag_magnitude = 0.5 * AIR_DENSITY * speed**2 * DRAG_COEFFICIENT * area
        
        # Drag opposes velocity
        drag_direction = -self.velocity / speed
        self.drag_force = drag_magnitude * drag_direction
    
    def check_collisions(self):
        """Check and handle collisions with container walls and floor"""
        self.collision_force = np.array([0.0, 0.0, 0.0])
        
        half_container = CONTAINER_SIZE / 2
        half_cube = CUBE_SIZE / 2
        
        # Check each axis for collision
        axes = [0, 1, 2]  # x, y, z
        
        for axis in axes:
            # Check lower bound
            if self.position[axis] - half_cube <= -half_container:
                # Collision detected
                penetration = (-half_container) - (self.position[axis] - half_cube)
                
                # Normal force (restoring force)
                normal_force_magnitude = penetration * 100000  # Spring constant
                normal_force = np.zeros(3)
                normal_force[axis] = normal_force_magnitude
                self.collision_force += normal_force
                
                # Bounce (coefficient of restitution)
                if self.velocity[axis] < 0:
                    self.velocity[axis] *= -COEFFICIENT_OF_RESTITUTION
                    self.velocity[axis] *= (1 - FRICTION_COEFFICIENT)
                
                # Move cube out of wall
                self.position[axis] = -half_container + half_cube + 0.01
            
            # Check upper bound
            if self.position[axis] + half_cube >= half_container:
                penetration = (self.position[axis] + half_cube) - half_container
                
                normal_force_magnitude = penetration * 100000
                normal_force = np.zeros(3)
                normal_force[axis] = -normal_force_magnitude
                self.collision_force += normal_force
                
                if self.velocity[axis] > 0:
                    self.velocity[axis] *= -COEFFICIENT_OF_RESTITUTION
                    self.velocity[axis] *= (1 - FRICTION_COEFFICIENT)
                
                self.position[axis] = half_container - half_cube - 0.01
    
    def calculate_forces(self):
        """Calculate all forces acting on the cube"""
        # Gravity (always present)
        self.gravity_force = np.array([0.0, -CUBE_MASS * G, 0.0])
        
        # Drag force (depends on velocity)
        self.calculate_drag_force()
        
        # Collision forces (if applicable)
        self.check_collisions()
        
        # Net force
        self.net_force = self.gravity_force + self.drag_force + self.collision_force
    
    def update_energy(self):
        """Calculate kinetic and potential energy"""
        speed = np.linalg.norm(self.velocity)
        self.kinetic_energy = 0.5 * CUBE_MASS * speed**2
        
        # Potential energy (relative to bottom of container)
        height = self.position[1] + CONTAINER_SIZE / 2
        self.potential_energy = CUBE_MASS * G * height
        
        self.total_energy = self.kinetic_energy + self.potential_energy
    
    def step(self, dt):
        """Perform one simulation step using Velocity Verlet integration"""
        # Calculate forces at current position
        self.calculate_forces()
        
        # Velocity Verlet integration (more accurate than Euler method)
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        
        # Store old acceleration
        old_acceleration = self.acceleration.copy()
        
        # Update position
        self.position += self.velocity * dt + 0.5 * old_acceleration * dt**2
        
        # Recalculate forces at new position
        self.calculate_forces()
        
        # Calculate new acceleration
        self.acceleration = self.net_force / CUBE_MASS
        
        # Update velocity
        self.velocity += 0.5 * (old_acceleration + self.acceleration) * dt
        
        # Update energy
        self.update_energy()
    
    def record_data(self, current_time):
        """Record simulation data for analysis"""
        self.time_data.append(current_time)
        self.position_data.append(self.position.copy())
        self.velocity_data.append(self.velocity.copy())
        self.force_data.append({
            'gravity': np.linalg.norm(self.gravity_force),
            'drag': np.linalg.norm(self.drag_force),
            'collision': np.linalg.norm(self.collision_force),
            'net': np.linalg.norm(self.net_force)
        })
        self.energy_data.append({
            'kinetic': self.kinetic_energy,
            'potential': self.potential_energy,
            'total': self.total_energy
        })
    
    def get_cube_vertices(self):
        """Get the 8 vertices of the cube at current position"""
        half_size = CUBE_SIZE / 2
        vertices = np.array([
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size],
        ])
        return vertices + self.position
    
    def get_force_arrows(self):
        """Get force vectors scaled for visualization"""
        scale_factor = 0.01  # Scale down for visualization
        return {
            'gravity': self.gravity_force * scale_factor,
            'drag': self.drag_force * scale_factor,
            'collision': self.collision_force * scale_factor,
            'net': self.net_force * scale_factor
        }

def run_simulation():
    """Run the complete simulation"""
    sim = CubePhysicsSimulator()
    
    # Calculate number of steps
    num_steps = int(SIMULATION_TIME / DT)
    
    print(f"Starting 3D Cube Physics Simulation")
    print(f"Simulation time: {SIMULATION_TIME} seconds")
    print(f"Time step: {DT} seconds")
    print(f"Total steps: {num_steps}")
    print(f"\nPhysics Parameters:")
    print(f"  Gravity: {G} m/s²")
    print(f"  Cube mass: {CUBE_MASS} kg")
    print(f"  Coefficient of restitution: {COEFFICIENT_OF_RESTITUTION}")
    print(f"  Drag coefficient: {DRAG_COEFFICIENT}")
    print(f"\nRunning simulation...")
    
    # Run simulation
    for step in range(num_steps):
        current_time = step * DT
        sim.step(DT)
        sim.record_data(current_time)
        
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps} ({current_time:.1f}s)")
    
    print(f"Simulation complete!")
    return sim

def create_visualization(sim):
    """Create 3D visualization of the simulation"""
    fig = plt.figure(figsize=(16, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Trajectory
    positions = np.array(sim.position_data)
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'c-', alpha=0.3, linewidth=0.5, label='Trajectory')
    
    # Draw container cube
    half_container = CONTAINER_SIZE / 2
    container_vertices = np.array([
        [-half_container, -half_container, -half_container],
        [half_container, -half_container, -half_container],
        [half_container, half_container, -half_container],
        [-half_container, half_container, -half_container],
        [-half_container, -half_container, half_container],
        [half_container, -half_container, half_container],
        [half_container, half_container, half_container],
        [-half_container, half_container, half_container],
    ])
    
    # Container edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
    ]
    
    for edge in edges:
        points = container_vertices[edge]
        ax1.plot3D(*points.T, 'gray', alpha=0.3, linewidth=1)
    
    # Starting and ending positions
    ax1.scatter(*sim.position_data[0], color='green', s=100, label='Start')
    ax1.scatter(*sim.position_data[-1], color='red', s=100, label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Cube Trajectory in Container')
    ax1.legend()
    ax1.set_xlim([-CONTAINER_SIZE/2, CONTAINER_SIZE/2])
    ax1.set_ylim([-CONTAINER_SIZE/2, CONTAINER_SIZE/2])
    ax1.set_zlim([-CONTAINER_SIZE/2, CONTAINER_SIZE/2])
    
    # Force data plot
    ax2 = fig.add_subplot(122)
    times = np.array(sim.time_data)
    force_magnitudes = np.array([sim.force_data[i] for i in range(len(sim.force_data))])
    
    ax2.plot(times, force_magnitudes[:, 0], label='Gravity', linewidth=2)
    ax2.plot(times, force_magnitudes[:, 1], label='Drag', linewidth=2)
    ax2.plot(times, force_magnitudes[:, 2], label='Collision', linewidth=2)
    ax2.plot(times, force_magnitudes[:, 3], label='Net Force', linewidth=2.5, linestyle='--')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force Magnitude (N)')
    ax2.set_title('Force Components Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_statistics(sim):
    """Print simulation statistics"""
    print("\n" + "="*60)
    print("SIMULATION STATISTICS")
    print("="*60)
    
    positions = np.array(sim.position_data)
    velocities = np.array(sim.velocity_data)
    energy_data = sim.energy_data
    
    print(f"\nPosition Statistics:")
    print(f"  Initial: {sim.position_data[0]}")
    print(f"  Final: {sim.position_data[-1]}")
    print(f"  Max height: {np.max(positions[:, 1]):.2f} m")
    print(f"  Min height: {np.min(positions[:, 1]):.2f} m")
    
    speeds = np.linalg.norm(velocities, axis=1)
    print(f"\nVelocity Statistics:")
    print(f"  Initial speed: {speeds[0]:.2f} m/s")
    print(f"  Final speed: {speeds[-1]:.2f} m/s")
    print(f"  Max speed: {np.max(speeds):.2f} m/s")
    print(f"  Average speed: {np.mean(speeds):.2f} m/s")
    
    print(f"\nEnergy Statistics:")
    print(f"  Initial KE: {energy_data[0]['kinetic']:.2f} J")
    print(f"  Initial PE: {energy_data[0]['potential']:.2f} J")
    print(f"  Initial Total: {energy_data[0]['total']:.2f} J")
    print(f"  Final Total: {energy_data[-1]['total']:.2f} J")
    print(f"  Energy lost: {energy_data[0]['total'] - energy_data[-1]['total']:.2f} J")
    print(f"  Energy loss %: {100*(energy_data[0]['total'] - energy_data[-1]['total'])/energy_data[0]['total']:.1f}%")
    
    force_data = np.array([sim.force_data[i] for i in range(len(sim.force_data))])
    print(f"\nForce Statistics:")
    print(f"  Average gravity: {np.mean(force_data[:, 0]):.2f} N")
    print(f"  Average drag: {np.mean(force_data[:, 1]):.2f} N")
    print(f"  Peak collision force: {np.max(force_data[:, 2]):.2f} N")
    print(f"  Average net force: {np.mean(force_data[:, 3]):.2f} N")
    print("="*60)

if __name__ == "__main__":
    # Run simulation
    sim = run_simulation()
    
    # Print statistics
    print_statistics(sim)
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = create_visualization(sim)
    
    # Show plot
    plt.show()
