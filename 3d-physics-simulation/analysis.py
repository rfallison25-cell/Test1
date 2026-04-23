"""Advanced analysis and visualization tools for the simulation."""

import numpy as np
import matplotlib.pyplot as plt
from main import run_simulation, SIMULATION_TIME, DT

def plot_energy_conservation(sim):
    """Plot energy over time to show energy dissipation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = np.array(sim.time_data)
    energy_data = sim.energy_data
    
    ke = [e['kinetic'] for e in energy_data]
    pe = [e['potential'] for e in energy_data]
    te = [e['total'] for e in energy_data]
    
    ax.plot(times, ke, label='Kinetic Energy', linewidth=2)
    ax.plot(times, pe, label='Potential Energy', linewidth=2)
    ax.plot(times, te, label='Total Energy', linewidth=2.5, linestyle='--')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy (J)', fontsize=12)
    ax.set_title('Energy Conservation: Dissipation Due to Collisions and Drag', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_velocity_components(sim):
    """Plot velocity components over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = np.array(sim.time_data)
    velocities = np.array(sim.velocity_data)
    speeds = np.linalg.norm(velocities, axis=1)
    
    ax.plot(times, velocities[:, 0], label='Vx (East-West)', alpha=0.7, linewidth=1.5)
    ax.plot(times, velocities[:, 1], label='Vy (Up-Down)', alpha=0.7, linewidth=1.5)
    ax.plot(times, velocities[:, 2], label='Vz (North-South)', alpha=0.7, linewidth=1.5)
    ax.plot(times, speeds, label='Speed (|V|)', linewidth=2.5, linestyle='--', color='black')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity Components Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_position_evolution(sim):
    """Plot position changes over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = np.array(sim.time_data)
    positions = np.array(sim.position_data)
    
    ax.plot(times, positions[:, 0], label='X Position', alpha=0.7, linewidth=1.5)
    ax.plot(times, positions[:, 1], label='Y Position (Height)', alpha=0.7, linewidth=1.5)
    ax.plot(times, positions[:, 2], label='Z Position', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position (m)', fontsize=12)
    ax.set_title('Cube Position Components Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_acceleration_analysis(sim):
    """Plot acceleration derived from velocity changes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = np.array(sim.time_data)
    velocities = np.array(sim.velocity_data)
    
    # Calculate accelerations
    accelerations = np.gradient(velocities, DT, axis=0)
    acc_magnitude = np.linalg.norm(accelerations, axis=1)
    
    ax.plot(times, accelerations[:, 0], label='Ax', alpha=0.6, linewidth=1)
    ax.plot(times, accelerations[:, 1], label='Ay', alpha=0.6, linewidth=1)
    ax.plot(times, accelerations[:, 2], label='Az', alpha=0.6, linewidth=1)
    ax.plot(times, acc_magnitude, label='|A| (Total)', linewidth=2.5, linestyle='--')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Acceleration Over Time (Derived from Velocity)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_force_analysis_detailed(sim):
    """Detailed force analysis with multiple views."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = np.array(sim.time_data)
    force_data = np.array([sim.force_data[i] for i in range(len(sim.force_data))])
    
    # Plot 1: All forces
    ax = axes[0, 0]
    ax.plot(times, force_data[:, 0], label='Gravity', linewidth=2)
    ax.plot(times, force_data[:, 1], label='Drag', linewidth=2)
    ax.plot(times, force_data[:, 2], label='Collision', linewidth=2)
    ax.plot(times, force_data[:, 3], label='Net Force', linewidth=2.5, linestyle='--')
    ax.set_ylabel('Force (N)', fontsize=11)
    ax.set_title('All Force Components', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Drag force detail
    ax = axes[0, 1]
    ax.plot(times, force_data[:, 1], color='blue', linewidth=2)
    ax.fill_between(times, force_data[:, 1], alpha=0.3)
    ax.set_ylabel('Drag Force (N)', fontsize=11)
    ax.set_title('Air Drag Over Time', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Collision force detail
    ax = axes[1, 0]
    ax.plot(times, force_data[:, 2], color='orange', linewidth=2)
    ax.fill_between(times, force_data[:, 2], alpha=0.3)
    ax.set_ylabel('Collision Force (N)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Wall/Floor Impact Forces (Spikes = Collisions)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Force ratio analysis
    ax = axes[1, 1]
    gravity_line = np.full_like(times, force_data[0, 0])  # Gravity is roughly constant
    ax.plot(times, force_data[:, 1] / gravity_line, label='Drag/Gravity Ratio', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Terminal velocity point')
    ax.set_ylabel('Force Ratio', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Drag Force Relative to Gravity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def print_detailed_analysis(sim):
    """Print detailed analysis of simulation results."""
    print("\n" + "="*70)
    print("DETAILED SIMULATION ANALYSIS")
    print("="*70)
    
    positions = np.array(sim.position_data)
    velocities = np.array(sim.velocity_data)
    force_data = np.array([sim.force_data[i] for i in range(len(sim.force_data))])
    energy_data = sim.energy_data
    
    # Collision detection
    collision_events = np.where(force_data[:, 2] > 100)[0]  # Significant collision force
    
    print(f"\nCollision Events:")
    print(f"  Number of significant collisions: {len(collision_events)}")
    if len(collision_events) > 0:
        print(f"  Times of major collisions: ", end="")
        for i in collision_events[::max(1, len(collision_events)//5)]:
            print(f"{sim.time_data[i]:.2f}s ", end="")
        print()
        print(f"  Peak collision force: {np.max(force_data[:, 2]):.2f} N")
    
    # Terminal velocity analysis
    speeds = np.linalg.norm(velocities, axis=1)
    print(f"\nTerminal Velocity Analysis:")
    print(f"  Initial speed: {speeds[0]:.2f} m/s")
    print(f"  Maximum speed: {np.max(speeds):.2f} m/s")
    print(f"  Final speed: {speeds[-1]:.2f} m/s")
    # Terminal velocity occurs when gravity ≈ drag
    print(f"  Speed decrease: {speeds[0] - speeds[-1]:.2f} m/s")
    
    # Bouncing analysis
    y_positions = positions[:, 1]
    print(f"\nBouncing Behavior:")
    print(f"  Initial height: {y_positions[0]:.2f} m")
    print(f"  Maximum height: {np.max(y_positions):.2f} m")
    print(f"  Minimum height: {np.min(y_positions):.2f} m")
    print(f"  Final height: {y_positions[-1]:.2f} m")
    
    # Energy efficiency
    print(f"\nEnergy Dissipation:")
    initial_energy = energy_data[0]['total']
    final_energy = energy_data[-1]['total']
    lost_energy = initial_energy - final_energy
    
    print(f"  Initial total energy: {initial_energy:.2f} J")
    print(f"  Final total energy: {final_energy:.2f} J")
    print(f"  Energy lost: {lost_energy:.2f} J ({100*lost_energy/initial_energy:.1f}%)")
    
    # Force statistics
    print(f"\nForce Statistics:")
    print(f"  Gravity force (constant): {force_data[0, 0]:.2f} N")
    print(f"  Average drag force: {np.mean(force_data[:, 1]):.2f} N")
    print(f"  Peak drag force: {np.max(force_data[:, 1]):.2f} N")
    print(f"  Average net force: {np.mean(force_data[:, 3]):.2f} N")
    print(f"  Peak net force (at collision): {np.max(force_data[:, 3]):.2f} N")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Running advanced analysis...")
    
    # Run simulation
    sim = run_simulation()
    
    # Print detailed analysis
    print_detailed_analysis(sim)
    
    # Generate plots
    print("\nGenerating analysis plots...")
    
    fig1 = plot_energy_conservation(sim)
    fig2 = plot_velocity_components(sim)
    fig3 = plot_position_evolution(sim)
    fig4 = plot_acceleration_analysis(sim)
    fig5 = plot_force_analysis_detailed(sim)
    
    print("Displaying plots...")
    plt.show()
