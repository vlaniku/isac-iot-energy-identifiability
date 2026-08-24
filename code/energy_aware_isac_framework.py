"""
Energy-Aware ISAC Framework for IoT Applications in Smart Cities
Author: Vullnet Laniku
Research: Energy-Efficient Dynamic Resource Allocation for IoT-Enabled ISAC Systems

This implementation provides a comprehensive framework for energy-aware Integrated Sensing 
and Communication (ISAC) systems specifically designed for IoT applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass
from enum import Enum
import time
from isac_physical_models import ISACPhysicalModels

class DeviceType(Enum):
    """Types of IoT devices in the smart city environment"""
    SENSOR = "sensor"
    CAMERA = "camera"
    ACTUATOR = "actuator"
    TRAFFIC_LIGHT = "traffic_light"
    SURVEILLANCE = "surveillance"

class TaskPriority(Enum):
    """Priority levels for different tasks"""
    CRITICAL = 3      # Emergency situations, safety-critical
    HIGH = 2          # Traffic management, public safety
    NORMAL = 1        # Regular monitoring, data collection
    LOW = 0           # Background tasks, maintenance

@dataclass
class DeviceState:
    """Current state of an IoT device"""
    device_id: str
    device_type: DeviceType
    battery_level: float  # 0.0 to 1.0
    current_power: float  # Watts
    location: Tuple[float, float]  # (x, y) coordinates
    is_active: bool
    last_update: float
    task_queue: List[TaskPriority]

@dataclass
class NetworkConditions:
    """Current network conditions affecting communication"""
    channel_quality: float  # 0.0 to 1.0
    interference_level: float  # 0.0 to 1.0
    available_bandwidth: float  # Mbps
    latency: float  # milliseconds
    packet_loss_rate: float  # 0.0 to 1.0

@dataclass
class OptimizationResult:
    """Result of the multi-objective optimization"""
    sensing_power: float
    communication_power: float
    processing_power: float
    sensing_accuracy: float
    communication_reliability: float
    energy_efficiency: float
    total_energy: float
    pareto_rank: int

class EnergyAwareISACFramework:
    """
    Main framework for energy-aware ISAC optimization
    """
    
    def __init__(self, 
                 num_devices: int = 100,
                 simulation_area: Tuple[float, float] = (1000.0, 1000.0),
                 time_horizon: float = 3600.0):  # 1 hour simulation
        """
        Initialize the Energy-Aware ISAC Framework
        
        Args:
            num_devices: Number of IoT devices in the smart city
            simulation_area: (width, height) of the simulation area in meters
            time_horizon: Total simulation time in seconds
        """
        self.num_devices = num_devices
        self.simulation_area = simulation_area
        self.time_horizon = time_horizon
        self.current_time = 0.0
        
        # Initialize devices and network
        self.devices = self._initialize_devices()
        self.network_conditions = self._initialize_network_conditions()
        
        # Initialize physical models for rigorous ISAC modeling
        self.physical_models = ISACPhysicalModels()
        
        # Optimization parameters
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # Performance tracking
        self.performance_history = []
        self.energy_history = []
        self.optimization_history = []
        
        print(f"Energy-Aware ISAC Framework initialized with {num_devices} devices")
        print(f"Simulation area: {simulation_area[0]}m x {simulation_area[1]}m")
        print(f"Time horizon: {time_horizon} seconds")
    
    def _initialize_devices(self) -> List[DeviceState]:
        """Initialize IoT devices with realistic parameters"""
        devices = []
        
        for i in range(self.num_devices):
            # Random device type distribution
            device_type = random.choice(list(DeviceType))
            
            # Realistic battery levels (some devices may be low on battery)
            battery_level = random.uniform(0.2, 1.0)
            
            # Random location in the smart city
            x = random.uniform(0, self.simulation_area[0])
            y = random.uniform(0, self.simulation_area[1])
            
            # Initial power consumption based on device type
            base_power = {
                DeviceType.SENSOR: 0.1,
                DeviceType.CAMERA: 0.5,
                DeviceType.ACTUATOR: 0.3,
                DeviceType.TRAFFIC_LIGHT: 0.2,
                DeviceType.SURVEILLANCE: 0.8
            }[device_type]
            
            current_power = base_power * random.uniform(0.8, 1.2)
            
            # Initialize task queue
            num_tasks = random.randint(1, 5)
            task_queue = [random.choice(list(TaskPriority)) for _ in range(num_tasks)]
            
            device = DeviceState(
                device_id=f"device_{i:03d}",
                device_type=device_type,
                battery_level=battery_level,
                current_power=current_power,
                location=(x, y),
                is_active=True,
                last_update=0.0,
                task_queue=task_queue
            )
            devices.append(device)
        
        return devices
    
    def _initialize_network_conditions(self) -> NetworkConditions:
        """Initialize network conditions"""
        return NetworkConditions(
            channel_quality=random.uniform(0.6, 0.95),
            interference_level=random.uniform(0.1, 0.4),
            available_bandwidth=random.uniform(50, 200),  # Mbps
            latency=random.uniform(5, 25),  # ms
            packet_loss_rate=random.uniform(0.01, 0.05)
        )
    
    def update_network_conditions(self, time_step: float):
        """Update network conditions based on time and device activity"""
        # Simulate time-varying network conditions
        time_factor = np.sin(2 * np.pi * self.current_time / 3600)  # Daily cycle
        
        # Channel quality varies with time and device density
        active_devices = sum(1 for d in self.devices if d.is_active)
        density_factor = active_devices / self.num_devices
        
        self.network_conditions.channel_quality = max(0.3, min(0.98, 
            0.8 + 0.15 * time_factor - 0.1 * density_factor + random.uniform(-0.05, 0.05)))
        
        # Interference increases with device density
        self.network_conditions.interference_level = min(0.8, 
            0.2 + 0.3 * density_factor + random.uniform(-0.02, 0.02))
        
        # Bandwidth allocation based on active devices
        self.network_conditions.available_bandwidth = max(20, 
            200 - 2 * active_devices + random.uniform(-10, 10))
        
        # Latency increases with network load
        self.network_conditions.latency = max(2, 
            10 + 0.1 * active_devices + random.uniform(-2, 2))
        
        # Packet loss increases with interference
        self.network_conditions.packet_loss_rate = min(0.1, 
            0.02 + 0.05 * self.network_conditions.interference_level + random.uniform(-0.01, 0.01))
    
    def calculate_energy_consumption(self, 
                                   sensing_power: float,
                                   communication_power: float,
                                   processing_power: float,
                                   duration: float) -> float:
        """
        Calculate total energy consumption for given power levels and duration
        
        Args:
            sensing_power: Power consumption for sensing (Watts)
            communication_power: Power consumption for communication (Watts)
            processing_power: Power consumption for processing (Watts)
            duration: Duration of operation (seconds)
            
        Returns:
            Total energy consumption in Joules
        """
        total_power = sensing_power + communication_power + processing_power
        return total_power * duration
    
    def calculate_sensing_accuracy(self, 
                                 sensing_power: float,
                                 device_type: DeviceType,
                                 battery_level: float,
                                 target_range: float = 100.0,
                                 frequency: float = 2.4e9,
                                 use_physical_model: bool = True) -> float:
        """
        Calculate sensing accuracy based on power allocation and device state
        Uses rigorous physical model (Pd/Pfa) when use_physical_model=True
        
        Args:
            sensing_power: Allocated sensing power (Watts)
            device_type: Type of IoT device
            battery_level: Current battery level (0.0 to 1.0)
            target_range: Target range for sensing in meters
            frequency: Operating frequency in Hz
            use_physical_model: If True, use rigorous physical model (Pd/Pfa)
            
        Returns:
            Sensing accuracy (0.0 to 1.0)
        """
        if use_physical_model:
            # Use rigorous physical model
            # Convert power to dBm
            sensing_power_db = 10 * np.log10(sensing_power * 1000)  # W to dBm
            
            # Use physical model for detection probability
            bandwidth = 20e6  # 20 MHz
            pd, pfa = self.physical_models.sensing_detection_probability(
                sensing_power_db, bandwidth, target_range,
                target_rcs=1.0, frequency=frequency, pulse_duration=1e-6
            )
            
            # Convert to accuracy
            accuracy = self.physical_models.sensing_accuracy_from_detection(pd, pfa)
            
            # Apply battery effect (lower battery reduces power/accuracy)
            battery_factor = 0.7 + 0.3 * battery_level
            accuracy *= battery_factor
            
        else:
            # Fallback to simple model (for backward compatibility)
            base_accuracy = {
                DeviceType.SENSOR: 0.85,
                DeviceType.CAMERA: 0.92,
                DeviceType.ACTUATOR: 0.78,
                DeviceType.TRAFFIC_LIGHT: 0.90,
                DeviceType.SURVEILLANCE: 0.95
            }[device_type]
            
            power_factor = min(1.0, sensing_power / 1.0)
            battery_factor = 0.7 + 0.3 * battery_level
            accuracy = base_accuracy * power_factor * battery_factor
        
        return np.clip(accuracy, 0.0, 1.0)
    
    def calculate_communication_reliability(self,
                                         communication_power: float,
                                         channel_quality: float,
                                         interference_level: float,
                                         distance: float = 100.0,
                                         frequency: float = 2.4e9,
                                         use_physical_model: bool = True) -> float:
        """
        Calculate communication reliability based on power and network conditions
        Uses rigorous physical model (SINR/PER) when use_physical_model=True
        
        Args:
            communication_power: Allocated communication power (Watts)
            channel_quality: Current channel quality (0.0 to 1.0)
            interference_level: Current interference level (0.0 to 1.0)
            distance: Distance to base station in meters
            frequency: Operating frequency in Hz
            use_physical_model: If True, use rigorous physical model (SINR/PER)
            
        Returns:
            Communication reliability (0.0 to 1.0)
        """
        if use_physical_model:
            # Use rigorous physical model
            # Convert power to dBm
            comm_power_db = 10 * np.log10(communication_power * 1000)  # W to dBm
            
            # Convert interference level to dBm (normalized 0-1 to dBm scale)
            # Assume interference ranges from -100 dBm (low) to -70 dBm (high)
            interference_power_db = -100 + (1 - interference_level) * 30
            
            # Use physical model
            bandwidth = 20e6  # 20 MHz
            reliability = self.physical_models.communication_reliability(
                comm_power_db, distance, frequency,
                interference_power_db, bandwidth,
                modulation='QAM16',
                coding_rate=0.75,
                packet_length=1024,
                environment='urban'
            )
            
            # Apply channel quality effect (additional factor)
            reliability *= channel_quality
            
        else:
            # Fallback to simple model (for backward compatibility)
            power_factor = min(1.0, communication_power / 2.0)
            channel_effect = channel_quality ** 2
            interference_effect = 1.0 - interference_level ** 1.5
            reliability = power_factor * channel_effect * interference_effect
        
        return np.clip(reliability, 0.0, 1.0)
    
    def multi_objective_optimization(self, device: DeviceState) -> OptimizationResult:
        """
        Perform multi-objective optimization for a single device
        
        Args:
            device: The IoT device to optimize
            
        Returns:
            OptimizationResult with optimal power allocation
        """
        best_solution = None
        best_fitness = float('inf')
        
        # Generate initial population
        population = self._generate_initial_population()
        
        for generation in range(self.generations):
            # Evaluate fitness for all solutions
            fitness_scores = []
            for solution in population:
                fitness = self._evaluate_fitness(solution, device)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
            
            # Selection, crossover, and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Track optimization progress
            if generation % 10 == 0:
                self.optimization_history.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean(fitness_scores)
                })
        
        # Convert best solution to OptimizationResult
        return self._solution_to_result(best_solution, device)
    
    def _generate_initial_population(self) -> List[Dict[str, float]]:
        """Generate initial population for genetic algorithm"""
        population = []
        
        for _ in range(self.population_size):
            solution = {
                'sensing_power': random.uniform(0.1, 2.0),
                'communication_power': random.uniform(0.1, 3.0),
                'processing_power': random.uniform(0.05, 1.0)
            }
            population.append(solution)
        
        return population
    
    def _evaluate_fitness(self, solution: Dict[str, float], device: DeviceState) -> float:
        """
        Evaluate fitness of a solution using weighted sum approach
        
        Args:
            solution: Power allocation solution
            device: Target device
            
        Returns:
            Fitness score (lower is better)
        """
        # Extract power allocations
        sensing_power = solution['sensing_power']
        communication_power = solution['communication_power']
        processing_power = solution['processing_power']
        
        # Calculate metrics
        energy_consumption = self.calculate_energy_consumption(
            sensing_power, communication_power, processing_power, 1.0)
        
        sensing_accuracy = self.calculate_sensing_accuracy(
            sensing_power, device.device_type, device.battery_level)
        
        communication_reliability = self.calculate_communication_reliability(
            communication_power, 
            self.network_conditions.channel_quality,
            self.network_conditions.interference_level)
        
        # Weighted fitness function (energy efficiency is prioritized)
        weights = {
            'energy': 0.5,      # 50% weight on energy efficiency
            'accuracy': 0.3,    # 30% weight on sensing accuracy
            'reliability': 0.2  # 20% weight on communication reliability
        }
        
        # Normalize metrics to 0-1 range
        normalized_energy = min(1.0, energy_consumption / 5.0)  # Normalize to 5W
        normalized_accuracy = 1.0 - sensing_accuracy  # Invert so lower is better
        normalized_reliability = 1.0 - communication_reliability  # Invert so lower is better
        
        # Calculate weighted fitness
        fitness = (weights['energy'] * normalized_energy + 
                  weights['accuracy'] * normalized_accuracy + 
                  weights['reliability'] * normalized_reliability)
        
        return fitness
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float]) -> Dict[str, float]:
        """Tournament selection for genetic algorithm"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_index]
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Crossover operation for genetic algorithm"""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, solution: Dict[str, float]) -> Dict[str, float]:
        """Mutation operation for genetic algorithm"""
        mutated = solution.copy()
        for key in mutated.keys():
            if random.random() < 0.3:  # 30% mutation probability
                # Add Gaussian noise
                noise = np.random.normal(0, 0.1)
                mutated[key] = max(0.01, mutated[key] + noise)
        return mutated
    
    def _solution_to_result(self, solution: Dict[str, float], device: DeviceState) -> OptimizationResult:
        """Convert solution to OptimizationResult"""
        sensing_power = solution['sensing_power']
        communication_power = solution['communication_power']
        processing_power = solution['processing_power']
        
        # Calculate metrics
        total_energy = self.calculate_energy_consumption(
            sensing_power, communication_power, processing_power, 1.0)
        
        sensing_accuracy = self.calculate_sensing_accuracy(
            sensing_power, device.device_type, device.battery_level)
        
        communication_reliability = self.calculate_communication_reliability(
            communication_power,
            self.network_conditions.channel_quality,
            self.network_conditions.interference_level)
        
        energy_efficiency = 1.0 / (total_energy + 1e-6)  # Avoid division by zero
        
        return OptimizationResult(
            sensing_power=sensing_power,
            communication_power=communication_power,
            processing_power=processing_power,
            sensing_accuracy=sensing_accuracy,
            communication_reliability=communication_reliability,
            energy_efficiency=energy_efficiency,
            total_energy=total_energy,
            pareto_rank=1
        )
    
    def run_simulation_step(self, time_step: float = 60.0):
        """
        Run one simulation step
        
        Args:
            time_step: Duration of the simulation step in seconds
        """
        print(f"\n--- Simulation Step at {self.current_time:.0f}s ---")
        
        # Update network conditions
        self.update_network_conditions(time_step)
        
        # Optimize each active device
        total_energy = 0.0
        total_accuracy = 0.0
        total_reliability = 0.0
        
        active_devices = [d for d in self.devices if d.is_active]
        
        for device in active_devices:
            # Skip optimization if battery is critically low
            if device.battery_level < 0.1:
                device.is_active = False
                print(f"Device {device.device_id} deactivated due to low battery")
                continue
            
            # Perform optimization
            result = self.multi_objective_optimization(device)
            
            # Update device state
            device.current_power = result.total_energy
            device.battery_level = max(0.0, device.battery_level - 
                                     (result.total_energy * time_step) / 3600.0)  # Convert to battery units
            
            # Update device location (simulate movement for some devices)
            if device.device_type in [DeviceType.SURVEILLANCE, DeviceType.CAMERA]:
                # Add small random movement
                dx = random.uniform(-5, 5)
                dy = random.uniform(-5, 5)
                x, y = device.location
                device.location = (max(0, min(self.simulation_area[0], x + dx)),
                                 max(0, min(self.simulation_area[1], y + dy)))
            
            # Accumulate metrics
            total_energy += result.total_energy
            total_accuracy += result.sensing_accuracy
            total_reliability += result.communication_reliability
            
            # Update task queue
            if device.task_queue:
                completed_task = device.task_queue.pop(0)
                # Add new task based on priority
                new_task = random.choice(list(TaskPriority))
                device.task_queue.append(new_task)
        
        # Calculate average metrics
        if active_devices:
            avg_energy = total_energy / len(active_devices)
            avg_accuracy = total_accuracy / len(active_devices)
            avg_reliability = total_reliability / len(active_devices)
        else:
            avg_energy = avg_accuracy = avg_reliability = 0.0
        
        # Store performance metrics
        self.performance_history.append({
            'time': self.current_time,
            'active_devices': len(active_devices),
            'avg_energy': avg_energy,
            'avg_accuracy': avg_accuracy,
            'avg_reliability': avg_reliability,
            'network_channel_quality': self.network_conditions.channel_quality
        })
        
        # Print summary
        print(f"Active devices: {len(active_devices)}/{self.num_devices}")
        print(f"Average energy consumption: {avg_energy:.3f} W")
        print(f"Average sensing accuracy: {avg_accuracy:.3f}")
        print(f"Average communication reliability: {avg_reliability:.3f}")
        print(f"Network channel quality: {self.network_conditions.channel_quality:.3f}")
        
        # Update time
        self.current_time += time_step
    
    def run_full_simulation(self):
        """Run the complete simulation"""
        print("Starting full simulation...")
        
        time_step = 60.0  # 1-minute steps
        num_steps = int(self.time_horizon / time_step)
        
        for step in range(num_steps):
            self.run_simulation_step(time_step)
            
            # Check if all devices are inactive
            active_count = sum(1 for d in self.devices if d.is_active)
            if active_count == 0:
                print("All devices have been deactivated. Simulation ending early.")
                break
        
        print("\n=== Simulation Complete ===")
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print final simulation statistics"""
        if not self.performance_history:
            print("No performance data available.")
            return
        
        # Calculate final statistics
        final_stats = self.performance_history[-1]
        energy_values = [p['avg_energy'] for p in self.performance_history]
        accuracy_values = [p['avg_accuracy'] for p in self.performance_history]
        reliability_values = [p['avg_reliability'] for p in self.performance_history]
        
        print(f"\nFinal Statistics:")
        print(f"Total simulation time: {self.current_time:.0f} seconds")
        print(f"Final active devices: {final_stats['active_devices']}")
        print(f"Average energy consumption: {np.mean(energy_values):.3f} ± {np.std(energy_values):.3f} W")
        print(f"Average sensing accuracy: {np.mean(accuracy_values):.3f} ± {np.std(accuracy_values):.3f}")
        print(f"Average communication reliability: {np.mean(reliability_values):.3f} ± {np.std(reliability_values):.3f}")
        
        # Energy efficiency improvement
        if len(energy_values) > 1:
            initial_energy = energy_values[0]
            final_energy = energy_values[-1]
            improvement = ((initial_energy - final_energy) / initial_energy) * 100
            print(f"Energy efficiency improvement: {improvement:.1f}%")
    
    def plot_performance_metrics(self):
        """Plot performance metrics over time"""
        if not self.performance_history:
            print("No performance data to plot.")
            return
        
        # Extract data
        times = [p['time'] for p in self.performance_history]
        energy = [p['avg_energy'] for p in self.performance_history]
        accuracy = [p['avg_accuracy'] for p in self.performance_history]
        reliability = [p['avg_reliability'] for p in self.performance_history]
        active_devices = [p['active_devices'] for p in self.performance_history]
        channel_quality = [p['network_channel_quality'] for p in self.performance_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Energy-Aware ISAC Framework Performance Metrics', fontsize=16)
        
        # Energy consumption
        axes[0, 0].plot(times, energy, 'b-', linewidth=2)
        axes[0, 0].set_title('Average Energy Consumption')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sensing accuracy
        axes[0, 1].plot(times, accuracy, 'g-', linewidth=2)
        axes[0, 1].set_title('Average Sensing Accuracy')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Communication reliability
        axes[0, 2].plot(times, reliability, 'r-', linewidth=2)
        axes[0, 2].set_title('Average Communication Reliability')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Reliability')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Active devices
        axes[1, 0].plot(times, active_devices, 'purple', linewidth=2)
        axes[1, 0].set_title('Active Devices')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Number of Devices')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Network channel quality
        axes[1, 1].plot(times, channel_quality, 'orange', linewidth=2)
        axes[1, 1].set_title('Network Channel Quality')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Channel Quality')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Energy vs Accuracy trade-off
        axes[1, 2].scatter(energy, accuracy, c=times, cmap='viridis', alpha=0.7)
        axes[1, 2].set_title('Energy vs Accuracy Trade-off')
        axes[1, 2].set_xlabel('Energy Consumption (W)')
        axes[1, 2].set_ylabel('Sensing Accuracy')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename: str = "isac_simulation_results.txt"):
        """Save simulation results to a file"""
        with open(filename, 'w') as f:
            f.write("Energy-Aware ISAC Framework Simulation Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Simulation Parameters:\n")
            f.write(f"Number of devices: {self.num_devices}\n")
            f.write(f"Simulation area: {self.simulation_area[0]}m x {self.simulation_area[1]}m\n")
            f.write(f"Time horizon: {self.time_horizon} seconds\n\n")
            
            f.write("Performance History:\n")
            f.write("-" * 30 + "\n")
            for entry in self.performance_history:
                f.write(f"Time: {entry['time']:.0f}s, "
                       f"Active: {entry['active_devices']}, "
                       f"Energy: {entry['avg_energy']:.3f}W, "
                       f"Accuracy: {entry['avg_accuracy']:.3f}, "
                       f"Reliability: {entry['avg_reliability']:.3f}\n")
        
        print(f"Results saved to {filename}")

def main():
    """Main function to demonstrate the framework"""
    print("Energy-Aware ISAC Framework for IoT Applications")
    print("=" * 55)
    
    # Create framework instance
    framework = EnergyAwareISACFramework(
        num_devices=50,  # Reduced for faster demonstration
        simulation_area=(500.0, 500.0),
        time_horizon=1800.0  # 30 minutes
    )
    
    # Run simulation
    framework.run_full_simulation()
    
    # Plot results
    framework.plot_performance_metrics()
    
    # Save results
    framework.save_results()
    
    print("\nFramework demonstration completed successfully!")

if __name__ == "__main__":
    main()

