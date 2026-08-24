"""
Closed-Loop ISAC Simulator
Allows actions to actually change outcomes based on physical models
Author: Vullnet Laniku
Research: Energy-Efficient Dynamic Resource Allocation for IoT-Enabled ISAC Systems
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from isac_physical_models import ISACPhysicalModels

@dataclass
class SimulatorDeviceState:
    """Device state for closed-loop simulator"""
    device_id: str
    battery_level: float  # 0.0 to 1.0
    battery_capacity_joules: float
    location: Tuple[float, float]
    queue_length: int = 0
    distance_to_base: float = 100.0  # meters
    target_range: float = 100.0  # meters for sensing

@dataclass
class ISACAction:
    """Action for ISAC system"""
    sensing_power_db: float  # dBm
    comm_power_db: float  # dBm
    sensing_bandwidth_ratio: float  # 0.0 to 1.0
    sensing_rate: float = 1.0  # Hz

class ClosedLoopISACSimulator:
    """
    Simulator that allows actions to actually change outcomes
    Uses rigorous physical models for ISAC performance
    """
    
    def __init__(self, 
                 devices: List[SimulatorDeviceState],
                 shared_bandwidth: float = 20e6,  # 20 MHz
                 frequency: float = 2.4e9,
                 environment: str = 'urban',
                 base_station_location: Tuple[float, float] = (0.0, 0.0),
                 mac_mode: str = 'ofdma'):
        """
        Initialize closed-loop simulator
        
        Args:
            devices: List of device states
            shared_bandwidth: Total shared bandwidth in Hz
            frequency: Operating frequency in Hz
            environment: Environment type ('urban', 'suburban', 'rural')
            base_station_location: Base station location (x, y)
            mac_mode: 'ofdma' (concurrent, shared BW) or 'tdma' (scheduled,
                      full BW per device, no inter-device interference)
        """
        self.devices = {d.device_id: d for d in devices}
        self.physical_models = ISACPhysicalModels()
        self.shared_bandwidth = shared_bandwidth
        self.frequency = frequency
        self.environment = environment
        self.base_station_location = base_station_location
        self.mac_mode = mac_mode.lower()
        
        # Network state
        self.network_interference = {}  # device_id -> interference_power_db
        self.step_count = 0
        
    def step(self, actions: Dict[str, ISACAction]) -> Dict[str, Dict[str, float]]:
        """
        Execute actions and observe real outcomes
        
        Args:
            actions: Dict mapping device_id to ISACAction
            
        Returns:
            Dict mapping device_id to outcomes (energy, accuracy, reliability, latency, battery_level, etc.)
        """
        outcomes = {}
        
        # First pass: record each device's sensing power and location for
        # inter-device interference calculation (with path-loss attenuation)
        device_sensing = {}
        for device_id, action in actions.items():
            if device_id in self.devices:
                device_sensing[device_id] = {
                    'power_db': action.sensing_power_db,
                    'location': self.devices[device_id].location
                }
        
        # Second pass: calculate outcomes for each device
        for device_id, action in actions.items():
            if device_id not in self.devices:
                continue
                
            device = self.devices[device_id]
            
            n_active = max(len(actions), 1)
            
            if self.mac_mode == 'tdma':
                # TDMA: each device transmits in its own time slot with
                # the full system bandwidth.  No other device transmits
                # concurrently, so inter-device interference is zero.
                # Slot fraction = 1/N of the epoch per device.
                other_interference_watt = 0.0
                per_device_bw = self.shared_bandwidth  # full 20 MHz
                slot_fraction = 1.0 / n_active
            else:
                # OFDMA: all devices transmit concurrently, sharing
                # bandwidth. Inter-device interference is present.
                # Each device is active for the full epoch.
                other_interference_watt = 0.0
                for other_id, info in device_sensing.items():
                    if other_id != device_id:
                        other_dev = self.devices[other_id]
                        intf_power_db = info['power_db'] - 20.0
                        pl_db = self.physical_models.path_loss_model(
                            max(other_dev.distance_to_base, 1.0),
                            self.frequency, self.environment
                        )
                        intf_at_rx_db = intf_power_db - pl_db
                        intf_watt = 10 ** ((intf_at_rx_db - 30) / 10)
                        other_interference_watt += max(intf_watt, 0.0)
                per_device_bw = self.shared_bandwidth / n_active
                slot_fraction = 1.0
            
            # External interference floor (-110 dBm, typical for dedicated
            # IoT ISAC spectrum in urban smart-city deployment)
            external_interference_watt = 10 ** ((-110 - 30) / 10)
            
            total_interference_watt = other_interference_watt + external_interference_watt
            interference_power_db = (10 * np.log10(total_interference_watt * 1000)
                                     if total_interference_watt > 0 else -200)
            
            # Calculate ISAC joint performance using physical models
            isac_performance = self.physical_models.isac_joint_performance(
                sensing_power_db=action.sensing_power_db,
                comm_power_db=action.comm_power_db,
                shared_bandwidth=per_device_bw,
                sensing_bandwidth_ratio=action.sensing_bandwidth_ratio,
                target_range=device.target_range,
                target_rcs=1.0,  # m^2
                distance=device.distance_to_base,
                frequency=self.frequency,
                queue_length=device.queue_length,
                packet_size=256,  # IoT payload (small packets)
                interference_power_db=interference_power_db,
                environment=self.environment,
                tdma_slot_fraction=slot_fraction
            )
            
            # Get actual energy consumption from physical model
            energy_joules = isac_performance['energy_consumption']
            
            # Update device state based on ACTUAL energy consumption
            battery_depletion = energy_joules / device.battery_capacity_joules
            device.battery_level = max(0.0, min(1.0, device.battery_level - battery_depletion))
            
            # Update queue (simplified model)
            # If communication is reliable, packets are transmitted
            if isac_performance['communication_reliability'] > 0.9:
                # Transmit packets from queue
                packets_transmitted = min(device.queue_length, int(isac_performance['communication_reliability'] * 10))
                device.queue_length = max(0, device.queue_length - packets_transmitted)
            
            # Add new packets to queue (simplified arrival model)
            arrival_rate = 5.0  # packets per second
            if np.random.random() < arrival_rate / 100.0:  # Simplified Poisson
                device.queue_length += 1
            
            # Store outcomes
            outcomes[device_id] = {
                'energy_joules': energy_joules,
                'energy_normalized': battery_depletion,
                'accuracy': isac_performance['sensing_accuracy'],
                'reliability': isac_performance['communication_reliability'],
                'latency': isac_performance['latency'],
                'battery_level': device.battery_level,
                'sensing_pd': isac_performance['sensing_pd'],
                'sensing_pfa': isac_performance['sensing_pfa'],
                'comm_sinr_db': isac_performance['comm_sinr_db'],
                'sensing_power_watt': isac_performance['sensing_power_watt'],
                'comm_power_watt': isac_performance['comm_power_watt'],
                'total_power_watt': isac_performance['total_power_watt'],
                'queue_length': device.queue_length
            }
        
        self.step_count += 1
        return outcomes
    
    def get_current_states(self) -> Dict[str, SimulatorDeviceState]:
        """Get current device states"""
        return self.devices.copy()
    
    def get_device(self, device_id: str) -> Optional[SimulatorDeviceState]:
        """Get specific device state"""
        return self.devices.get(device_id)
    
    def reset(self, initial_states: List[SimulatorDeviceState]):
        """Reset simulator to initial states"""
        self.devices = {d.device_id: d for d in initial_states}
        self.network_interference = {}
        self.step_count = 0
    
    def calculate_distance(self, device_location: Tuple[float, float]) -> float:
        """Calculate distance from device to base station"""
        dx = device_location[0] - self.base_station_location[0]
        dy = device_location[1] - self.base_station_location[1]
        return np.sqrt(dx**2 + dy**2)
    
    def update_device_distances(self):
        """Update distances for all devices"""
        for device in self.devices.values():
            device.distance_to_base = self.calculate_distance(device.location)



