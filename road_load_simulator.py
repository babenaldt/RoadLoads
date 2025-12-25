"""
Road Load Simulator
====================
Simulates vehicle road loads for drive cycles with variable grade.
Calculates power requirements, energy consumption, and generates plots.

Author: Generated with GitHub Copilot
Date: December 2024
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional


# =============================================================================
# Constants
# =============================================================================
AIR_DENSITY = 1.225  # kg/m³ (at sea level, 15°C)
GRAVITY = 9.81  # m/s²

# Unit conversion factors
MPS_TO_MPH = 2.23694
W_TO_HP = 0.00134102
KWH_TO_HP_HR = 1.34102
M_TO_FT = 3.28084
KG_TO_LB = 2.20462
M2_TO_FT2 = 10.7639

# Energy constants
GASOLINE_ENERGY_KWH_PER_GALLON = 33.7  # kWh per gallon of gasoline


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class VehicleParams:
    """Vehicle parameters for road load calculations."""
    mass: float  # kg
    frontal_area: float  # m²
    drag_coefficient: float  # dimensionless
    rolling_resistance: float  # dimensionless (typical: 0.01-0.02)
    
    # Vehicle class: 'bev', 'erev', or 'gasoline'
    vehicle_class: str = 'bev'
    
    # BEV/EREV parameters
    regen_efficiency: float = 0.7  # regenerative braking efficiency (0-1)
    auxiliary_power: float = 0.0  # constant auxiliary power draw in Watts (HVAC, electronics, etc.)
    battery_capacity: float = 75.0  # battery capacity in kWh
    usable_battery_pct: float = 90.0  # usable percentage of battery capacity
    
    # EREV-specific parameters
    generator_power_kw: float = 0.0  # generator max output power in kW
    fuel_tank_gallons: float = 0.0  # fuel tank capacity in gallons
    bsfc_g_kwh: float = 250.0  # brake specific fuel consumption in g/kWh
    erev_mode: str = 'charge_depleting'  # 'charge_depleting', 'blended', or 'hold'
    soc_sustain_pct: float = 20.0  # SOC threshold for sustaining mode (%)
    soc_blended_threshold_pct: float = 30.0  # SOC threshold for blended mode assist (%)
    
    # Gasoline-specific parameters
    fuel_economy_mpg: float = 25.0  # fuel economy in miles per gallon

    def __str__(self) -> str:
        base = (
            f"Vehicle Parameters:\n"
            f"  Vehicle Class: {self.vehicle_class.upper()}\n"
            f"  Mass: {self.mass:.1f} kg ({self.mass * KG_TO_LB:.1f} lb)\n"
            f"  Frontal Area: {self.frontal_area:.2f} m² ({self.frontal_area * M2_TO_FT2:.2f} ft²)\n"
            f"  Drag Coefficient: {self.drag_coefficient:.3f}\n"
            f"  Rolling Resistance: {self.rolling_resistance:.4f}\n"
        )
        
        if self.vehicle_class == 'bev':
            base += (
                f"  Regen Efficiency: {self.regen_efficiency * 100:.1f}%\n"
                f"  Auxiliary Power: {self.auxiliary_power:.0f} W\n"
                f"  Battery Capacity: {self.battery_capacity:.1f} kWh\n"
                f"  Usable Battery: {self.usable_battery_pct:.0f}%"
            )
        elif self.vehicle_class == 'erev':
            base += (
                f"  Regen Efficiency: {self.regen_efficiency * 100:.1f}%\n"
                f"  Auxiliary Power: {self.auxiliary_power:.0f} W\n"
                f"  Battery Capacity: {self.battery_capacity:.1f} kWh\n"
                f"  Usable Battery: {self.usable_battery_pct:.0f}%\n"
                f"  Generator Power: {self.generator_power_kw:.1f} kW\n"
                f"  Fuel Tank: {self.fuel_tank_gallons:.1f} gal\n"
                f"  BSFC: {self.bsfc_g_kwh:.0f} g/kWh\n"
                f"  EREV Mode: {self.erev_mode}\n"
                f"  SOC Sustain Threshold: {self.soc_sustain_pct:.0f}%"
            )
        elif self.vehicle_class == 'gasoline':
            base += (
                f"  Fuel Tank: {self.fuel_tank_gallons:.1f} gal\n"
                f"  Fuel Economy: {self.fuel_economy_mpg:.1f} MPG"
            )
        
        return base


@dataclass
class DriveCyclePoint:
    """Single point in a drive cycle."""
    time: float  # seconds
    speed: float  # m/s
    grade: float  # percent (e.g., 6.0 for 6%)


@dataclass
class SimulationResults:
    """Results from road load simulation."""
    time: np.ndarray  # seconds
    speed: np.ndarray  # m/s
    grade: np.ndarray  # percent
    acceleration: np.ndarray  # m/s²
    power: np.ndarray  # Watts (positive = traction, negative = regen)
    energy_cumulative: np.ndarray  # Joules
    
    # Force components (for analysis)
    force_aero: np.ndarray  # N
    force_rolling: np.ndarray  # N
    force_grade: np.ndarray  # N
    force_accel: np.ndarray  # N
    force_total: np.ndarray  # N

    @property
    def peak_power_traction(self) -> float:
        """Peak traction power in Watts."""
        return float(np.max(self.power))

    @property
    def peak_power_regen(self) -> float:
        """Peak regenerative power in Watts (returned as positive value)."""
        return float(-np.min(self.power))

    @property
    def average_power(self) -> float:
        """Average power in Watts."""
        return float(np.mean(self.power))

    @property
    def total_energy(self) -> float:
        """Total net energy in Joules."""
        return float(self.energy_cumulative[-1])

    @property
    def traction_energy(self) -> float:
        """Total traction energy (positive power only) in Joules."""
        dt = np.diff(self.time, prepend=0)
        positive_power = np.maximum(self.power, 0)
        return float(np.sum(positive_power * dt))

    @property
    def regen_energy(self) -> float:
        """Total regenerated energy (negative power) in Joules."""
        dt = np.diff(self.time, prepend=0)
        negative_power = np.minimum(self.power, 0)
        return float(-np.sum(negative_power * dt))


# =============================================================================
# Drive Cycle Loading
# =============================================================================
def load_drive_cycle(filepath: str) -> List[DriveCyclePoint]:
    """
    Load drive cycle from CSV file.
    
    Expected CSV format:
        time,speed,grade
        0,0,0
        1,5.2,0
        ...
    
    Note: Speed values in CSV must be in m/s (meters per second).
          This is the standard format for EPA drive cycles.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of DriveCyclePoint objects
    """
    cycle_points = []
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time = float(row['time'])
            speed = float(row['speed'])  # Must be in m/s
            grade = float(row.get('grade', 0))  # Default grade to 0 if not provided
            
            cycle_points.append(DriveCyclePoint(time=time, speed=speed, grade=grade))
    
    return cycle_points


# =============================================================================
# Physics Calculations
# =============================================================================
def calculate_road_load(
    vehicle: VehicleParams,
    cycle: List[DriveCyclePoint]
) -> SimulationResults:
    """
    Calculate road load forces and power for a drive cycle.
    
    Road load equation:
        F_total = F_aero + F_rolling + F_grade + F_accel
        
    Where:
        F_aero = 0.5 * ρ * Cd * A * v²
        F_rolling = Crr * m * g * cos(θ)
        F_grade = m * g * sin(θ)
        F_accel = m * a
        
    Args:
        vehicle: Vehicle parameters
        cycle: Drive cycle data
        
    Returns:
        SimulationResults object with all calculated values
    """
    n = len(cycle)
    
    # Extract arrays
    time = np.array([p.time for p in cycle])
    speed = np.array([p.speed for p in cycle])
    grade_percent = np.array([p.grade for p in cycle])
    
    # Convert grade from percent to radians
    grade_rad = np.arctan(grade_percent / 100)
    
    # Calculate acceleration (forward difference, zero at start)
    acceleration = np.zeros(n)
    dt = np.diff(time)
    dv = np.diff(speed)
    # Avoid division by zero
    valid_dt = dt > 0
    acceleration[1:][valid_dt] = dv[valid_dt] / dt[valid_dt]
    
    # Calculate forces
    # Aerodynamic drag: F = 0.5 * ρ * Cd * A * v²
    force_aero = 0.5 * AIR_DENSITY * vehicle.drag_coefficient * vehicle.frontal_area * speed**2
    
    # Rolling resistance: F = Crr * m * g * cos(θ)
    force_rolling = vehicle.rolling_resistance * vehicle.mass * GRAVITY * np.cos(grade_rad)
    
    # Grade force: F = m * g * sin(θ)
    force_grade = vehicle.mass * GRAVITY * np.sin(grade_rad)
    
    # Acceleration force: F = m * a
    force_accel = vehicle.mass * acceleration
    
    # Total force at wheels
    force_total = force_aero + force_rolling + force_grade + force_accel
    
    # Power = Force * Velocity
    power_raw = force_total * speed
    
    # Apply regenerative braking efficiency for negative power (braking)
    power = np.where(
        power_raw < 0,
        power_raw * vehicle.regen_efficiency,  # Regen: only recover portion of energy
        power_raw
    )
    
    # Add continuous auxiliary power drain (HVAC, electronics, etc.)
    power = power + vehicle.auxiliary_power
    
    # Calculate cumulative energy (trapezoidal integration)
    dt_full = np.diff(time, prepend=0)
    energy_cumulative = np.cumsum(power * dt_full)
    
    return SimulationResults(
        time=time,
        speed=speed,
        grade=grade_percent,
        acceleration=acceleration,
        power=power,
        energy_cumulative=energy_cumulative,
        force_aero=force_aero,
        force_rolling=force_rolling,
        force_grade=force_grade,
        force_accel=force_accel,
        force_total=force_total
    )


# =============================================================================
# Output Functions
# =============================================================================
def create_output_directory(base_path: str, cycle_name: str, timestamp: str) -> str:
    """Create outputs directory with run-specific subfolder."""
    run_folder = f"{cycle_name}_{timestamp}"
    full_path = os.path.join(base_path, run_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def generate_plots(
    results: SimulationResults,
    cycle_name: str,
    output_dir: str
) -> str:
    """
    Generate and save plots.
    
    Creates a 2x2 subplot figure with:
        - Speed vs Time
        - Power vs Time
        - Cumulative Energy vs Time
        - Grade vs Time
        
    Returns:
        Path to saved plot file
    """
    # Distance vector (miles) for x-axis
    dt = np.diff(results.time, prepend=0)
    distance_m = np.cumsum(results.speed * dt)
    distance_miles = distance_m / 1609.34

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Road Load Simulation: {cycle_name}', fontsize=14, fontweight='bold')
    
    # Speed vs Time
    ax1 = axes[0, 0]
    ax1.plot(distance_miles, results.speed * MPS_TO_MPH, 'b-', linewidth=1.5, label='mph')
    ax1.set_xlabel('Distance (miles)')
    ax1.set_ylabel('Speed (mph)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Vehicle Speed')
    # Secondary y-axis for m/s
    ax1_twin = ax1.twinx()
    ax1_twin.plot(distance_miles, results.speed, 'b-', linewidth=1.5, alpha=0)  # Invisible, just for scale
    ax1_twin.set_ylabel('Speed (m/s)', color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')
    
    # Power vs Time
    ax2 = axes[0, 1]
    power_kw = results.power / 1000
    power_hp = results.power * W_TO_HP
    ax2.plot(distance_miles, power_kw, 'r-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.fill_between(distance_miles, 0, power_kw, where=power_kw >= 0, 
                     color='red', alpha=0.3, label='Traction')
    ax2.fill_between(distance_miles, 0, power_kw, where=power_kw < 0,
                     color='green', alpha=0.3, label='Regen')
    ax2.set_xlabel('Distance (miles)')
    ax2.set_ylabel('Power (kW)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_title('Power Demand')
    # Secondary y-axis for hp
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Power (hp)', color='gray')
    ax2_twin.set_ylim(ax2.get_ylim()[0] * W_TO_HP * 1000, ax2.get_ylim()[1] * W_TO_HP * 1000)
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    
    # Cumulative Energy vs Time
    ax3 = axes[1, 0]
    energy_kwh = results.energy_cumulative / 3.6e6  # J to kWh
    ax3.plot(distance_miles, energy_kwh, 'g-', linewidth=1.5)
    ax3.set_xlabel('Distance (miles)')
    ax3.set_ylabel('Cumulative Energy (kWh)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Energy Consumption')
    # Mark final value
    ax3.annotate(f'{energy_kwh[-1]:.3f} kWh', 
                 xy=(distance_miles[-1], energy_kwh[-1]),
                 xytext=(-50, 10), textcoords='offset points',
                 fontsize=10, color='g',
                 arrowprops=dict(arrowstyle='->', color='g', alpha=0.5))
    
    # Grade vs Time
    ax4 = axes[1, 1]
    ax4.plot(distance_miles, results.grade, 'm-', linewidth=1.5)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.fill_between(distance_miles, 0, results.grade, alpha=0.3, color='m')
    ax4.set_xlabel('Distance (miles)')
    ax4.set_ylabel('Grade (%)')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Road Grade')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = "plots.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def generate_erev_plots(
    base_results: SimulationResults,
    erev_results: EREVResults,
    cycle_name: str,
    output_dir: str
) -> str:
    """Generate EREV-specific plots (distance-based).

    Subplots:
      - Speed vs distance
      - SOC vs distance
      - Generator output vs distance (with on/off fill)
      - Fuel remaining vs distance
    """
    dt = np.diff(base_results.time, prepend=0)
    distance_miles = np.cumsum(base_results.speed * dt) / 1609.34

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EREV Simulation: {cycle_name}', fontsize=14, fontweight='bold')

    # Speed
    axes[0, 0].plot(distance_miles, base_results.speed * MPS_TO_MPH, color='b', linewidth=1.4)
    axes[0, 0].set_xlabel('Distance (miles)')
    axes[0, 0].set_ylabel('Speed (mph)')
    axes[0, 0].set_title('Vehicle Speed')
    axes[0, 0].grid(True, alpha=0.3)

    # SOC
    axes[0, 1].plot(erev_results.distance_miles_trace, erev_results.soc_timeline, color='g', linewidth=1.4)
    axes[0, 1].axhline(erev_results.min_soc, color='r', linestyle='--', linewidth=0.8, label='Min SOC')
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('SOC (%)')
    axes[0, 1].set_title('State of Charge')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Generator output
    axes[1, 0].plot(erev_results.distance_miles_trace, erev_results.generator_output_kw, color='orange', linewidth=1.4, label='Generator kW')
    axes[1, 0].fill_between(
        erev_results.distance_miles_trace,
        0,
        erev_results.generator_output_kw,
        where=erev_results.generator_on_flags > 0,
        color='orange', alpha=0.25, label='Generator On'
    )
    axes[1, 0].set_xlabel('Distance (miles)')
    axes[1, 0].set_ylabel('Power (kW)')
    axes[1, 0].set_title('Generator Output')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)

    # Fuel remaining
    axes[1, 1].plot(erev_results.distance_miles_trace, erev_results.fuel_remaining_gal, color='purple', linewidth=1.4)
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Fuel Remaining (gal)')
    axes[1, 1].set_title('Fuel Remaining')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "erev_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def generate_erev_range_plots(
    range_data: dict,
    cycle_name: str,
    output_dir: str
) -> str:
    """Generate plots for EREV range estimation traces."""
    distance = range_data.get('distance_miles_trace', np.array([]))
    soc = range_data.get('soc_trace', np.array([]))
    gen_kw = range_data.get('generator_output_kw', np.array([]))
    gen_on = range_data.get('generator_on_flags', np.array([]))
    fuel_rem = range_data.get('fuel_remaining_gal', np.array([]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EREV Range Estimation: {cycle_name}', fontsize=14, fontweight='bold')

    # SOC
    axes[0, 0].plot(distance, soc, color='g', linewidth=1.4)
    axes[0, 0].set_xlabel('Distance (miles)')
    axes[0, 0].set_ylabel('SOC (%)')
    axes[0, 0].set_title('State of Charge')
    axes[0, 0].grid(True, alpha=0.3)

    # Generator output
    axes[0, 1].plot(distance, gen_kw, color='orange', linewidth=1.2, label='Generator kW')
    axes[0, 1].fill_between(distance, 0, gen_kw, where=gen_on > 0, color='orange', alpha=0.25, label='Generator On')
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('Power (kW)')
    axes[0, 1].set_title('Generator Output')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Fuel remaining
    axes[1, 0].plot(distance, fuel_rem, color='purple', linewidth=1.2)
    axes[1, 0].set_xlabel('Distance (miles)')
    axes[1, 0].set_ylabel('Fuel Remaining (gal)')
    axes[1, 0].set_title('Fuel Remaining')
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative distance (simple visual)
    axes[1, 1].plot(distance, distance, color='blue', linewidth=1.2)
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Distance (miles)')
    axes[1, 1].set_title('Distance Progression')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "erev_range_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def generate_erev_range_detailed_plots(
    vehicle: VehicleParams,
    cycle_filepath: str,
    range_data: dict,
    cycle_name: str,
    output_dir: str
) -> str:
    """Generate detailed plots for EREV range estimation (Speed, Power, Energy, Grade vs Distance)."""
    # Load the drive cycle to get the full data
    cycle = load_drive_cycle(cycle_filepath)
    
    # Run the simulation to get detailed results
    results = calculate_road_load(vehicle, cycle)
    
    # Calculate distance
    dt = np.diff(results.time, prepend=0)
    distance_m = np.cumsum(results.speed * dt)
    distance_miles = distance_m / 1609.34
    
    # Get the range to know how many cycles were completed
    cycles_completed = range_data.get('cycles_completed', 1)
    
    # Replicate the data for all completed cycles
    if cycles_completed > 1:
        full_distance = []
        full_speed = []
        full_power = []
        full_energy = []
        full_grade = []
        
        cycle_distance = distance_miles[-1]
        cycle_energy = results.energy_cumulative[-1]
        
        for i in range(cycles_completed):
            offset_distance = i * cycle_distance
            offset_energy = i * cycle_energy
            full_distance.extend(distance_miles + offset_distance)
            full_speed.extend(results.speed * MPS_TO_MPH)
            full_power.extend(results.power / 1000)
            full_energy.extend((results.energy_cumulative + offset_energy) / 3.6e6)
            full_grade.extend(results.grade)
        
        distance_miles = np.array(full_distance)
        speed_mph = np.array(full_speed)
        power_kw = np.array(full_power)
        energy_kwh = np.array(full_energy)
        grade = np.array(full_grade)
    else:
        speed_mph = results.speed * MPS_TO_MPH
        power_kw = results.power / 1000
        energy_kwh = results.energy_cumulative / 3.6e6
        grade = results.grade
    
    # Create 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EREV Range Analysis Detail: {cycle_name}', fontsize=14, fontweight='bold')
    
    # Speed vs Distance
    axes[0, 0].plot(distance_miles, speed_mph, color='b', linewidth=1.2)
    axes[0, 0].set_xlabel('Distance (miles)')
    axes[0, 0].set_ylabel('Speed (mph)')
    axes[0, 0].set_title('Speed vs Distance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power vs Distance
    axes[0, 1].plot(distance_miles, power_kw, color='purple', linewidth=1.0)
    axes[0, 1].fill_between(distance_miles, 0, power_kw, where=power_kw > 0, 
                            color='red', alpha=0.3, label='Traction')
    axes[0, 1].fill_between(distance_miles, 0, power_kw, where=power_kw < 0, 
                            color='green', alpha=0.3, label='Regen')
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('Power (kW)')
    axes[0, 1].set_title('Power vs Distance')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative Energy vs Distance
    axes[1, 0].plot(distance_miles, energy_kwh, color='orange', linewidth=1.4)
    axes[1, 0].set_xlabel('Distance (miles)')
    axes[1, 0].set_ylabel('Energy (kWh)')
    axes[1, 0].set_title('Cumulative Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Grade vs Distance
    axes[1, 1].fill_between(distance_miles, 0, grade, color='brown', alpha=0.4)
    axes[1, 1].plot(distance_miles, grade, color='brown', linewidth=1.2)
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Grade (%)')
    axes[1, 1].set_title('Road Grade')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "erev_range_detailed_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def generate_bev_range_detailed_plots(
    vehicle: VehicleParams,
    cycle_filepath: str,
    range_data: dict,
    cycle_name: str,
    output_dir: str
) -> str:
    """Generate detailed plots for BEV range estimation (Speed, Power, Energy, Grade vs Distance)."""
    # Load the drive cycle to get the full data
    cycle = load_drive_cycle(cycle_filepath)
    
    # Run the simulation to get detailed results
    results = calculate_road_load(vehicle, cycle)
    
    # Calculate distance
    dt = np.diff(results.time, prepend=0)
    distance_m = np.cumsum(results.speed * dt)
    distance_miles = distance_m / 1609.34
    
    # Get the range to know how many cycles were completed
    cycles_completed = range_data.get('cycles_completed', range_data.get('total_cycles', 1))
    
    # Replicate the data for all completed cycles
    if cycles_completed > 1:
        full_distance = []
        full_speed = []
        full_power = []
        full_energy = []
        full_grade = []
        
        cycle_distance = distance_miles[-1]
        cycle_energy = results.energy_cumulative[-1]
        
        for i in range(cycles_completed):
            offset_distance = i * cycle_distance
            offset_energy = i * cycle_energy
            full_distance.extend(distance_miles + offset_distance)
            full_speed.extend(results.speed * MPS_TO_MPH)
            full_power.extend(results.power / 1000)
            full_energy.extend((results.energy_cumulative + offset_energy) / 3.6e6)
            full_grade.extend(results.grade)
        
        distance_miles = np.array(full_distance)
        speed_mph = np.array(full_speed)
        power_kw = np.array(full_power)
        energy_kwh = np.array(full_energy)
        grade = np.array(full_grade)
    else:
        speed_mph = results.speed * MPS_TO_MPH
        power_kw = results.power / 1000
        energy_kwh = results.energy_cumulative / 3.6e6
        grade = results.grade
    
    # Create 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BEV Range Analysis Detail: {cycle_name}', fontsize=14, fontweight='bold')
    
    # Speed vs Distance
    axes[0, 0].plot(distance_miles, speed_mph, color='b', linewidth=1.2)
    axes[0, 0].set_xlabel('Distance (miles)')
    axes[0, 0].set_ylabel('Speed (mph)')
    axes[0, 0].set_title('Speed vs Distance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power vs Distance
    axes[0, 1].plot(distance_miles, power_kw, color='purple', linewidth=1.0)
    axes[0, 1].fill_between(distance_miles, 0, power_kw, where=power_kw > 0, 
                            color='red', alpha=0.3, label='Traction')
    axes[0, 1].fill_between(distance_miles, 0, power_kw, where=power_kw < 0, 
                            color='green', alpha=0.3, label='Regen')
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('Power (kW)')
    axes[0, 1].set_title('Power vs Distance')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative Energy vs Distance
    axes[1, 0].plot(distance_miles, energy_kwh, color='orange', linewidth=1.4)
    axes[1, 0].set_xlabel('Distance (miles)')
    axes[1, 0].set_ylabel('Energy (kWh)')
    axes[1, 0].set_title('Cumulative Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Grade vs Distance
    axes[1, 1].fill_between(distance_miles, 0, grade, color='brown', alpha=0.4)
    axes[1, 1].plot(distance_miles, grade, color='brown', linewidth=1.2)
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Grade (%)')
    axes[1, 1].set_title('Road Grade')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bev_range_detailed_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def save_erev_range_html(
    vehicle: VehicleParams,
    range_data: dict,
    cycle_name: str,
    output_dir: str,
    plot_filename: str = "erev_range_plots.png",
    detailed_plot_filename: str = None
) -> str:
    """Generate and save HTML report for EREV range estimation."""
    
    # Extract range data
    range_miles = range_data.get('range_miles', 0)
    range_km = range_data.get('range_km', 0)
    ev_only_miles = range_data.get('ev_only_miles', 0)
    generator_miles = range_data.get('generator_miles', 0)
    battery_energy_kwh = range_data.get('battery_energy_kwh', 0)
    fuel_used_gallons = range_data.get('fuel_used_gallons', 0)
    final_soc = range_data.get('final_soc', 0)
    cycles_completed = range_data.get('cycles_completed', 0)
    mpge = range_data.get('mpge', 0)
    kwh_per_mile = range_data.get('kwh_per_mile', 0)
    
    # Calculate additional metrics
    ev_percentage = (ev_only_miles / range_miles * 100) if range_miles > 0 else 0
    generator_percentage = (generator_miles / range_miles * 100) if range_miles > 0 else 0
    fuel_economy_mpg = generator_miles / fuel_used_gallons if fuel_used_gallons > 0 else 0
    
    # Calculate energy breakdown
    fuel_energy_kwh = fuel_used_gallons * GASOLINE_ENERGY_KWH_PER_GALLON
    total_energy_kwh = battery_energy_kwh + fuel_energy_kwh
    battery_energy_pct = (battery_energy_kwh / total_energy_kwh * 100) if total_energy_kwh > 0 else 0
    fuel_energy_pct = (fuel_energy_kwh / total_energy_kwh * 100) if total_energy_kwh > 0 else 0
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EREV Range Report: {cycle_name}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #3498db;
        }}
        .card.range {{
            border-left-color: #9b59b6;
        }}
        .card.energy {{
            border-left-color: #27ae60;
        }}
        .card.fuel {{
            border-left-color: #e67e22;
        }}
        .card.efficiency {{
            border-left-color: #16a085;
        }}
        .card h3 {{
            color: #2c3e50;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .unit {{
            font-size: 0.5em;
            color: #7f8c8d;
            font-weight: normal;
        }}
        .secondary {{
            color: #95a5a6;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 0.85em;
            text-align: center;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            float: left;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .ev-portion {{
            background: linear-gradient(90deg, #27ae60, #2ecc71);
        }}
        .gen-portion {{
            background: linear-gradient(90deg, #e67e22, #f39c12);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡🔋 EREV Range Estimation Report</h1>
        <div class="meta">
            <strong>Drive Cycle:</strong> {cycle_name} &nbsp;|&nbsp;
            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
            <strong>EREV Mode:</strong> {vehicle.erev_mode}
        </div>

        <h2>📊 Total Range</h2>
        <div class="grid">
            <div class="card range">
                <h3>Total Range</h3>
                <div class="value">{range_miles:.1f} <span class="unit">mi</span></div>
                <div class="secondary">{range_km:.1f} km</div>
            </div>
            <div class="card range">
                <h3>EV-Only Range</h3>
                <div class="value">{ev_only_miles:.1f} <span class="unit">mi</span></div>
                <div class="secondary">{ev_percentage:.1f}% of total</div>
            </div>
            <div class="card range">
                <h3>Generator-Assisted Range</h3>
                <div class="value">{generator_miles:.1f} <span class="unit">mi</span></div>
                <div class="secondary">{generator_percentage:.1f}% of total</div>
            </div>
            <div class="card range">
                <h3>Cycles Completed</h3>
                <div class="value">{cycles_completed}</div>
                <div class="secondary">Drive cycles</div>
            </div>
        </div>

        <h2>🔌 Range Breakdown</h2>
        <div class="progress-bar">
            <div class="progress-fill ev-portion" style="width: {ev_percentage:.1f}%">
                EV: {ev_only_miles:.0f} mi
            </div>
            <div class="progress-fill gen-portion" style="width: {generator_percentage:.1f}%">
                Generator: {generator_miles:.0f} mi
            </div>
        </div>

        <h2>⚡ Energy Consumption</h2>
        <div class="grid">
            <div class="card energy">
                <h3>Battery Energy Used</h3>
                <div class="value">{battery_energy_kwh:.2f} <span class="unit">kWh</span></div>
                <div class="secondary">{battery_energy_pct:.1f}% of total energy</div>
            </div>
            <div class="card energy">
                <h3>Fuel Energy Used</h3>
                <div class="value">{fuel_energy_kwh:.2f} <span class="unit">kWh</span></div>
                <div class="secondary">{fuel_energy_pct:.1f}% of total energy</div>
            </div>
            <div class="card energy">
                <h3>Total Energy</h3>
                <div class="value">{total_energy_kwh:.2f} <span class="unit">kWh</span></div>
                <div class="secondary">{kwh_per_mile:.3f} kWh/mi</div>
            </div>
            <div class="card energy">
                <h3>Final State of Charge</h3>
                <div class="value">{final_soc:.1f} <span class="unit">%</span></div>
                <div class="secondary">Battery SOC</div>
            </div>
        </div>

        <h2>⛽ Fuel Consumption</h2>
        <div class="grid">
            <div class="card fuel">
                <h3>Fuel Used</h3>
                <div class="value">{fuel_used_gallons:.3f} <span class="unit">gal</span></div>
                <div class="secondary">{fuel_used_gallons * 3.78541:.2f} liters</div>
            </div>
            <div class="card fuel">
                <h3>Generator Fuel Economy</h3>
                <div class="value">{fuel_economy_mpg:.1f} <span class="unit">MPG</span></div>
                <div class="secondary">During generator operation</div>
            </div>
            <div class="card fuel">
                <h3>Tank Capacity</h3>
                <div class="value">{vehicle.fuel_tank_gallons:.1f} <span class="unit">gal</span></div>
                <div class="secondary">{(fuel_used_gallons/vehicle.fuel_tank_gallons*100):.1f}% used</div>
            </div>
            <div class="card fuel">
                <h3>Fuel Remaining</h3>
                <div class="value">{vehicle.fuel_tank_gallons - fuel_used_gallons:.2f} <span class="unit">gal</span></div>
                <div class="secondary">{((1-fuel_used_gallons/vehicle.fuel_tank_gallons)*100):.1f}% remaining</div>
            </div>
        </div>

        <h2>📈 Efficiency Metrics</h2>
        <div class="grid">
            <div class="card efficiency">
                <h3>Combined MPGe</h3>
                <div class="value">{mpge:.1f} <span class="unit">MPGe</span></div>
                <div class="secondary">Miles per gallon equivalent</div>
            </div>
            <div class="card efficiency">
                <h3>Energy per Mile</h3>
                <div class="value">{kwh_per_mile:.3f} <span class="unit">kWh/mi</span></div>
                <div class="secondary">{kwh_per_mile * 1000:.1f} Wh/mi</div>
            </div>
            <div class="card efficiency">
                <h3>Battery Efficiency</h3>
                <div class="value">{(battery_energy_kwh/ev_only_miles if ev_only_miles > 0 else 0):.3f} <span class="unit">kWh/mi</span></div>
                <div class="secondary">EV-only portion</div>
            </div>
            <div class="card efficiency">
                <h3>Generator Efficiency</h3>
                <div class="value">{vehicle.bsfc_g_kwh:.0f} <span class="unit">g/kWh</span></div>
                <div class="secondary">BSFC rating</div>
            </div>
        </div>

        <h2>🚙 Vehicle Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Vehicle Class</td>
                <td>EREV (Extended Range Electric Vehicle)</td>
            </tr>
            <tr>
                <td>EREV Mode</td>
                <td>{vehicle.erev_mode.replace('_', ' ').title()}</td>
            </tr>
            <tr>
                <td>Mass</td>
                <td>{vehicle.mass:.1f} kg ({vehicle.mass * KG_TO_LB:.1f} lb)</td>
            </tr>
            <tr>
                <td>Frontal Area</td>
                <td>{vehicle.frontal_area:.2f} m² ({vehicle.frontal_area * M2_TO_FT2:.2f} ft²)</td>
            </tr>
            <tr>
                <td>Drag Coefficient</td>
                <td>{vehicle.drag_coefficient:.3f}</td>
            </tr>
            <tr>
                <td>Rolling Resistance</td>
                <td>{vehicle.rolling_resistance:.4f}</td>
            </tr>
            <tr>
                <td>Battery Capacity</td>
                <td>{vehicle.battery_capacity:.1f} kWh ({vehicle.usable_battery_pct:.0f}% usable)</td>
            </tr>
            <tr>
                <td>Generator Power</td>
                <td>{vehicle.generator_power_kw:.1f} kW</td>
            </tr>
            <tr>
                <td>Fuel Tank Capacity</td>
                <td>{vehicle.fuel_tank_gallons:.1f} gallons</td>
            </tr>
            <tr>
                <td>SOC Sustain Threshold</td>
                <td>{vehicle.soc_sustain_pct:.0f}%</td>
            </tr>
            <tr>
                <td>Regen Efficiency</td>
                <td>{vehicle.regen_efficiency * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Auxiliary Power</td>
                <td>{vehicle.auxiliary_power:.0f} W</td>
            </tr>
        </table>

        <h2>📈 Range Estimation Plots</h2>
        <div class="plot-container">
            <img src="{plot_filename}" alt="EREV Range Plots">
        </div>
        
        {f'<h2>📊 Detailed Drive Cycle Analysis</h2><div class="plot-container"><img src="{detailed_plot_filename}" alt="EREV Detailed Plots"></div>' if detailed_plot_filename else ''}

        <div class="footer">
            Generated by Road Load Simulator - EREV Range Analysis | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    filename = "erev_range_report.html"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    return filepath


def generate_bev_range_plots(
    range_data: dict,
    cycle_name: str,
    output_dir: str
) -> str:
    """Generate a simple BEV range summary plot (bars for range and energy)."""
    range_mi = range_data.get('range_miles', 0)
    energy_per_mile = range_data.get('energy_per_mile', 0)
    energy_used = range_data.get('energy_used_kwh', 0)
    cycles_completed = range_data.get('cycles_completed', range_data.get('total_cycles', 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'BEV Range Summary: {cycle_name}', fontsize=14, fontweight='bold')

    axes[0].bar(['Range'], [range_mi], color='#2c7be5')
    axes[0].set_ylabel('Miles')
    axes[0].set_title('Estimated Range')
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(['Energy/mi', 'Energy Used'], [energy_per_mile, energy_used], color=['#52b788', '#f6aa1c'])
    axes[1].set_ylabel('kWh')
    axes[1].set_title('Energy Metrics')
    axes[1].grid(True, axis='y', alpha=0.3)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(0)

    fig.text(0.02, 0.02, f"Cycles completed: {cycles_completed}", fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bev_range_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def save_bev_range_html(
    vehicle: VehicleParams,
    range_data: dict,
    cycle_name: str,
    output_dir: str,
    plot_filename: str = "bev_range_plots.png",
    detailed_plot_filename: str = None
) -> str:
    """Generate and save HTML report for BEV range estimation."""
    
    # Extract range data
    range_miles = range_data.get('range_miles', 0)
    range_km = range_data.get('range_km', 0)
    energy_per_mile = range_data.get('energy_per_mile', 0)
    energy_per_km = range_data.get('energy_per_km', 0)
    energy_used_kwh = range_data.get('energy_used_kwh', 0)
    cycles_completed = range_data.get('cycles_completed', range_data.get('total_cycles', 0))
    final_soc = range_data.get('final_soc', 0)
    
    # Calculate additional metrics
    usable_capacity_kwh = vehicle.battery_capacity * (vehicle.usable_battery_pct / 100.0)
    battery_used_pct = (energy_used_kwh / usable_capacity_kwh * 100) if usable_capacity_kwh > 0 else 0
    wh_per_mile = energy_per_mile * 1000
    mpge = range_miles / (energy_used_kwh / 33.7) if energy_used_kwh > 0 else 0
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Range Report: {cycle_name}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #3498db;
        }}
        .card.range {{
            border-left-color: #2c7be5;
        }}
        .card.energy {{
            border-left-color: #27ae60;
        }}
        .card.efficiency {{
            border-left-color: #16a085;
        }}
        .card.battery {{
            border-left-color: #9b59b6;
        }}
        .card h3 {{
            color: #2c3e50;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .unit {{
            font-size: 0.5em;
            color: #7f8c8d;
            font-weight: normal;
        }}
        .secondary {{
            color: #95a5a6;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 0.85em;
            text-align: center;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #2c7be5, #3498db);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔋 BEV Range Estimation Report</h1>
        <div class="meta">
            <strong>Drive Cycle:</strong> {cycle_name} &nbsp;|&nbsp;
            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
            <strong>Vehicle Type:</strong> Battery Electric Vehicle
        </div>

        <h2>📊 Range Summary</h2>
        <div class="grid">
            <div class="card range">
                <h3>Estimated Range</h3>
                <div class="value">{range_miles:.1f} <span class="unit">mi</span></div>
                <div class="secondary">{range_km:.1f} km</div>
            </div>
            <div class="card range">
                <h3>Cycles Completed</h3>
                <div class="value">{cycles_completed}</div>
                <div class="secondary">Drive cycles</div>
            </div>
            <div class="card battery">
                <h3>Battery Capacity Used</h3>
                <div class="value">{battery_used_pct:.1f} <span class="unit">%</span></div>
                <div class="secondary">{energy_used_kwh:.2f} kWh of {usable_capacity_kwh:.1f} kWh</div>
            </div>
            <div class="card battery">
                <h3>Final SOC</h3>
                <div class="value">{final_soc:.1f} <span class="unit">%</span></div>
                <div class="secondary">State of charge</div>
            </div>
        </div>

        <h2>🔌 Battery Usage</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {battery_used_pct:.1f}%">
                {energy_used_kwh:.2f} kWh used ({battery_used_pct:.1f}%)
            </div>
        </div>

        <h2>⚡ Energy Consumption</h2>
        <div class="grid">
            <div class="card energy">
                <h3>Energy Used</h3>
                <div class="value">{energy_used_kwh:.2f} <span class="unit">kWh</span></div>
                <div class="secondary">{energy_used_kwh * 3.6:.1f} MJ</div>
            </div>
            <div class="card energy">
                <h3>Energy per Mile</h3>
                <div class="value">{energy_per_mile:.3f} <span class="unit">kWh/mi</span></div>
                <div class="secondary">{wh_per_mile:.1f} Wh/mi</div>
            </div>
            <div class="card energy">
                <h3>Energy per Kilometer</h3>
                <div class="value">{energy_per_km:.3f} <span class="unit">kWh/km</span></div>
                <div class="secondary">{energy_per_km * 1000:.1f} Wh/km</div>
            </div>
            <div class="card energy">
                <h3>Battery Capacity</h3>
                <div class="value">{vehicle.battery_capacity:.1f} <span class="unit">kWh</span></div>
                <div class="secondary">{vehicle.usable_battery_pct:.0f}% usable ({usable_capacity_kwh:.1f} kWh)</div>
            </div>
        </div>

        <h2>📈 Efficiency Metrics</h2>
        <div class="grid">
            <div class="card efficiency">
                <h3>MPGe</h3>
                <div class="value">{mpge:.1f} <span class="unit">MPGe</span></div>
                <div class="secondary">Miles per gallon equivalent</div>
            </div>
            <div class="card efficiency">
                <h3>Wh per Mile</h3>
                <div class="value">{wh_per_mile:.0f} <span class="unit">Wh/mi</span></div>
                <div class="secondary">Energy efficiency</div>
            </div>
            <div class="card efficiency">
                <h3>Miles per kWh</h3>
                <div class="value">{(1/energy_per_mile if energy_per_mile > 0 else 0):.2f} <span class="unit">mi/kWh</span></div>
                <div class="secondary">Distance efficiency</div>
            </div>
            <div class="card efficiency">
                <h3>Regen Efficiency</h3>
                <div class="value">{vehicle.regen_efficiency * 100:.0f} <span class="unit">%</span></div>
                <div class="secondary">Regenerative braking</div>
            </div>
        </div>

        <h2>🚙 Vehicle Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Vehicle Class</td>
                <td>BEV (Battery Electric Vehicle)</td>
            </tr>
            <tr>
                <td>Mass</td>
                <td>{vehicle.mass:.1f} kg ({vehicle.mass * KG_TO_LB:.1f} lb)</td>
            </tr>
            <tr>
                <td>Frontal Area</td>
                <td>{vehicle.frontal_area:.2f} m² ({vehicle.frontal_area * M2_TO_FT2:.2f} ft²)</td>
            </tr>
            <tr>
                <td>Drag Coefficient</td>
                <td>{vehicle.drag_coefficient:.3f}</td>
            </tr>
            <tr>
                <td>Rolling Resistance</td>
                <td>{vehicle.rolling_resistance:.4f}</td>
            </tr>
            <tr>
                <td>Battery Capacity (Total)</td>
                <td>{vehicle.battery_capacity:.1f} kWh</td>
            </tr>
            <tr>
                <td>Usable Battery Capacity</td>
                <td>{usable_capacity_kwh:.1f} kWh ({vehicle.usable_battery_pct:.0f}%)</td>
            </tr>
            <tr>
                <td>Regen Efficiency</td>
                <td>{vehicle.regen_efficiency * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Auxiliary Power</td>
                <td>{vehicle.auxiliary_power:.0f} W</td>
            </tr>
        </table>

        <h2>📈 Range Estimation Plots</h2>
        <div class="plot-container">
            <img src="{plot_filename}" alt="BEV Range Plots">
        </div>
        
        {f'<h2>📊 Detailed Drive Cycle Analysis</h2><div class="plot-container"><img src="{detailed_plot_filename}" alt="BEV Detailed Plots"></div>' if detailed_plot_filename else ''}

        <div class="footer">
            Generated by Road Load Simulator - BEV Range Analysis | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    filename = "bev_range_report.html"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    return filepath


def generate_bev_plots(
    results: SimulationResults,
    cycle_name: str,
    output_dir: str
) -> str:
    """Generate BEV-focused plots including efficiency vs distance."""
    dt = np.diff(results.time, prepend=0)
    distance_m = np.cumsum(results.speed * dt)
    distance_miles = distance_m / 1609.34
    power_kw = results.power / 1000
    energy_kwh = results.energy_cumulative / 3.6e6

    # Cumulative efficiency (Wh/mi)
    with np.errstate(divide='ignore', invalid='ignore'):
        wh_per_mile = np.where(distance_miles > 0, energy_kwh * 1000 / distance_miles, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BEV Simulation: {cycle_name}', fontsize=14, fontweight='bold')

    # Speed
    axes[0, 0].plot(distance_miles, results.speed * MPS_TO_MPH, color='b', linewidth=1.4)
    axes[0, 0].set_xlabel('Distance (miles)')
    axes[0, 0].set_ylabel('Speed (mph)')
    axes[0, 0].set_title('Vehicle Speed')
    axes[0, 0].grid(True, alpha=0.3)

    # Power
    axes[0, 1].plot(distance_miles, power_kw, color='r', linewidth=1.4)
    axes[0, 1].axhline(0, color='k', linewidth=0.5)
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('Power (kW)')
    axes[0, 1].set_title('Power Demand')
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative Energy
    axes[1, 0].plot(distance_miles, energy_kwh, color='g', linewidth=1.4)
    axes[1, 0].set_xlabel('Distance (miles)')
    axes[1, 0].set_ylabel('Energy (kWh)')
    axes[1, 0].set_title('Cumulative Energy')
    axes[1, 0].grid(True, alpha=0.3)

    # Efficiency
    axes[1, 1].plot(distance_miles, wh_per_mile, color='purple', linewidth=1.4)
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Wh/mi')
    axes[1, 1].set_title('Cumulative Efficiency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bev_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def save_erev_timeseries_csv(
    erev_trace: dict,
    output_dir: str,
    filename: str = "erev_timeseries.csv"
) -> str:
    """Save EREV time-series data (SOC, generator, fuel) to CSV.

    Expects keys: time_trace, distance_miles_trace, soc_trace, generator_output_kw, generator_on_flags, fuel_remaining_gal.
    """
    filepath = os.path.join(output_dir, filename)
    time_arr = erev_trace.get('time_trace', np.array([]))
    distance = erev_trace.get('distance_miles_trace', np.array([]))
    soc = erev_trace.get('soc_trace', np.array([]))
    gen_kw = erev_trace.get('generator_output_kw', np.array([]))
    gen_on = erev_trace.get('generator_on_flags', np.array([]))
    fuel = erev_trace.get('fuel_remaining_gal', np.array([]))

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time_s', 'distance_miles', 'soc_pct', 'generator_kw', 'generator_on_flag', 'fuel_remaining_gal'
        ])
        for i in range(len(time_arr)):
            writer.writerow([
                f"{time_arr[i]:.2f}",
                f"{distance[i]:.5f}",
                f"{soc[i]:.3f}",
                f"{gen_kw[i]:.3f}",
                int(gen_on[i]),
                f"{fuel[i]:.4f}"
            ])
    return filepath


def save_timeseries_csv(
    results: SimulationResults,
    output_dir: str
) -> str:
    """Save full time-series data to CSV."""
    filename = "timeseries.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time_s', 'speed_mps', 'speed_mph', 'grade_percent',
            'acceleration_mps2', 'power_W', 'power_kW', 'power_hp',
            'energy_cumulative_J', 'energy_cumulative_kWh',
            'force_aero_N', 'force_rolling_N', 'force_grade_N',
            'force_accel_N', 'force_total_N'
        ])
        
        for i in range(len(results.time)):
            writer.writerow([
                f"{results.time[i]:.2f}",
                f"{results.speed[i]:.4f}",
                f"{results.speed[i] * MPS_TO_MPH:.4f}",
                f"{results.grade[i]:.2f}",
                f"{results.acceleration[i]:.4f}",
                f"{results.power[i]:.2f}",
                f"{results.power[i] / 1000:.4f}",
                f"{results.power[i] * W_TO_HP:.4f}",
                f"{results.energy_cumulative[i]:.2f}",
                f"{results.energy_cumulative[i] / 3.6e6:.6f}",
                f"{results.force_aero[i]:.2f}",
                f"{results.force_rolling[i]:.2f}",
                f"{results.force_grade[i]:.2f}",
                f"{results.force_accel[i]:.2f}",
                f"{results.force_total[i]:.2f}"
            ])
    
    return filepath


def generate_summary_text(
    vehicle: VehicleParams,
    results: SimulationResults,
    cycle_name: str,
    cycle_filepath: str
) -> str:
    """Generate summary text for console and file output."""
    
    # Calculate values in both units
    peak_traction_kw = results.peak_power_traction / 1000
    peak_traction_hp = results.peak_power_traction * W_TO_HP
    peak_regen_kw = results.peak_power_regen / 1000
    peak_regen_hp = results.peak_power_regen * W_TO_HP
    avg_power_kw = results.average_power / 1000
    avg_power_hp = results.average_power * W_TO_HP
    total_energy_kwh = results.total_energy / 3.6e6
    total_energy_mj = results.total_energy / 1e6
    traction_energy_kwh = results.traction_energy / 3.6e6
    regen_energy_kwh = results.regen_energy / 3.6e6
    
    duration = results.time[-1] - results.time[0]
    max_speed_mps = np.max(results.speed)
    max_speed_mph = max_speed_mps * MPS_TO_MPH
    avg_speed_mps = np.mean(results.speed)
    avg_speed_mph = avg_speed_mps * MPS_TO_MPH
    
    summary = f"""
{'='*70}
ROAD LOAD SIMULATION RESULTS
{'='*70}

Drive Cycle: {cycle_name}
Source File: {cycle_filepath}
Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'-'*70}
VEHICLE PARAMETERS
{'-'*70}
{vehicle}

{'-'*70}
DRIVE CYCLE STATISTICS
{'-'*70}
  Duration: {duration:.1f} s ({duration/60:.2f} min)
  Max Speed: {max_speed_mps:.2f} m/s ({max_speed_mph:.1f} mph)
  Avg Speed: {avg_speed_mps:.2f} m/s ({avg_speed_mph:.1f} mph)
  Grade Range: {np.min(results.grade):.1f}% to {np.max(results.grade):.1f}%

{'-'*70}
POWER RESULTS
{'-'*70}
  Peak Traction Power: {peak_traction_kw:.2f} kW ({peak_traction_hp:.1f} hp)
  Peak Regen Power:    {peak_regen_kw:.2f} kW ({peak_regen_hp:.1f} hp)
  Average Power:       {avg_power_kw:.2f} kW ({avg_power_hp:.1f} hp)

{'-'*70}
ENERGY RESULTS
{'-'*70}
  Total Net Energy:    {total_energy_kwh:.4f} kWh ({total_energy_mj:.3f} MJ)
  Traction Energy:     {traction_energy_kwh:.4f} kWh
  Regenerated Energy:  {regen_energy_kwh:.4f} kWh
  Regen Efficiency:    {vehicle.regen_efficiency*100:.1f}%
  Energy Recovery:     {(regen_energy_kwh/traction_energy_kwh*100) if traction_energy_kwh > 0 else 0:.1f}% of traction energy

{'='*70}
"""
    return summary


def save_summary_text(
    summary: str,
    output_dir: str
) -> str:
    """Save summary to text file."""
    filename = "summary.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(summary)
    
    return filepath


def save_summary_html(
    vehicle: VehicleParams,
    results: SimulationResults,
    cycle_name: str,
    cycle_filepath: str,
    output_dir: str,
    plot_filename: str,
    erev_results: Optional[EREVResults] = None,
    secondary_plot_filename: Optional[str] = None
) -> str:
    """Generate and save HTML summary report.

    If `erev_results` is provided (vehicle_class == 'erev'), include generator/fuel/SOC details.
    """
    
    # Calculate values
    peak_traction_kw = results.peak_power_traction / 1000
    peak_traction_hp = results.peak_power_traction * W_TO_HP
    peak_regen_kw = results.peak_power_regen / 1000
    peak_regen_hp = results.peak_power_regen * W_TO_HP
    avg_power_kw = results.average_power / 1000
    avg_power_hp = results.average_power * W_TO_HP
    total_energy_kwh = results.total_energy / 3.6e6
    total_energy_mj = results.total_energy / 1e6
    traction_energy_kwh = results.traction_energy / 3.6e6
    regen_energy_kwh = results.regen_energy / 3.6e6
    
    duration = results.time[-1] - results.time[0]
    max_speed_mps = np.max(results.speed)
    max_speed_mph = max_speed_mps * MPS_TO_MPH
    avg_speed_mps = np.mean(results.speed)
    avg_speed_mph = avg_speed_mps * MPS_TO_MPH
    
    energy_recovery_pct = (regen_energy_kwh/traction_energy_kwh*100) if traction_energy_kwh > 0 else 0
    
    erev_block = ""
    if vehicle.vehicle_class == 'erev' and erev_results is not None:
        erev_block = f"""
        <h2>⚡ EREV Details</h2>
        <div class="grid">
            <div class="card energy">
                <h3>EV-Only Miles</h3>
                <div class="value">{erev_results.ev_only_miles:.2f} <span class="unit">mi</span></div>
                <div class="secondary">Generator miles: {erev_results.generator_miles:.2f} mi</div>
            </div>
            <div class="card energy">
                <h3>Generator Energy</h3>
                <div class="value">{erev_results.generator_energy_kwh:.2f} <span class="unit">kWh</span></div>
                <div class="secondary">Runtime: {erev_results.generator_runtime_seconds/60:.1f} min</div>
            </div>
            <div class="card energy">
                <h3>Fuel Used</h3>
                <div class="value">{erev_results.fuel_used_gallons:.3f} <span class="unit">gal</span></div>
                <div class="secondary">Final SOC: {erev_results.final_soc:.1f}% (min {erev_results.min_soc:.1f}%)</div>
            </div>
            <div class="card energy">
                <h3>MPGe</h3>
                <div class="value">{erev_results.mpge:.1f} <span class="unit">MPGe</span></div>
                <div class="secondary">Energy per mile: {erev_results.kwh_per_mile:.3f} kWh/mi</div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Load Simulation: {cycle_name}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;

        {f'<div class="plot-container"><img src="{secondary_plot_filename}" alt="Additional Plots"></div>' if secondary_plot_filename else ''}
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #3498db;
        }}
        .card.power {{
            border-left-color: #e74c3c;
        }}
        .card.energy {{
            border-left-color: #27ae60;
        }}
        .card h3 {{
            color: #2c3e50;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .unit {{
            font-size: 0.5em;
            color: #7f8c8d;
            font-weight: normal;
        }}
        .secondary {{
            color: #95a5a6;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 0.85em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Road Load Simulation Results</h1>
        <div class="meta">
            <strong>Drive Cycle:</strong> {cycle_name} &nbsp;|&nbsp;
            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
            <strong>Source:</strong> {os.path.basename(cycle_filepath)}
        </div>

        <h2>📊 Key Results</h2>
        <div class="grid">
            <div class="card power">
                <h3>Peak Traction Power</h3>
                <div class="value">{peak_traction_kw:.2f} <span class="unit">kW</span></div>
                <div class="secondary">{peak_traction_hp:.1f} hp</div>
            </div>
            <div class="card power">
                <h3>Peak Regen Power</h3>
                <div class="value">{peak_regen_kw:.2f} <span class="unit">kW</span></div>
                <div class="secondary">{peak_regen_hp:.1f} hp</div>
            </div>
            <div class="card power">
                <h3>Average Power</h3>
                <div class="value">{avg_power_kw:.2f} <span class="unit">kW</span></div>
                <div class="secondary">{avg_power_hp:.1f} hp</div>
            </div>
            <div class="card energy">
                <h3>Total Net Energy</h3>
                <div class="value">{total_energy_kwh:.4f} <span class="unit">kWh</span></div>
                <div class="secondary">{total_energy_mj:.3f} MJ</div>
            </div>
            <div class="card energy">
                <h3>Traction Energy</h3>
                <div class="value">{traction_energy_kwh:.4f} <span class="unit">kWh</span></div>
                <div class="secondary">Positive power only</div>
            </div>
            <div class="card energy">
                <h3>Regenerated Energy</h3>
                <div class="value">{regen_energy_kwh:.4f} <span class="unit">kWh</span></div>
                <div class="secondary">{energy_recovery_pct:.1f}% recovery</div>
            </div>
        </div>

        {erev_block}

        <h2>🚙 Vehicle Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>SI Value</th>
                <th>Imperial Value</th>
            </tr>
            <tr>
                <td>Mass</td>
                <td>{vehicle.mass:.1f} kg</td>
                <td>{vehicle.mass * KG_TO_LB:.1f} lb</td>
            </tr>
            <tr>
                <td>Frontal Area</td>
                <td>{vehicle.frontal_area:.2f} m²</td>
                <td>{vehicle.frontal_area * M2_TO_FT2:.2f} ft²</td>
            </tr>
            <tr>
                <td>Drag Coefficient</td>
                <td colspan="2">{vehicle.drag_coefficient:.3f}</td>
            </tr>
            <tr>
                <td>Rolling Resistance</td>
                <td colspan="2">{vehicle.rolling_resistance:.4f}</td>
            </tr>
            <tr>
                <td>Regen Efficiency</td>
                <td colspan="2">{vehicle.regen_efficiency * 100:.1f}%</td>
            </tr>
        </table>

        <h2>🛣️ Drive Cycle Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>SI Value</th>
                <th>Imperial Value</th>
            </tr>
            <tr>
                <td>Duration</td>
                <td>{duration:.1f} s</td>
                <td>{duration/60:.2f} min</td>
            </tr>
            <tr>
                <td>Max Speed</td>
                <td>{max_speed_mps:.2f} m/s</td>
                <td>{max_speed_mph:.1f} mph</td>
            </tr>
            <tr>
                <td>Avg Speed</td>
                <td>{avg_speed_mps:.2f} m/s</td>
                <td>{avg_speed_mph:.1f} mph</td>
            </tr>
            <tr>
                <td>Grade Range</td>
                <td colspan="2">{np.min(results.grade):.1f}% to {np.max(results.grade):.1f}%</td>
            </tr>
        </table>

        <h2>📈 Simulation Plots</h2>
        <div class="plot-container">
            <img src="{plot_filename}" alt="Simulation Plots">
        </div>

        <div class="footer">
            Generated by Road Load Simulator | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    filename = "report.html"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    return filepath


# =============================================================================
# Main Simulation Function
# =============================================================================
def run_simulation(
    vehicle: VehicleParams,
    cycle_filepath: str,
    cycle_name: Optional[str] = None,
    speed_unit: str = 'mps',
    output_dir: str = 'outputs'
) -> SimulationResults:
    """
    Run complete road load simulation.
    
    Args:
        vehicle: Vehicle parameters
        cycle_filepath: Path to drive cycle CSV file
        cycle_name: Name for the drive cycle (defaults to filename)
        speed_unit: 'mps' for m/s, 'mph' for miles per hour
        output_dir: Directory for output files
        
    Returns:
        SimulationResults object
    """
    # Get cycle name from filepath if not provided
    if cycle_name is None:
        cycle_name = os.path.splitext(os.path.basename(cycle_filepath))[0]
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    # Create output directory with run-specific subfolder
    run_output_dir = create_output_directory(output_dir, cycle_name, timestamp)
    
    # Load drive cycle
    print(f"\nLoading drive cycle: {cycle_filepath}")
    cycle = load_drive_cycle(cycle_filepath, speed_unit)
    print(f"  Loaded {len(cycle)} data points")
    
    # Run simulation
    print("Running road load simulation...")
    results = calculate_road_load(vehicle, cycle)
    
    # Generate summary
    summary = generate_summary_text(vehicle, results, cycle_name, cycle_filepath)
    
    # Print to console
    print(summary)
    
    # Save outputs
    print(f"Saving outputs to '{run_output_dir}/'...")
    
    # Save summary text
    txt_path = save_summary_text(summary, run_output_dir)
    print(f"  ✓ Summary: {os.path.basename(txt_path)}")
    
    # Generate and save plots
    plot_path = generate_plots(results, cycle_name, run_output_dir)
    plot_filename = os.path.basename(plot_path)
    print(f"  ✓ Plots: {plot_filename}")
    
    # Save time-series CSV
    csv_path = save_timeseries_csv(results, run_output_dir)
    print(f"  ✓ Time-series: {os.path.basename(csv_path)}")
    
    # Save HTML report
    html_path = save_summary_html(
        vehicle, results, cycle_name, cycle_filepath,
        run_output_dir, plot_filename
    )
    print(f"  ✓ HTML Report: {os.path.basename(html_path)}")
    
    print(f"\nSimulation complete!")
    
    return results


# =============================================================================
# Example Usage / Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Example vehicle parameters (Tesla Model 3-like)
    vehicle = VehicleParams(
        mass=1847,              # kg
        frontal_area=2.22,      # m²
        drag_coefficient=0.23,  # Cd
        rolling_resistance=0.01,  # Crr
        regen_efficiency=0.75   # 75% regen efficiency
    )
    
    # Path to drive cycle CSV
    # Expected format: time,speed,grade
    # Speed in m/s (or set speed_unit='mph' for mph)
    cycle_file = "drive_cycles/example_cycle.csv"
    
    # Check if example file exists
    if not os.path.exists(cycle_file):
        print(f"Example drive cycle not found: {cycle_file}")
        print("\nPlease create a CSV file with columns: time,speed,grade")
        print("  - time: Time in seconds")
        print("  - speed: Speed in m/s (or mph with speed_unit='mph')")
        print("  - grade: Road grade in percent (e.g., 6 for 6% grade)")
        print("\nExample:")
        print("  time,speed,grade")
        print("  0,0,0")
        print("  1,5.0,0")
        print("  2,10.0,0")
        print("  ...")
    else:
        # Run simulation
        results = run_simulation(
            vehicle=vehicle,
            cycle_filepath=cycle_file,
            cycle_name="Example",
            speed_unit='mps',
            output_dir='outputs'
        )


# =============================================================================
# Range Estimation
# =============================================================================
def estimate_range(
    vehicle: VehicleParams,
    cycle_filepath: str,
    usable_battery_pct: float = 0.9
) -> dict:
    """
    Estimate vehicle range by repeating drive cycle until battery is depleted.
    
    Args:
        vehicle: Vehicle parameters including battery capacity
        cycle_filepath: Path to drive cycle CSV file (speed must be in m/s)
        usable_battery_pct: Percentage of battery capacity that is usable (default 0.9 for 90%)
    
    Returns:
        Dictionary with range estimation results:
            - range_miles: Total range in miles
            - range_km: Total range in kilometers
            - energy_per_mile: kWh per mile
            - energy_per_km: kWh per kilometer
            - cycles_completed: Number of full cycles completed
            - final_soc: Final state of charge (%)
    """
    # Load the drive cycle (in m/s)
    cycle = load_drive_cycle(cycle_filepath)
    
    # Calculate usable battery energy in Joules
    usable_battery_j = vehicle.battery_capacity * usable_battery_pct * 3.6e6  # kWh to J
    
    # Run cycles until battery is depleted
    total_distance_m = 0.0
    total_energy_j = 0.0
    cycles_completed = 0
    
    while total_energy_j < usable_battery_j:
        # Run one cycle
        results = calculate_road_load(vehicle, cycle)
        
        # Get net energy for this cycle
        cycle_energy_j = results.total_energy
        
        # Check if we can complete this cycle
        if total_energy_j + cycle_energy_j <= usable_battery_j:
            # Complete full cycle
            total_energy_j += cycle_energy_j
            # Calculate distance for this cycle
            dt = np.diff(results.time, prepend=0)
            cycle_distance_m = np.sum(results.speed * dt)
            total_distance_m += cycle_distance_m
            cycles_completed += 1
        else:
            # Partial cycle - estimate remaining distance
            remaining_energy_j = usable_battery_j - total_energy_j
            fraction = remaining_energy_j / cycle_energy_j
            dt = np.diff(results.time, prepend=0)
            cycle_distance_m = np.sum(results.speed * dt)
            total_distance_m += cycle_distance_m * fraction
            total_energy_j = usable_battery_j
            break
    
    # Convert to useful units
    range_miles = total_distance_m / 1609.34
    range_km = total_distance_m / 1000.0
    energy_kwh = total_energy_j / 3.6e6
    energy_per_mile = energy_kwh / range_miles if range_miles > 0 else 0
    energy_per_km = energy_kwh / range_km if range_km > 0 else 0
    final_soc = (1 - usable_battery_pct) * 100  # Remaining unusable portion
    
    return {
        'range_miles': range_miles,
        'range_km': range_km,
        'energy_per_mile': energy_per_mile,
        'energy_per_km': energy_per_km,
        'energy_used_kwh': energy_kwh,
        'cycles_completed': cycles_completed,
        'final_soc': final_soc
    }


def estimate_range_multi_cycle(
    vehicle: VehicleParams,
    usable_battery_pct: float = 0.9
) -> dict:
    """
    Estimate vehicle range using EPA multi-cycle test.
    Consists of 4 city cycles (UDDS), 2 highway cycles (HWFET), and 2 constant speed cycles.
    
    Args:
        vehicle: Vehicle parameters including battery capacity
        speed_unit: 'mps' or 'mph'
        
    Returns:
        Dictionary with range estimation results:
            - range_miles: Total range in miles
            - range_km: Total range in kilometers
            - energy_per_mile: kWh per mile
            - energy_per_km: kWh per kilometer
            - total_cycles: Total number of multi-cycle sequences completed
            - final_soc: Final state of charge (%)
            - cycle_breakdown: Details for each cycle type
    """
    # Define the multi-cycle sequence
    cycle_sequence = [
        ('UDDS', 4),           # 4 city cycles
        ('HWFET', 2),          # 2 highway cycles
        ('constant_70mph', 2)  # 2 constant speed cycles
    ]
    
    # Load all cycle files (all in m/s format)
    cycles_data = {}
    for cycle_name, _ in cycle_sequence:
        filepath = os.path.join('drive_cycles', f'{cycle_name}.csv')
        cycles_data[cycle_name] = load_drive_cycle(filepath)
    
    # Calculate usable battery energy in Joules
    usable_battery_j = vehicle.battery_capacity * usable_battery_pct * 3.6e6  # kWh to J
    
    # Track totals
    total_distance_m = 0.0
    total_energy_j = 0.0
    total_multi_cycles = 0
    cycle_breakdown = {name: {'count': 0, 'distance_mi': 0, 'energy_kwh': 0} 
                      for name, _ in cycle_sequence}
    
    # Run multi-cycle sequences until battery is depleted
    while total_energy_j < usable_battery_j:
        sequence_complete = True
        
        # Run through the entire sequence
        for cycle_name, count in cycle_sequence:
            cycle = cycles_data[cycle_name]
            
            for _ in range(count):
                # Run one cycle
                results = calculate_road_load(vehicle, cycle)
                cycle_energy_j = results.total_energy
                
                # Check if we can complete this cycle
                if total_energy_j + cycle_energy_j <= usable_battery_j:
                    # Complete full cycle
                    total_energy_j += cycle_energy_j
                    dt = np.diff(results.time, prepend=0)
                    cycle_distance_m = np.sum(results.speed * dt)
                    total_distance_m += cycle_distance_m
                    
                    # Update breakdown
                    cycle_breakdown[cycle_name]['count'] += 1
                    cycle_breakdown[cycle_name]['distance_mi'] += cycle_distance_m / 1609.34
                    cycle_breakdown[cycle_name]['energy_kwh'] += cycle_energy_j / 3.6e6
                else:
                    # Partial cycle - estimate remaining distance
                    remaining_energy_j = usable_battery_j - total_energy_j
                    fraction = remaining_energy_j / cycle_energy_j
                    dt = np.diff(results.time, prepend=0)
                    cycle_distance_m = np.sum(results.speed * dt)
                    total_distance_m += cycle_distance_m * fraction
                    total_energy_j = usable_battery_j
                    
                    # Update breakdown for partial cycle
                    cycle_breakdown[cycle_name]['distance_mi'] += (cycle_distance_m / 1609.34) * fraction
                    cycle_breakdown[cycle_name]['energy_kwh'] += (cycle_energy_j / 3.6e6) * fraction
                    
                    sequence_complete = False
                    break
            
            if not sequence_complete:
                break
        
        if sequence_complete:
            total_multi_cycles += 1
        else:
            break
    
    # Convert to useful units
    range_miles = total_distance_m / 1609.34
    range_km = total_distance_m / 1000.0
    energy_kwh = total_energy_j / 3.6e6
    energy_per_mile = energy_kwh / range_miles if range_miles > 0 else 0
    energy_per_km = energy_kwh / range_km if range_km > 0 else 0
    final_soc = (1 - usable_battery_pct) * 100  # Remaining unusable portion
    
    return {
        'range_miles': range_miles,
        'range_km': range_km,
        'energy_per_mile': energy_per_mile,
        'energy_per_km': energy_per_km,
        'energy_used_kwh': energy_kwh,
        'total_cycles': total_multi_cycles,
        'final_soc': final_soc,
        'cycle_breakdown': cycle_breakdown
    }


# =============================================================================
# Gasoline Vehicle Simulation
# =============================================================================


def simulate_gasoline(
    vehicle: VehicleParams,
    cycle_filepath: str
) -> dict:
    """
    Simulate a gasoline vehicle over a drive cycle.
    Uses simple MPG-based fuel consumption calculation.
    
    Args:
        vehicle: Vehicle parameters with fuel_economy_mpg set
        cycle_filepath: Path to drive cycle CSV file (speed must be in m/s)
    
    Returns:
        Dictionary with simulation results:
            - distance_miles: Total distance traveled
            - fuel_used_gallons: Fuel consumed
            - mpg_achieved: Actual MPG for this cycle
            - range_miles: Estimated range from full tank
    """
    # Load and run the base simulation (cycle in m/s)
    cycle = load_drive_cycle(cycle_filepath)
    results = calculate_road_load(vehicle, cycle)
    
    # Calculate distance
    dt = np.diff(results.time, prepend=0)
    distance_m = np.sum(results.speed * dt)
    distance_miles = distance_m / 1609.34
    
    # Calculate fuel consumption based on MPG
    fuel_used_gallons = distance_miles / vehicle.fuel_economy_mpg
    
    # Calculate range from full tank
    range_miles = vehicle.fuel_tank_gallons * vehicle.fuel_economy_mpg
    
    return {
        'distance_miles': distance_miles,
        'fuel_used_gallons': fuel_used_gallons,
        'mpg_achieved': vehicle.fuel_economy_mpg,  # For simple model, same as input
        'range_miles': range_miles,
        'fuel_remaining_gallons': vehicle.fuel_tank_gallons - fuel_used_gallons
    }


# =============================================================================
# EREV (Extended Range Electric Vehicle) Simulation
# =============================================================================

@dataclass
class EREVResults:
    """Results from EREV simulation."""
    # Distance tracking
    total_distance_miles: float
    ev_only_miles: float
    generator_miles: float
    
    # Energy tracking
    battery_energy_used_kwh: float
    fuel_used_gallons: float
    
    # SOC tracking
    starting_soc: float
    final_soc: float
    min_soc: float
    max_soc: float
    soc_timeline: np.ndarray  # SOC at each timestep
    
    # Generator tracking
    generator_runtime_seconds: float
    generator_energy_kwh: float
    generator_output_kw: np.ndarray  # generator output over time
    generator_on_flags: np.ndarray  # bool/int flag for generator state
    fuel_remaining_gal: np.ndarray  # fuel remaining over time
    distance_miles_trace: np.ndarray  # cumulative distance for plotting
    time_trace: np.ndarray  # time trace matching SOC timeline
    
    # Power deficit tracking
    power_deficit_count: int
    max_power_deficit_kw: float
    power_deficit_events: List[dict]  # List of {time, demand_kw, available_kw, deficit_kw}
    
    # Efficiency metrics
    mpge: float  # Miles per gallon equivalent
    kwh_per_mile: float


def simulate_erev(
    vehicle: VehicleParams,
    cycle_filepath: str,
    speed_unit: str = 'mps',
    starting_soc: float = 100.0,
    precomputed_results: Optional[SimulationResults] = None
) -> EREVResults:
    """
    Simulate an Extended Range Electric Vehicle over a drive cycle.
    
    Implements three operating modes:
        - charge_depleting: Pure EV until SOC hits sustain threshold, then generator maintains
        - blended: Generator assists when power demand exceeds threshold or SOC below blended threshold
        - hold: Generator maintains starting SOC, battery only covers peaks
    
    Args:
        vehicle: Vehicle parameters with EREV fields set
        cycle_filepath: Path to drive cycle CSV file
        speed_unit: 'mps' or 'mph'
        starting_soc: Initial state of charge (%)
    
    Returns:
        EREVResults with comprehensive simulation data
    """
    # Load and run the base simulation to get power demand at each timestep
    if precomputed_results is not None:
        results = precomputed_results
    else:
        cycle = load_drive_cycle(cycle_filepath, speed_unit)
        results = calculate_road_load(vehicle, cycle)
    
    # Extract time-series data
    time = results.time
    power_demand_w = results.power  # Positive = traction, negative = regen
    speed = results.speed
    n = len(time)
    
    # Calculate timestep durations
    dt = np.diff(time, prepend=0)
    
    # Battery parameters
    usable_capacity_kwh = vehicle.battery_capacity * (vehicle.usable_battery_pct / 100.0)
    usable_capacity_j = usable_capacity_kwh * 3.6e6
    # Allow starting below the nominal minimum without clamping upward, but never above 100 or below 0
    starting_soc = min(max(starting_soc, 0.0), 100.0)
    min_soc_limit = max(0.0, 100.0 - vehicle.usable_battery_pct)  # nominal floor from usable window
    min_soc = min(starting_soc, min_soc_limit)
    
    # Generator parameters
    generator_power_w = vehicle.generator_power_kw * 1000.0
    
    # Fuel parameters (BSFC: grams per kWh)
    # Convert to gallons per Joule: g/kWh -> g/J -> gallons/J
    # gasoline density ~2834 g/gallon
    gasoline_density_g_per_gallon = 2834.0
    bsfc_g_per_j = vehicle.bsfc_g_kwh / 3.6e6  # g/kWh to g/J
    fuel_consumption_gal_per_j = bsfc_g_per_j / gasoline_density_g_per_gallon
    
    # Initialize state
    current_soc = starting_soc
    fuel_remaining_gallons = vehicle.fuel_tank_gallons

    # Generator sustain band to avoid on/off thrash
    sustain_on_pct = vehicle.soc_sustain_pct
    sustain_off_pct = min(100.0, vehicle.soc_sustain_pct + 2.0)
    generator_on = False
    
    # Tracking arrays and counters
    soc_timeline = np.zeros(n)
    generator_output_trace = np.zeros(n)
    generator_on_trace = np.zeros(n)
    fuel_remaining_trace = np.zeros(n)
    distance_miles_trace = np.zeros(n)
    time_trace = np.zeros(n)
    ev_distance_m = 0.0
    generator_distance_m = 0.0
    generator_runtime_s = 0.0
    generator_energy_j = 0.0
    battery_energy_used_j = 0.0
    
    power_deficit_events = []
    min_soc_reached = starting_soc
    max_soc_reached = starting_soc
    cumulative_distance_m = 0.0
    cumulative_time = 0.0
    
    # Simulation loop
    for i in range(n):
        soc_timeline[i] = current_soc
        min_soc_reached = min(min_soc_reached, current_soc)
        max_soc_reached = max(max_soc_reached, current_soc)
        fuel_remaining_trace[i] = fuel_remaining_gallons
        distance_miles_trace[i] = cumulative_distance_m / 1609.34
        time_trace[i] = cumulative_time
        
        demand_w = power_demand_w[i]
        step_dt = dt[i]
        distance_this_step = speed[i] * step_dt
        cumulative_distance_m += distance_this_step
        cumulative_time += step_dt
        
        # Determine if generator should run based on mode
        generator_active = False
        generator_output_w = 0.0
        
        if vehicle.erev_mode == 'charge_depleting':
            # Latch generator on once below sustain threshold; turn off after recovering a small band above
            if generator_on:
                if current_soc >= sustain_off_pct or fuel_remaining_gallons <= 0:
                    generator_on = False
            else:
                if current_soc <= sustain_on_pct and fuel_remaining_gallons > 0:
                    generator_on = True

            if generator_on:
                generator_active = True
                generator_output_w = generator_power_w
                
        elif vehicle.erev_mode == 'blended':
            # Generator assists when:
            # 1. SOC is below blended threshold, OR
            # 2. Power demand exceeds what battery can reasonably provide
            if fuel_remaining_gallons > 0:
                if current_soc <= vehicle.soc_blended_threshold_pct:
                    generator_active = True
                    generator_output_w = generator_power_w
                elif demand_w > 0 and demand_w > generator_power_w * 0.8:
                    # High power demand - blend generator with battery
                    generator_active = True
                    generator_output_w = min(demand_w * 0.5, generator_power_w)
                    
        elif vehicle.erev_mode == 'hold':
            # Generator runs whenever there's positive power demand
            if demand_w > 0 and fuel_remaining_gallons > 0:
                generator_active = True
                # Try to provide all needed power from generator
                generator_output_w = min(demand_w, generator_power_w)
        
        # Calculate power split
        if demand_w > 0:  # Traction
            # Generator provides what it can
            power_from_generator = generator_output_w if generator_active else 0.0
            # Battery provides the rest
            power_from_battery = demand_w - power_from_generator
            
            # Check for power deficit (generator + battery can't meet demand)
            available_power = power_from_generator + generator_power_w  # Max available
            if generator_active and demand_w > available_power:
                deficit_kw = (demand_w - available_power) / 1000.0
                power_deficit_events.append({
                    'time': time[i],
                    'demand_kw': demand_w / 1000.0,
                    'available_kw': available_power / 1000.0,
                    'deficit_kw': deficit_kw
                })
            
            # Update battery SOC
            battery_energy_j = power_from_battery * step_dt
            soc_change = (battery_energy_j / usable_capacity_j) * 100.0
            current_soc -= soc_change
            current_soc = max(current_soc, min_soc)
            battery_energy_used_j += battery_energy_j
            
            # Update generator fuel consumption
            if generator_active and power_from_generator > 0:
                generator_energy_this_step_j = power_from_generator * step_dt
                fuel_used_this_step = generator_energy_this_step_j * fuel_consumption_gal_per_j
                fuel_remaining_gallons -= fuel_used_this_step
                fuel_remaining_gallons = max(fuel_remaining_gallons, 0)
                generator_energy_j += generator_energy_this_step_j
                generator_runtime_s += step_dt
                generator_distance_m += distance_this_step
                generator_output_trace[i] = power_from_generator / 1000.0
                generator_on_trace[i] = 1
            else:
                ev_distance_m += distance_this_step
                generator_output_trace[i] = 0.0
                generator_on_trace[i] = 0
                
        else:  # Regenerative braking
            # Regen goes to battery (already accounted for in power value from calculate_road_load)
            regen_energy_j = abs(demand_w) * step_dt
            soc_change = (regen_energy_j / usable_capacity_j) * 100.0
            current_soc += soc_change
            current_soc = min(current_soc, 100.0)
            ev_distance_m += distance_this_step  # Still EV driving during regen
            generator_output_trace[i] = 0.0
            generator_on_trace[i] = 0
    
    # Calculate totals
    total_distance_miles = (ev_distance_m + generator_distance_m) / 1609.34
    ev_only_miles = ev_distance_m / 1609.34
    generator_miles = generator_distance_m / 1609.34
    
    battery_energy_used_kwh = battery_energy_used_j / 3.6e6
    fuel_used_gallons = vehicle.fuel_tank_gallons - fuel_remaining_gallons
    generator_energy_kwh = generator_energy_j / 3.6e6
    
    # Calculate efficiency metrics
    total_energy_kwh = battery_energy_used_kwh + (fuel_used_gallons * GASOLINE_ENERGY_KWH_PER_GALLON)
    kwh_per_mile = total_energy_kwh / total_distance_miles if total_distance_miles > 0 else 0
    
    # MPGe: miles per 33.7 kWh equivalent
    mpge = total_distance_miles / (total_energy_kwh / 33.7) if total_energy_kwh > 0 else 0
    
    # Power deficit summary
    power_deficit_count = len(power_deficit_events)
    max_power_deficit_kw = max([e['deficit_kw'] for e in power_deficit_events], default=0.0)
    
    return EREVResults(
        total_distance_miles=total_distance_miles,
        ev_only_miles=ev_only_miles,
        generator_miles=generator_miles,
        battery_energy_used_kwh=battery_energy_used_kwh,
        fuel_used_gallons=fuel_used_gallons,
        starting_soc=starting_soc,
        final_soc=current_soc,
        min_soc=min_soc_reached,
        max_soc=max_soc_reached,
        soc_timeline=soc_timeline,
        generator_runtime_seconds=generator_runtime_s,
        generator_energy_kwh=generator_energy_kwh,
        power_deficit_count=power_deficit_count,
        max_power_deficit_kw=max_power_deficit_kw,
        power_deficit_events=power_deficit_events,
        mpge=mpge,
        kwh_per_mile=kwh_per_mile,
        generator_output_kw=generator_output_trace,
        generator_on_flags=generator_on_trace,
        fuel_remaining_gal=fuel_remaining_trace,
        distance_miles_trace=distance_miles_trace,
        time_trace=time_trace
    )


def estimate_erev_range(
    vehicle: VehicleParams,
    cycle_filepath: str,
    starting_soc: float = 100.0
) -> dict:
    """
    Estimate total range for EREV by running cycles until both battery and fuel are depleted.
    
    Args:
        vehicle: Vehicle parameters with EREV fields set
        cycle_filepath: Path to drive cycle CSV file (speed must be in m/s)
        starting_soc: Initial state of charge (%)
    
    Returns:
        Dictionary with range estimation results
    """
    # Load the drive cycle (in m/s)
    cycle = load_drive_cycle(cycle_filepath)
    
    # Battery parameters
    usable_capacity_kwh = vehicle.battery_capacity * (vehicle.usable_battery_pct / 100.0)
    usable_capacity_j = usable_capacity_kwh * 3.6e6
    starting_soc = min(max(starting_soc, 0.0), 100.0)
    min_soc_limit = max(0.0, 100.0 - vehicle.usable_battery_pct)
    min_soc = min(starting_soc, min_soc_limit)
    
    # Generator/fuel parameters
    generator_power_w = vehicle.generator_power_kw * 1000.0
    gasoline_density_g_per_gallon = 2834.0
    bsfc_g_per_j = vehicle.bsfc_g_kwh / 3.6e6
    fuel_consumption_gal_per_j = bsfc_g_per_j / gasoline_density_g_per_gallon
    
    # Initialize state
    current_soc = starting_soc
    fuel_remaining_gallons = vehicle.fuel_tank_gallons

    # Generator sustain band to avoid on/off thrash
    sustain_on_pct = vehicle.soc_sustain_pct
    sustain_off_pct = min(100.0, vehicle.soc_sustain_pct + 2.0)
    generator_on = False
    
    # Tracking
    total_distance_m = 0.0
    ev_distance_m = 0.0
    generator_distance_m = 0.0
    total_fuel_used = 0.0
    total_battery_energy_j = 0.0
    cycles_completed = 0
    soc_trace = []
    generator_output_trace = []
    generator_on_trace = []
    fuel_remaining_trace = []
    distance_miles_trace = []
    time_trace = []
    cumulative_distance_m = 0.0
    cumulative_time = 0.0
    
    # Run until both battery and fuel are depleted
    max_cycles = 1000  # Safety limit
    while cycles_completed < max_cycles:
        # Check if we can continue
        if current_soc <= min_soc and fuel_remaining_gallons <= 0:
            break
        
        # Run one cycle
        results = calculate_road_load(vehicle, cycle)
        time = results.time
        power_demand_w = results.power
        speed = results.speed
        dt = np.diff(time, prepend=0)
        
        cycle_complete = True
        
        for i in range(len(time)):
            demand_w = power_demand_w[i]
            step_dt = dt[i]
            distance_this_step = speed[i] * step_dt
            cumulative_distance_m += distance_this_step
            cumulative_time += step_dt
            
            # Determine generator activity
            generator_active = False
            generator_output_w = 0.0
            
            if vehicle.erev_mode == 'charge_depleting':
                if generator_on:
                    if current_soc >= sustain_off_pct or fuel_remaining_gallons <= 0:
                        generator_on = False
                else:
                    if current_soc <= sustain_on_pct and fuel_remaining_gallons > 0:
                        generator_on = True

                if generator_on:
                    generator_active = True
                    generator_output_w = generator_power_w
            elif vehicle.erev_mode == 'blended':
                if fuel_remaining_gallons > 0 and current_soc <= vehicle.soc_blended_threshold_pct:
                    generator_active = True
                    generator_output_w = generator_power_w
            elif vehicle.erev_mode == 'hold':
                if demand_w > 0 and fuel_remaining_gallons > 0:
                    generator_active = True
                    generator_output_w = min(demand_w, generator_power_w)
            
            if demand_w > 0:  # Traction
                power_from_generator = generator_output_w if generator_active else 0.0
                power_from_battery = demand_w - power_from_generator
                
                # Check if we have enough power
                if power_from_battery > 0 and current_soc <= min_soc:
                    if not generator_active or fuel_remaining_gallons <= 0:
                        cycle_complete = False
                        break
                
                # Update battery
                if power_from_battery > 0:
                    battery_energy_j = power_from_battery * step_dt
                    soc_change = (battery_energy_j / usable_capacity_j) * 100.0
                    current_soc -= soc_change
                    current_soc = max(current_soc, min_soc)
                    total_battery_energy_j += battery_energy_j
                
                # Update fuel
                if generator_active and power_from_generator > 0:
                    generator_energy_j = power_from_generator * step_dt
                    fuel_used = generator_energy_j * fuel_consumption_gal_per_j
                    fuel_remaining_gallons -= fuel_used
                    fuel_remaining_gallons = max(fuel_remaining_gallons, 0)
                    total_fuel_used += fuel_used
                    generator_distance_m += distance_this_step
                    generator_output_trace.append(power_from_generator / 1000.0)
                    generator_on_trace.append(1)
                else:
                    ev_distance_m += distance_this_step
                    generator_output_trace.append(0.0)
                    generator_on_trace.append(0)
                    
                total_distance_m += distance_this_step
                
            else:  # Regen
                regen_energy_j = abs(demand_w) * step_dt
                soc_change = (regen_energy_j / usable_capacity_j) * 100.0
                current_soc += soc_change
                current_soc = min(current_soc, 100.0)
                ev_distance_m += distance_this_step
                total_distance_m += distance_this_step
                generator_output_trace.append(0.0)
                generator_on_trace.append(0)

            soc_trace.append(current_soc)
            fuel_remaining_trace.append(fuel_remaining_gallons)
            distance_miles_trace.append(cumulative_distance_m / 1609.34)
            time_trace.append(cumulative_time)
        
        if cycle_complete:
            cycles_completed += 1
        else:
            break
    
    # Calculate results
    total_distance_miles = total_distance_m / 1609.34
    ev_only_miles = ev_distance_m / 1609.34
    generator_miles = generator_distance_m / 1609.34
    battery_energy_kwh = total_battery_energy_j / 3.6e6
    
    # Efficiency
    total_energy_kwh = battery_energy_kwh + (total_fuel_used * GASOLINE_ENERGY_KWH_PER_GALLON)
    kwh_per_mile = total_energy_kwh / total_distance_miles if total_distance_miles > 0 else 0
    mpge = total_distance_miles / (total_energy_kwh / 33.7) if total_energy_kwh > 0 else 0
    
    return {
        'range_miles': total_distance_miles,
        'range_km': total_distance_miles * 1.60934,
        'ev_only_miles': ev_only_miles,
        'generator_miles': generator_miles,
        'battery_energy_kwh': battery_energy_kwh,
        'fuel_used_gallons': total_fuel_used,
        'final_soc': current_soc,
        'cycles_completed': cycles_completed,
        'mpge': mpge,
        'kwh_per_mile': kwh_per_mile,
        'soc_trace': np.array(soc_trace),
        'generator_output_kw': np.array(generator_output_trace),
        'generator_on_flags': np.array(generator_on_trace),
        'fuel_remaining_gal': np.array(fuel_remaining_trace),
        'distance_miles_trace': np.array(distance_miles_trace),
        'time_trace': np.array(time_trace)
    }

