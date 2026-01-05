#!/usr/bin/env python3
"""
Scout Terra Generator Sizing Study
===================================
Batch simulation of Scout Terra EREV configurations across multiple drive cycles.
Generates comparative HTML report with tables and charts.

Author: GitHub Copilot
Date: December 2024
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

from road_load_simulator import (
    VehicleParams,
    load_drive_cycle,
    calculate_road_load,
    simulate_erev,
    estimate_erev_range,
    estimate_erev_range_multi_cycle,
    GASOLINE_ENERGY_KWH_PER_GALLON,
    MPS_TO_MPH,
    AIR_DENSITY,
    GRAVITY
)
from scipy.optimize import brentq


# =============================================================================
# Configuration
# =============================================================================
STUDY_NAME = "scout_terra_generator_study_2025-12-30"
OUTPUT_BASE = "outputs"

# Drive cycles to test
DRIVE_CYCLES = {
    'davis_dam_charger': {
        'file': 'drive_cycles/davis_dam_charger.csv',
        'display_name': 'Davis Dam'
    },
    'el_rancho_to_frisco': {
        'file': 'drive_cycles/el_rancho_to_frisco.csv',
        'display_name': 'I-70 Climb'
    },
    'highway_75mph': {
        'file': 'drive_cycles/highway_75mph.csv',
        'display_name': 'Highway 75 MPH'
    },
    'US06': {
        'file': 'drive_cycles/US06.csv',
        'display_name': 'US06'
    }
}

# Starting SOC for simulations
STARTING_SOC_100 = 100.0
STARTING_SOC_50 = 50.0

# Generator capacity for reference
GENERATOR_CAPACITY_KW = 100.0


# =============================================================================
# Helper Functions
# =============================================================================
def load_scout_terra_presets(json_path: str = 'vehicle_presets.json') -> List[Dict[str, Any]]:
    """Load all Scout Terra EREV presets from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter for Scout Terra EREV variants
    scout_presets = [
        p for p in data['presets'] 
        if 'Scout Terra EREV' in p.get('name', '')
    ]
    
    return scout_presets


def preset_to_vehicle_params(preset: Dict[str, Any]) -> VehicleParams:
    """Convert a JSON preset to VehicleParams object."""
    return VehicleParams(
        mass=preset['mass'],
        frontal_area=preset['frontal_area'],
        drag_coefficient=preset['cd'],
        rolling_resistance=preset['crr'],
        vehicle_class=preset['vehicle_class'],
        drivetrain_efficiency=preset.get('drivetrain_efficiency', 0.9),
        regen_efficiency=preset['regen'],
        auxiliary_power=preset['aux_power'],
        battery_capacity=preset['battery'],
        usable_battery_pct=preset['usable_battery_pct'],
        generator_power_kw=preset['generator_power_kw'],
        fuel_tank_gallons=preset['fuel_tank_gallons'],
        bsfc_g_kwh=preset['bsfc_g_kwh'],
        erev_mode=preset['erev_mode'],
        soc_sustain_pct=preset['soc_sustain_pct'],
        soc_blended_threshold_pct=preset['soc_blended_threshold_pct']
    )


def calculate_efficiency_metrics(
    distance_miles: float,
    battery_energy_kwh: float,
    fuel_gallons: float
) -> Dict[str, float]:
    """Calculate efficiency metrics: battery mi/kWh, overall mi/kWh, MPG, MPGe."""
    metrics = {
        'mi_per_kwh': 0.0,
        'overall_mi_per_kwh': 0.0,
        'mpg': 0.0,
        'mpge': 0.0
    }
    
    # mi/kWh (battery only - can be misleadingly high when generator runs more)
    if battery_energy_kwh > 0:
        metrics['mi_per_kwh'] = distance_miles / battery_energy_kwh
    
    # Overall mi/kWh (total energy efficiency - battery + fuel combined)
    fuel_energy_kwh = fuel_gallons * GASOLINE_ENERGY_KWH_PER_GALLON
    total_energy_kwh = battery_energy_kwh + fuel_energy_kwh
    if total_energy_kwh > 0:
        metrics['overall_mi_per_kwh'] = distance_miles / total_energy_kwh
    
    # MPG (fuel only, for generator-assisted portion)
    if fuel_gallons > 0:
        metrics['mpg'] = distance_miles / fuel_gallons
    
    # MPGe (combined efficiency per EPA formula)
    if total_energy_kwh > 0:
        # EPA MPGe = (distance / total_energy_kwh) * 33.7
        metrics['mpge'] = (distance_miles / total_energy_kwh) * GASOLINE_ENERGY_KWH_PER_GALLON
    
    return metrics


@dataclass
class CycleResult:
    """Results from a single cycle simulation."""
    vehicle_name: str
    cycle_name: str
    cycle_display_name: str
    starting_soc: float
    distance_miles: float
    battery_energy_kwh: float
    fuel_gallons: float
    generator_runtime_min: float
    generator_energy_kwh: float
    final_soc: float
    min_soc: float
    peak_power_kw: float
    avg_power_kw: float
    power_deficit_count: int
    max_power_deficit_kw: float
    mi_per_kwh: float
    overall_mi_per_kwh: float
    mpg: float
    mpge: float
    soc_timeline: np.ndarray
    generator_output_kw: np.ndarray
    distance_trace: np.ndarray


@dataclass
class RangeResult:
    """Results from multi-cycle range test."""
    vehicle_name: str
    test_type: str  # 'multi_cycle' or 'highway_75mph'
    starting_soc: float
    total_range_miles: float
    ev_range_miles: float
    generator_range_miles: float
    fuel_used_gallons: float
    battery_energy_kwh: float
    mpge: float
    peak_power_kw: float = 0.0  # For 75mph test
    exceeds_generator: bool = False  # True if peak power > 100kW


@dataclass
class GeneratorSpeedResult:
    """Results from max generator-only speed calculation."""
    vehicle_name: str
    max_speed_mph: float
    power_at_max_speed_kw: float
    power_at_75mph_kw: float
    can_sustain_75mph: bool


# =============================================================================
# Simulation Runner
# =============================================================================
def run_single_cycle_simulation(
    vehicle: VehicleParams,
    vehicle_name: str,
    cycle_key: str,
    cycle_info: Dict[str, str],
    starting_soc: float = 100.0
) -> CycleResult:
    """Run simulation for a single vehicle/cycle combination."""
    cycle_file = cycle_info['file']
    display_name = cycle_info['display_name']
    
    # Load cycle and calculate road load for power stats
    cycle = load_drive_cycle(cycle_file)
    road_load = calculate_road_load(vehicle, cycle)
    
    # Peak and average power (at wheel, positive only for traction)
    power_kw = road_load.power / 1000.0
    peak_power_kw = float(np.max(power_kw))
    avg_power_kw = float(np.mean(power_kw[power_kw > 0])) if np.any(power_kw > 0) else 0.0
    
    # Run EREV simulation
    erev_result = simulate_erev(
        vehicle=vehicle,
        cycle_filepath=cycle_file,
        starting_soc=starting_soc,
        precomputed_results=road_load
    )
    
    # Calculate efficiency metrics
    metrics = calculate_efficiency_metrics(
        erev_result.total_distance_miles,
        erev_result.battery_energy_used_kwh,
        erev_result.fuel_used_gallons
    )
    
    return CycleResult(
        vehicle_name=vehicle_name,
        cycle_name=cycle_key,
        cycle_display_name=display_name,
        starting_soc=starting_soc,
        distance_miles=erev_result.total_distance_miles,
        battery_energy_kwh=erev_result.battery_energy_used_kwh,
        fuel_gallons=erev_result.fuel_used_gallons,
        generator_runtime_min=erev_result.generator_runtime_seconds / 60.0,
        generator_energy_kwh=erev_result.generator_energy_kwh,
        final_soc=erev_result.final_soc,
        min_soc=erev_result.min_soc,
        peak_power_kw=peak_power_kw,
        avg_power_kw=avg_power_kw,
        power_deficit_count=erev_result.power_deficit_count,
        max_power_deficit_kw=erev_result.max_power_deficit_kw,
        mi_per_kwh=metrics['mi_per_kwh'],
        overall_mi_per_kwh=metrics['overall_mi_per_kwh'],
        mpg=metrics['mpg'],
        mpge=metrics['mpge'],
        soc_timeline=erev_result.soc_timeline,
        generator_output_kw=erev_result.generator_output_kw,
        distance_trace=erev_result.distance_miles_trace
    )


def run_multi_cycle_range_test(
    vehicle: VehicleParams,
    vehicle_name: str,
    starting_soc: float = 100.0
) -> RangeResult:
    """Run multi-cycle range test for a vehicle."""
    result = estimate_erev_range_multi_cycle(vehicle, starting_soc=starting_soc)
    
    # Extract values from result dict (note: keys differ from single-cycle results)
    total_range = result.get('range_miles', 0.0)
    ev_range = result.get('ev_only_miles', 0.0)
    generator_range = result.get('generator_miles', 0.0)
    fuel_used = result.get('fuel_used_gallons', 0.0)
    battery_energy = result.get('battery_energy_kwh', 0.0)
    mpge = result.get('mpge', 0.0)
    
    return RangeResult(
        vehicle_name=vehicle_name,
        test_type='multi_cycle',
        starting_soc=starting_soc,
        total_range_miles=total_range,
        ev_range_miles=ev_range,
        generator_range_miles=generator_range,
        fuel_used_gallons=fuel_used,
        battery_energy_kwh=battery_energy,
        mpge=mpge
    )


def run_highway_75mph_range_test(
    vehicle: VehicleParams,
    vehicle_name: str,
    starting_soc: float = 100.0
) -> RangeResult:
    """Run range test using 75mph steady-state cycle."""
    cycle_file = 'drive_cycles/highway_75mph.csv'
    
    # Calculate road load to get peak power
    cycle = load_drive_cycle(cycle_file)
    road_load = calculate_road_load(vehicle, cycle)
    power_kw = road_load.power / 1000.0
    peak_power_kw = float(np.max(power_kw))
    
    # Run range estimation
    result = estimate_erev_range(vehicle, cycle_file, starting_soc=starting_soc)
    
    total_range = result.get('range_miles', 0.0)
    ev_range = result.get('ev_only_miles', 0.0)
    generator_range = result.get('generator_miles', 0.0)
    fuel_used = result.get('fuel_used_gallons', 0.0)
    battery_energy = result.get('battery_energy_kwh', 0.0)
    mpge = result.get('mpge', 0.0)
    
    return RangeResult(
        vehicle_name=vehicle_name,
        test_type='highway_75mph',
        starting_soc=starting_soc,
        total_range_miles=total_range,
        ev_range_miles=ev_range,
        generator_range_miles=generator_range,
        fuel_used_gallons=fuel_used,
        battery_energy_kwh=battery_energy,
        mpge=mpge,
        peak_power_kw=peak_power_kw,
        exceeds_generator=peak_power_kw > GENERATOR_CAPACITY_KW
    )


def calculate_max_generator_speed(
    vehicle: VehicleParams,
    vehicle_name: str,
    generator_power_kw: float = 100.0
) -> GeneratorSpeedResult:
    """
    Calculate maximum steady-state speed achievable on generator power alone.
    
    Solves for speed where: P_generator = P_aero + P_rolling
    Where:
        P_aero = 0.5 * rho * Cd * A * v^3
        P_rolling = Crr * m * g * v
    """
    generator_power_w = generator_power_kw * 1000.0
    
    def road_load_power(v_mps):
        """Calculate power required at given speed (flat road, no acceleration)."""
        # Aerodynamic power: P = 0.5 * rho * Cd * A * v^3
        p_aero = 0.5 * AIR_DENSITY * vehicle.drag_coefficient * vehicle.frontal_area * (v_mps ** 3)
        # Rolling resistance power: P = Crr * m * g * v
        p_rolling = vehicle.rolling_resistance * vehicle.mass * GRAVITY * v_mps
        # Auxiliary power
        p_aux = vehicle.auxiliary_power
        return p_aero + p_rolling + p_aux
    
    def power_balance(v_mps):
        """Return difference between generator power and required power."""
        return generator_power_w - road_load_power(v_mps)
    
    # Find max speed where generator can sustain (search from 1 m/s to 60 m/s = 134 mph)
    try:
        # Check if generator can even sustain low speed
        if power_balance(1.0) < 0:
            max_speed_mps = 0.0
        else:
            # Find the speed where power balance = 0
            max_speed_mps = brentq(power_balance, 1.0, 60.0)
    except ValueError:
        # If brentq fails (no root in range), generator can sustain max speed
        max_speed_mps = 60.0 if power_balance(60.0) > 0 else 0.0
    
    max_speed_mph = max_speed_mps * MPS_TO_MPH
    power_at_max = road_load_power(max_speed_mps) / 1000.0
    
    # Calculate power at 75 mph
    speed_75mph_mps = 75.0 / MPS_TO_MPH
    power_at_75mph = road_load_power(speed_75mph_mps) / 1000.0
    
    return GeneratorSpeedResult(
        vehicle_name=vehicle_name,
        max_speed_mph=max_speed_mph,
        power_at_max_speed_kw=power_at_max,
        power_at_75mph_kw=power_at_75mph,
        can_sustain_75mph=(power_at_75mph <= generator_power_kw)
    )


# =============================================================================
# Chart Generation
# =============================================================================
def generate_efficiency_bar_chart(
    results: List[CycleResult],
    output_dir: str,
    metric: str,
    title: str,
    ylabel: str
) -> str:
    """Generate grouped bar chart for efficiency comparison."""
    # Group by cycle
    cycles = list(DRIVE_CYCLES.keys())
    vehicles = list(set(r.vehicle_name for r in results))
    
    # Sort vehicles - base first, then trailers
    vehicles.sort(key=lambda x: (0 if 'Harvester' in x else 1, x))
    
    # Create short labels for vehicles
    short_labels = []
    for v in vehicles:
        if 'Harvester' in v:
            short_labels.append('Base')
        else:
            short_labels.append(v.replace('Scout Terra EREV + ', '').replace(' Trailer', ''))
    
    x = np.arange(len(cycles))
    width = 0.12
    n_vehicles = len(vehicles)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        values = []
        for cycle_key in cycles:
            matching = [r for r in results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                values.append(getattr(matching[0], metric))
            else:
                values.append(0)
        
        offset = (i - n_vehicles/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label)
    
    ax.set_xlabel('Drive Cycle', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{metric}_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_power_demand_chart(
    results: List[CycleResult],
    output_dir: str
) -> str:
    """Generate peak vs average power chart."""
    cycles = list(DRIVE_CYCLES.keys())
    vehicles = list(set(r.vehicle_name for r in results))
    vehicles.sort(key=lambda x: (0 if 'Harvester' in x else 1, x))
    
    short_labels = []
    for v in vehicles:
        if 'Harvester' in v:
            short_labels.append('Base')
        else:
            short_labels.append(v.replace('Scout Terra EREV + ', '').replace(' Trailer', ''))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(cycles))
    width = 0.12
    n_vehicles = len(vehicles)
    
    # Peak power chart
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        values = []
        for cycle_key in cycles:
            matching = [r for r in results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                values.append(matching[0].peak_power_kw)
            else:
                values.append(0)
        
        offset = (i - n_vehicles/2 + 0.5) * width
        axes[0].bar(x + offset, values, width, label=label)
    
    # Add generator capacity line
    axes[0].axhline(y=100, color='red', linestyle='--', linewidth=2, label='Generator (100 kW)')
    
    axes[0].set_xlabel('Drive Cycle', fontsize=12)
    axes[0].set_ylabel('Peak Power (kW)', fontsize=12)
    axes[0].set_title('Peak Power Demand vs Generator Capacity', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Average power chart
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        values = []
        for cycle_key in cycles:
            matching = [r for r in results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                values.append(matching[0].avg_power_kw)
            else:
                values.append(0)
        
        offset = (i - n_vehicles/2 + 0.5) * width
        axes[1].bar(x + offset, values, width, label=label)
    
    axes[1].axhline(y=100, color='red', linestyle='--', linewidth=2, label='Generator (100 kW)')
    
    axes[1].set_xlabel('Drive Cycle', fontsize=12)
    axes[1].set_ylabel('Average Power (kW)', fontsize=12)
    axes[1].set_title('Average Power Demand vs Generator Capacity', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'power_demand_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'power_demand_comparison.png'


def generate_soc_profile_charts(
    results: List[CycleResult],
    output_dir: str,
    starting_soc: int = 100,
    filename_suffix: str = ''
) -> str:
    """Generate SOC profile charts for each cycle."""
    cycles = list(DRIVE_CYCLES.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, cycle_key in enumerate(cycles):
        ax = axes[idx]
        cycle_results = [r for r in results if r.cycle_name == cycle_key]
        
        for r in cycle_results:
            if 'Harvester' in r.vehicle_name:
                label = 'Base'
                linestyle = '-'
                linewidth = 2.5
            else:
                label = r.vehicle_name.replace('Scout Terra EREV + ', '').replace(' Trailer', '')
                linestyle = '--'
                linewidth = 1.5
            
            ax.plot(r.distance_trace, r.soc_timeline, label=label, linestyle=linestyle, linewidth=linewidth)
        
        ax.set_xlabel('Distance (miles)', fontsize=11)
        ax.set_ylabel('State of Charge (%)', fontsize=11)
        ax.set_title(f'{DRIVE_CYCLES[cycle_key]["display_name"]} - SOC Profile ({starting_soc}% Start)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    filename = f'soc_profiles{filename_suffix}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_generator_utilization_chart(
    results: List[CycleResult],
    output_dir: str
) -> str:
    """Generate generator runtime and energy charts."""
    cycles = list(DRIVE_CYCLES.keys())
    vehicles = list(set(r.vehicle_name for r in results))
    vehicles.sort(key=lambda x: (0 if 'Harvester' in x else 1, x))
    
    short_labels = []
    for v in vehicles:
        if 'Harvester' in v:
            short_labels.append('Base')
        else:
            short_labels.append(v.replace('Scout Terra EREV + ', '').replace(' Trailer', ''))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(cycles))
    width = 0.12
    n_vehicles = len(vehicles)
    
    # Generator runtime chart
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        values = []
        for cycle_key in cycles:
            matching = [r for r in results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                values.append(matching[0].generator_runtime_min)
            else:
                values.append(0)
        
        offset = (i - n_vehicles/2 + 0.5) * width
        axes[0].bar(x + offset, values, width, label=label)
    
    axes[0].set_xlabel('Drive Cycle', fontsize=12)
    axes[0].set_ylabel('Generator Runtime (min)', fontsize=12)
    axes[0].set_title('Generator Runtime by Configuration', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Generator energy chart
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        values = []
        for cycle_key in cycles:
            matching = [r for r in results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                values.append(matching[0].generator_energy_kwh)
            else:
                values.append(0)
        
        offset = (i - n_vehicles/2 + 0.5) * width
        axes[1].bar(x + offset, values, width, label=label)
    
    axes[1].set_xlabel('Drive Cycle', fontsize=12)
    axes[1].set_ylabel('Generator Energy (kWh)', fontsize=12)
    axes[1].set_title('Generator Energy Output by Configuration', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'generator_utilization.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'generator_utilization.png'


def generate_range_comparison_chart(
    range_results: List[RangeResult],
    output_dir: str
) -> str:
    """Generate multi-cycle range comparison chart with stacked bars."""
    range_results_sorted = sorted(range_results, key=lambda x: (0 if 'Harvester' in x.vehicle_name else 1, x.vehicle_name))
    
    short_labels = []
    for r in range_results_sorted:
        if 'Harvester' in r.vehicle_name:
            short_labels.append('Base')
        else:
            short_labels.append(r.vehicle_name.replace('Scout Terra EREV + ', '').replace(' Trailer', ''))
    
    x = np.arange(len(range_results_sorted))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ev_ranges = [r.ev_range_miles for r in range_results_sorted]
    gen_ranges = [r.generator_range_miles for r in range_results_sorted]
    
    # Stacked bar chart - generator range on top of EV range
    bars1 = ax.bar(x, ev_ranges, width, label='EV Range (Battery Only)', color='#2ecc71')
    bars2 = ax.bar(x, gen_ranges, width, bottom=ev_ranges, label='Generator-Assisted Range', color='#e74c3c')
    
    # Add total range labels on top
    for i, r in enumerate(range_results_sorted):
        ax.annotate(f'{r.total_range_miles:.0f} mi',
                   xy=(i, r.total_range_miles + 8),
                   ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Vehicle Configuration', fontsize=12)
    ax.set_ylabel('Range (miles)', fontsize=12)
    ax.set_title('Multi-Cycle Range Test Results (Stacked: EV + Generator)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'range_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'range_comparison.png'


# =============================================================================
# HTML Report Generation
# =============================================================================
def generate_html_report(
    presets: List[Dict[str, Any]],
    cycle_results: List[CycleResult],
    range_results: List[RangeResult],
    output_dir: str,
    chart_files: Dict[str, str]
) -> str:
    """Generate comprehensive HTML report."""
    
    # Sort vehicles
    vehicles = list(set(r.vehicle_name for r in cycle_results))
    vehicles.sort(key=lambda x: (0 if 'Harvester' in x else 1, x))
    
    cycles = list(DRIVE_CYCLES.keys())
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scout Terra Generator Sizing Study</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            color: white;
            margin-top: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 10px;
            text-align: center;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tr:hover {{
            background: #e8f4f8;
        }}
        .deficit {{
            background-color: #ffcccc !important;
            color: #c0392b;
            font-weight: bold;
        }}
        .ok {{
            background-color: #ccffcc !important;
            color: #27ae60;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .vehicle-name {{
            text-align: left;
            font-weight: 500;
        }}
        .cycle-header {{
            background: #2c3e50 !important;
        }}
        .metric-header {{
            background: #16a085 !important;
            font-size: 0.9em;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 2px 6px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>🔋 Scout Terra EREV Generator Sizing Study</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary-box">
        <h3>Study Overview</h3>
        <p>This study evaluates the Scout Terra EREV's <strong>100 kW generator</strong> across various towing configurations and demanding drive cycles.</p>
        <ul>
            <li><strong>Vehicles Tested:</strong> {len(vehicles)} configurations (base + 6 trailer combinations)</li>
            <li><strong>Drive Cycles:</strong> {len(cycles)} cycles (Davis Dam, I-70 Climb, Highway 75 MPH, US06)</li>
            <li><strong>Starting SOC:</strong> {STARTING_SOC}%</li>
            <li><strong>EREV Mode:</strong> Blended (generator assists at high power demand)</li>
        </ul>
    </div>
    
    <h2>📊 Table 1: Vehicle Configurations</h2>
    <div class="section">
        <table>
            <tr>
                <th>Configuration</th>
                <th>Mass (kg)</th>
                <th>Mass (lb)</th>
                <th>Frontal Area (m²)</th>
                <th>Drag Coefficient (Cd)</th>
                <th>Rolling Resistance (Crr)</th>
                <th>Generator (kW)</th>
            </tr>
"""
    
    # Add vehicle configuration rows
    for preset in presets:
        name = preset['name']
        if 'Harvester' in name:
            display_name = 'Base (No Trailer)'
        else:
            display_name = name.replace('Scout Terra EREV + ', '')
        
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td>{preset['mass']:.0f}</td>
                <td>{preset['mass'] * 2.205:.0f}</td>
                <td>{preset['frontal_area']:.1f}</td>
                <td>{preset['cd']:.2f}</td>
                <td>{preset['crr']:.3f}</td>
                <td>{preset['generator_power_kw']:.0f}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
    
    <h2>⚡ Table 2: Power Demand Summary</h2>
    <div class="section">
        <p>Peak and average power demand at the wheels for each vehicle/cycle combination. Values exceeding the 100 kW generator capacity are highlighted.</p>
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    # Add cycle headers for power table
    for cycle_key in cycles:
        display = DRIVE_CYCLES[cycle_key]['display_name']
        html += f'                <th colspan="2" class="cycle-header">{display}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    
    for _ in cycles:
        html += '                <th class="metric-header">Peak (kW)</th>\n'
        html += '                <th class="metric-header">Avg (kW)</th>\n'
    
    html += """            </tr>
"""
    
    # Add power data rows
    for vehicle in vehicles:
        if 'Harvester' in vehicle:
            display_name = 'Base (No Trailer)'
        else:
            display_name = vehicle.replace('Scout Terra EREV + ', '')
        
        html += f'            <tr>\n                <td class="vehicle-name">{display_name}</td>\n'
        
        for cycle_key in cycles:
            matching = [r for r in cycle_results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                r = matching[0]
                peak_class = 'deficit' if r.peak_power_kw > 100 else ''
                avg_class = 'deficit' if r.avg_power_kw > 100 else ''
                html += f'                <td class="{peak_class}">{r.peak_power_kw:.1f}</td>\n'
                html += f'                <td class="{avg_class}">{r.avg_power_kw:.1f}</td>\n'
            else:
                html += '                <td>-</td>\n                <td>-</td>\n'
        
        html += '            </tr>\n'
    
    html += """        </table>
    </div>
    
    <h2>🔥 Table 3: Efficiency Results</h2>
    <div class="section">
        <p><strong>Overall mi/kWh</strong> shows total energy efficiency (battery + fuel combined). This is the most accurate measure of drive cycle energy intensity. Cells with power deficits (generator undersized) are highlighted in <span class="deficit">red</span>.</p>
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    # Add cycle headers for efficiency table
    for cycle_key in cycles:
        display = DRIVE_CYCLES[cycle_key]['display_name']
        html += f'                <th colspan="3" class="cycle-header">{display}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    
    for _ in cycles:
        html += '                <th class="metric-header">Overall mi/kWh</th>\n'
        html += '                <th class="metric-header">MPG</th>\n'
        html += '                <th class="metric-header">MPGe</th>\n'
    
    html += """            </tr>
"""
    
    # Add efficiency data rows
    for vehicle in vehicles:
        if 'Harvester' in vehicle:
            display_name = 'Base (No Trailer)'
        else:
            display_name = vehicle.replace('Scout Terra EREV + ', '')
        
        html += f'            <tr>\n                <td class="vehicle-name">{display_name}</td>\n'
        
        for cycle_key in cycles:
            matching = [r for r in cycle_results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                r = matching[0]
                cell_class = 'deficit' if r.power_deficit_count > 0 else ''
                mpg_display = f'{r.mpg:.1f}' if r.fuel_gallons > 0.01 else 'N/A'
                html += f'                <td class="{cell_class}">{r.overall_mi_per_kwh:.2f}</td>\n'
                html += f'                <td class="{cell_class}">{mpg_display}</td>\n'
                html += f'                <td class="{cell_class}">{r.mpge:.1f}</td>\n'
            else:
                html += '                <td>-</td>\n                <td>-</td>\n                <td>-</td>\n'
        
        html += '            </tr>\n'
    
    html += """        </table>
    </div>
    
    <h2>🔌 Table 4: Generator Utilization</h2>
    <div class="section">
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    # Add cycle headers
    for cycle_key in cycles:
        display = DRIVE_CYCLES[cycle_key]['display_name']
        html += f'                <th colspan="4" class="cycle-header">{display}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    
    for _ in cycles:
        html += '                <th class="metric-header">Runtime (min)</th>\n'
        html += '                <th class="metric-header">Energy (kWh)</th>\n'
        html += '                <th class="metric-header">Fuel (gal)</th>\n'
        html += '                <th class="metric-header">Deficits</th>\n'
    
    html += """            </tr>
"""
    
    # Add generator utilization rows
    for vehicle in vehicles:
        if 'Harvester' in vehicle:
            display_name = 'Base (No Trailer)'
        else:
            display_name = vehicle.replace('Scout Terra EREV + ', '')
        
        html += f'            <tr>\n                <td class="vehicle-name">{display_name}</td>\n'
        
        for cycle_key in cycles:
            matching = [r for r in cycle_results if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                r = matching[0]
                deficit_class = 'deficit' if r.power_deficit_count > 0 else 'ok'
                deficit_text = f'{r.power_deficit_count} ({r.max_power_deficit_kw:.1f} kW)' if r.power_deficit_count > 0 else '0'
                html += f'                <td>{r.generator_runtime_min:.1f}</td>\n'
                html += f'                <td>{r.generator_energy_kwh:.1f}</td>\n'
                html += f'                <td>{r.fuel_gallons:.2f}</td>\n'
                html += f'                <td class="{deficit_class}">{deficit_text}</td>\n'
            else:
                html += '                <td>-</td>\n                <td>-</td>\n                <td>-</td>\n                <td>-</td>\n'
        
        html += '            </tr>\n'
    
    html += """        </table>
    </div>
    
    <h2>🛣️ Table 5: Multi-Cycle Range Test Results</h2>
    <div class="section">
        <p>EPA-style multi-cycle range test (4×UDDS + 2×HWFET + 2×Constant 70mph). Starting SOC: 100%, Full fuel tank.</p>
        <table>
            <tr>
                <th>Configuration</th>
                <th>Total Range (mi)</th>
                <th>EV Range (mi)</th>
                <th>Generator Range (mi)</th>
                <th>Fuel Used (gal)</th>
                <th>Battery Energy (kWh)</th>
                <th>MPGe</th>
            </tr>
"""
    
    # Sort range results
    range_results_sorted = sorted(range_results, key=lambda x: (0 if 'Harvester' in x.vehicle_name else 1, x.vehicle_name))
    
    for r in range_results_sorted:
        if 'Harvester' in r.vehicle_name:
            display_name = 'Base (No Trailer)'
        else:
            display_name = r.vehicle_name.replace('Scout Terra EREV + ', '')
        
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td><strong>{r.total_range_miles:.0f}</strong></td>
                <td>{r.ev_range_miles:.0f}</td>
                <td>{r.generator_range_miles:.0f}</td>
                <td>{r.fuel_used_gallons:.1f}</td>
                <td>{r.battery_energy_kwh:.1f}</td>
                <td>{r.mpge:.1f}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
    
    <h2>📈 Charts</h2>
    
    <div class="chart-container">
        <h3>Power Demand vs Generator Capacity</h3>
        <img src="{power_chart}" alt="Power Demand Comparison">
    </div>
    
    <div class="chart-container">
        <h3>MPGe Efficiency Comparison</h3>
        <img src="{mpge_chart}" alt="MPGe Comparison">
    </div>
    
    <div class="chart-container">
        <h3>Generator Utilization</h3>
        <img src="{gen_chart}" alt="Generator Utilization">
    </div>
    
    <div class="chart-container">
        <h3>SOC Profiles by Drive Cycle</h3>
        <img src="{soc_chart}" alt="SOC Profiles">
    </div>
    
    <div class="chart-container">
        <h3>Multi-Cycle Range Comparison</h3>
        <img src="{range_chart}" alt="Range Comparison">
    </div>
    
    <h2>📝 Key Findings</h2>
    <div class="section">
        <h3>Generator Sufficiency Analysis</h3>
        <ul>
""".format(
        power_chart=chart_files.get('power', 'power_demand_comparison.png'),
        mpge_chart=chart_files.get('mpge', 'mpge_comparison.png'),
        gen_chart=chart_files.get('generator', 'generator_utilization.png'),
        soc_chart=chart_files.get('soc', 'soc_profiles.png'),
        range_chart=chart_files.get('range', 'range_comparison.png')
    )
    
    # Analyze deficits
    deficit_configs = []
    for r in cycle_results:
        if r.power_deficit_count > 0:
            deficit_configs.append({
                'vehicle': r.vehicle_name,
                'cycle': r.cycle_display_name,
                'count': r.power_deficit_count,
                'max_deficit': r.max_power_deficit_kw
            })
    
    if deficit_configs:
        html += "            <li><strong>Power Deficits Detected:</strong> The 100 kW generator was insufficient in the following scenarios:</li>\n            <ul>\n"
        for d in deficit_configs:
            name = 'Base' if 'Harvester' in d['vehicle'] else d['vehicle'].replace('Scout Terra EREV + ', '')
            html += f"                <li>{name} on {d['cycle']}: {d['count']} deficit events, max shortfall of {d['max_deficit']:.1f} kW</li>\n"
        html += "            </ul>\n"
    else:
        html += "            <li><strong>No Power Deficits:</strong> The 100 kW generator was sufficient for all tested configurations and cycles.</li>\n"
    
    # Find highest power demands
    max_peak = max(cycle_results, key=lambda x: x.peak_power_kw)
    max_avg = max(cycle_results, key=lambda x: x.avg_power_kw)
    
    peak_name = 'Base' if 'Harvester' in max_peak.vehicle_name else max_peak.vehicle_name.replace('Scout Terra EREV + ', '')
    avg_name = 'Base' if 'Harvester' in max_avg.vehicle_name else max_avg.vehicle_name.replace('Scout Terra EREV + ', '')
    
    html += f"""            <li><strong>Highest Peak Power Demand:</strong> {peak_name} on {max_peak.cycle_display_name} at {max_peak.peak_power_kw:.1f} kW</li>
            <li><strong>Highest Average Power Demand:</strong> {avg_name} on {max_avg.cycle_display_name} at {max_avg.avg_power_kw:.1f} kW</li>
"""
    
    # Range impact
    base_range = next((r for r in range_results if 'Harvester' in r.vehicle_name), None)
    if base_range:
        worst_range = min(range_results, key=lambda x: x.total_range_miles)
        range_reduction = ((base_range.total_range_miles - worst_range.total_range_miles) / base_range.total_range_miles) * 100
        worst_name = worst_range.vehicle_name.replace('Scout Terra EREV + ', '')
        html += f"""            <li><strong>Range Impact:</strong> The {worst_name} configuration reduces total range by {range_reduction:.0f}% compared to the base vehicle ({worst_range.total_range_miles:.0f} mi vs {base_range.total_range_miles:.0f} mi).</li>
"""
    
    html += """        </ul>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #95a5a6; text-align: center;">
        <p>Scout Terra Generator Sizing Study • Road Load Simulator</p>
    </footer>
</body>
</html>
"""
    
    # Write HTML file
    report_path = os.path.join(output_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run the complete generator sizing study."""
    print("=" * 60)
    print("Scout Terra EREV Generator Sizing Study")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, STUDY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load vehicle presets
    print("\nLoading Scout Terra EREV presets...")
    presets = load_scout_terra_presets()
    print(f"Found {len(presets)} configurations:")
    for p in presets:
        print(f"  - {p['name']}")
    
    # ==========================================================================
    # Run single-cycle simulations at 100% SOC
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Running single-cycle simulations (100% SOC)...")
    print("-" * 40)
    
    cycle_results_100 = []
    total_sims = len(presets) * len(DRIVE_CYCLES)
    current_sim = 0
    
    for preset in presets:
        vehicle = preset_to_vehicle_params(preset)
        vehicle_name = preset['name']
        
        for cycle_key, cycle_info in DRIVE_CYCLES.items():
            current_sim += 1
            print(f"[{current_sim}/{total_sims}] {vehicle_name} on {cycle_info['display_name']}...", end=" ")
            
            try:
                result = run_single_cycle_simulation(vehicle, vehicle_name, cycle_key, cycle_info, starting_soc=STARTING_SOC_100)
                cycle_results_100.append(result)
                
                status = "⚠️ DEFICIT" if result.power_deficit_count > 0 else "✓"
                print(f"{status} ({result.distance_miles:.1f} mi, {result.mpge:.1f} MPGe)")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # ==========================================================================
    # Run single-cycle simulations at 50% SOC
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Running single-cycle simulations (50% SOC)...")
    print("-" * 40)
    
    cycle_results_50 = []
    current_sim = 0
    
    for preset in presets:
        vehicle = preset_to_vehicle_params(preset)
        vehicle_name = preset['name']
        
        for cycle_key, cycle_info in DRIVE_CYCLES.items():
            current_sim += 1
            print(f"[{current_sim}/{total_sims}] {vehicle_name} on {cycle_info['display_name']}...", end=" ")
            
            try:
                result = run_single_cycle_simulation(vehicle, vehicle_name, cycle_key, cycle_info, starting_soc=STARTING_SOC_50)
                cycle_results_50.append(result)
                
                status = "⚠️ DEFICIT" if result.power_deficit_count > 0 else "✓"
                print(f"{status} ({result.distance_miles:.1f} mi, {result.mpge:.1f} MPGe)")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # ==========================================================================
    # Run multi-cycle range tests (100% SOC)
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Running multi-cycle range tests (100% SOC)...")
    print("-" * 40)
    
    range_results_multi = []
    
    for preset in presets:
        vehicle = preset_to_vehicle_params(preset)
        vehicle_name = preset['name']
        
        print(f"Multi-cycle range: {vehicle_name}...", end=" ")
        
        try:
            result = run_multi_cycle_range_test(vehicle, vehicle_name, starting_soc=STARTING_SOC_100)
            range_results_multi.append(result)
            print(f"✓ ({result.total_range_miles:.0f} mi total)")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # ==========================================================================
    # Run 75mph steady-state range tests (100% SOC)
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Running 75mph steady-state range tests (100% SOC)...")
    print("-" * 40)
    
    range_results_75mph = []
    
    for preset in presets:
        vehicle = preset_to_vehicle_params(preset)
        vehicle_name = preset['name']
        
        print(f"75mph range: {vehicle_name}...", end=" ")
        
        try:
            result = run_highway_75mph_range_test(vehicle, vehicle_name, starting_soc=STARTING_SOC_100)
            range_results_75mph.append(result)
            exceed_flag = " ⚠️ >100kW" if result.exceeds_generator else ""
            print(f"✓ ({result.total_range_miles:.0f} mi, {result.peak_power_kw:.1f} kW{exceed_flag})")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # ==========================================================================
    # Calculate max generator-only steady-state speed
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Calculating max generator-only speeds...")
    print("-" * 40)
    
    generator_speed_results = []
    
    for preset in presets:
        vehicle = preset_to_vehicle_params(preset)
        vehicle_name = preset['name']
        
        print(f"Max speed: {vehicle_name}...", end=" ")
        
        try:
            result = calculate_max_generator_speed(vehicle, vehicle_name, GENERATOR_CAPACITY_KW)
            generator_speed_results.append(result)
            sustain_flag = "✓" if result.can_sustain_75mph else "✗"
            print(f"{result.max_speed_mph:.1f} mph (75mph: {sustain_flag})")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # ==========================================================================
    # Generate charts
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Generating charts...")
    print("-" * 40)
    
    chart_files = {}
    
    # Use 100% SOC results for main charts
    cycle_results = cycle_results_100
    
    print("  - Power demand comparison...")
    chart_files['power'] = generate_power_demand_chart(cycle_results, output_dir)
    
    print("  - MPGe comparison...")
    chart_files['mpge'] = generate_efficiency_bar_chart(
        cycle_results, output_dir, 'mpge', 'MPGe Efficiency by Configuration (100% SOC)', 'MPGe'
    )
    
    print("  - Generator utilization...")
    chart_files['generator'] = generate_generator_utilization_chart(cycle_results, output_dir)
    
    print("  - SOC profiles (100% SOC)...")
    chart_files['soc_100'] = generate_soc_profile_charts(cycle_results_100, output_dir, starting_soc=100, filename_suffix='_100')
    
    print("  - SOC profiles (50% SOC)...")
    chart_files['soc_50'] = generate_soc_profile_charts(cycle_results_50, output_dir, starting_soc=50, filename_suffix='_50')
    
    print("  - Range comparison...")
    chart_files['range'] = generate_range_comparison_chart(range_results_multi, output_dir)
    
    print("  - Max generator speed chart...")
    chart_files['max_speed'] = generate_max_speed_chart(generator_speed_results, output_dir)
    
    # ==========================================================================
    # Generate HTML report
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Generating HTML report...")
    print("-" * 40)
    
    report_path = generate_html_report_extended(
        presets=presets,
        cycle_results_100=cycle_results_100,
        cycle_results_50=cycle_results_50,
        range_results_multi=range_results_multi,
        range_results_75mph=range_results_75mph,
        generator_speed_results=generator_speed_results,
        output_dir=output_dir,
        chart_files=chart_files
    )
    print(f"Report saved: {report_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Study Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Open {os.path.join(output_dir, 'report.html')} to view the full report.")
    
    # Quick summary
    deficit_count_100 = sum(1 for r in cycle_results_100 if r.power_deficit_count > 0)
    deficit_count_50 = sum(1 for r in cycle_results_50 if r.power_deficit_count > 0)
    exceed_75mph_count = sum(1 for r in range_results_75mph if r.exceeds_generator)
    cannot_sustain_count = sum(1 for r in generator_speed_results if not r.can_sustain_75mph)
    
    print(f"\nQuick Summary:")
    print(f"  - Simulations run: {len(cycle_results_100) + len(cycle_results_50)}")
    print(f"  - Power deficits at 100% SOC: {deficit_count_100}")
    print(f"  - Power deficits at 50% SOC: {deficit_count_50}")
    print(f"  - Configs exceeding 100kW at 75mph: {exceed_75mph_count}")
    print(f"  - Configs that cannot sustain 75mph on generator: {cannot_sustain_count}")
    
    if exceed_75mph_count > 0:
        print(f"\n  ⚠️ WARNING: {exceed_75mph_count} configuration(s) require more than 100 kW at 75 mph!")
        for r in range_results_75mph:
            if r.exceeds_generator:
                name = 'Base' if 'Harvester' in r.vehicle_name else r.vehicle_name.replace('Scout Terra EREV + ', '')
                print(f"     - {name}: {r.peak_power_kw:.1f} kW")
    else:
        print("\n  ✓ All configurations can sustain 75 mph within generator capacity.")


def generate_max_speed_chart(
    results: List[GeneratorSpeedResult],
    output_dir: str
) -> str:
    """Generate chart showing max generator-only speeds."""
    results_sorted = sorted(results, key=lambda x: (0 if 'Harvester' in x.vehicle_name else 1, x.vehicle_name))
    
    short_labels = []
    for r in results_sorted:
        if 'Harvester' in r.vehicle_name:
            short_labels.append('Base')
        else:
            short_labels.append(r.vehicle_name.replace('Scout Terra EREV + ', '').replace(' Trailer', ''))
    
    x = np.arange(len(results_sorted))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    max_speeds = [r.max_speed_mph for r in results_sorted]
    colors = ['#2ecc71' if r.can_sustain_75mph else '#e74c3c' for r in results_sorted]
    
    bars = ax.bar(x, max_speeds, color=colors)
    
    # Add 75 mph reference line
    ax.axhline(y=75, color='blue', linestyle='--', linewidth=2, label='75 mph reference')
    
    # Add value labels
    for i, (speed, r) in enumerate(zip(max_speeds, results_sorted)):
        ax.annotate(f'{speed:.0f}',
                   xy=(i, speed + 2),
                   ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Vehicle Configuration', fontsize=12)
    ax.set_ylabel('Max Sustainable Speed (mph)', fontsize=12)
    ax.set_title('Maximum Steady-State Speed on Generator Power Alone (100 kW)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max_speeds) + 15)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Can sustain 75 mph'),
        Patch(facecolor='#e74c3c', label='Cannot sustain 75 mph'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='75 mph reference')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'max_speed_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'max_speed_comparison.png'


def generate_html_report_extended(
    presets: List[Dict[str, Any]],
    cycle_results_100: List[CycleResult],
    cycle_results_50: List[CycleResult],
    range_results_multi: List[RangeResult],
    range_results_75mph: List[RangeResult],
    generator_speed_results: List[GeneratorSpeedResult],
    output_dir: str,
    chart_files: Dict[str, str]
) -> str:
    """Generate comprehensive HTML report with all study data."""
    
    # Sort vehicles
    vehicles = list(set(r.vehicle_name for r in cycle_results_100))
    vehicles.sort(key=lambda x: (0 if 'Harvester' in x else 1, x))
    
    cycles = list(DRIVE_CYCLES.keys())
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scout Terra Generator Sizing Study</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; }}
        h3 {{ color: #7f8c8d; }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 10px; margin: 20px 0;
        }}
        .summary-box h3 {{ color: white; margin-top: 0; }}
        .warning-box {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white; padding: 20px; border-radius: 10px; margin: 20px 0;
        }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 8px; text-align: center; font-size: 0.9em; }}
        th {{ background: #3498db; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        tr:hover {{ background: #e8f4f8; }}
        .deficit {{ background-color: #ffcccc !important; color: #c0392b; font-weight: bold; }}
        .exceed {{ background-color: #ffe6cc !important; color: #d35400; font-weight: bold; }}
        .ok {{ background-color: #ccffcc !important; color: #27ae60; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .chart-container img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
        .vehicle-name {{ text-align: left; font-weight: 500; }}
        .cycle-header {{ background: #2c3e50 !important; }}
        .metric-header {{ background: #16a085 !important; font-size: 0.85em; }}
        .section {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
        .soc-label {{ background: #9b59b6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>🔋 Scout Terra EREV Generator Sizing Study</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary-box">
        <h3>Study Overview</h3>
        <p>This study evaluates the Scout Terra EREV's <strong>100 kW generator</strong> across various towing configurations and demanding drive cycles.</p>
        <ul>
            <li><strong>Vehicles Tested:</strong> {len(vehicles)} configurations (base + 6 trailer combinations)</li>
            <li><strong>Drive Cycles:</strong> {len(cycles)} cycles (Davis Dam, I-70 Climb, Highway 75 MPH, US06)</li>
            <li><strong>Starting SOC:</strong> 100% and 50%</li>
            <li><strong>EREV Mode:</strong> Blended (generator assists at high power demand)</li>
        </ul>
    </div>
"""
    
    # Check for any exceeds generator scenarios
    exceed_configs = [r for r in range_results_75mph if r.exceeds_generator]
    cannot_sustain = [r for r in generator_speed_results if not r.can_sustain_75mph]
    
    if exceed_configs or cannot_sustain:
        html += """
    <div class="warning-box">
        <h3>⚠️ Generator Capacity Warnings</h3>
        <ul>
"""
        for r in exceed_configs:
            name = 'Base' if 'Harvester' in r.vehicle_name else r.vehicle_name.replace('Scout Terra EREV + ', '')
            html += f"            <li><strong>{name}</strong>: Requires {r.peak_power_kw:.1f} kW at 75 mph (exceeds 100 kW generator)</li>\n"
        for r in cannot_sustain:
            name = 'Base' if 'Harvester' in r.vehicle_name else r.vehicle_name.replace('Scout Terra EREV + ', '')
            html += f"            <li><strong>{name}</strong>: Max sustainable speed on generator alone: {r.max_speed_mph:.0f} mph</li>\n"
        html += """        </ul>
    </div>
"""
    
    # Table 1: Vehicle Configurations
    html += """
    <h2>📊 Table 1: Vehicle Configurations</h2>
    <div class="section">
        <table>
            <tr>
                <th>Configuration</th>
                <th>Mass (kg)</th>
                <th>Mass (lb)</th>
                <th>Frontal Area (m²)</th>
                <th>Cd</th>
                <th>Crr</th>
                <th>Generator (kW)</th>
            </tr>
"""
    
    for preset in presets:
        name = preset['name']
        display_name = 'Base (No Trailer)' if 'Harvester' in name else name.replace('Scout Terra EREV + ', '')
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td>{preset['mass']:.0f}</td>
                <td>{preset['mass'] * 2.205:.0f}</td>
                <td>{preset['frontal_area']:.1f}</td>
                <td>{preset['cd']:.2f}</td>
                <td>{preset['crr']:.3f}</td>
                <td>{preset['generator_power_kw']:.0f}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
"""
    
    # Table 2: Max Generator-Only Speed
    html += """
    <h2>🏎️ Table 2: Maximum Generator-Only Sustainable Speed</h2>
    <div class="section">
        <p>Maximum steady-state speed achievable on flat ground using generator power (100 kW) alone, with no battery assist.</p>
        <table>
            <tr>
                <th>Configuration</th>
                <th>Max Speed (mph)</th>
                <th>Power at Max Speed (kW)</th>
                <th>Power at 75 mph (kW)</th>
                <th>Can Sustain 75 mph?</th>
            </tr>
"""
    
    for r in sorted(generator_speed_results, key=lambda x: (0 if 'Harvester' in x.vehicle_name else 1, x.vehicle_name)):
        display_name = 'Base (No Trailer)' if 'Harvester' in r.vehicle_name else r.vehicle_name.replace('Scout Terra EREV + ', '')
        sustain_class = 'ok' if r.can_sustain_75mph else 'exceed'
        sustain_text = 'Yes ✓' if r.can_sustain_75mph else 'No ✗'
        power_class = '' if r.power_at_75mph_kw <= 100 else 'exceed'
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td><strong>{r.max_speed_mph:.1f}</strong></td>
                <td>{r.power_at_max_speed_kw:.1f}</td>
                <td class="{power_class}">{r.power_at_75mph_kw:.1f}</td>
                <td class="{sustain_class}">{sustain_text}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
"""
    
    # Table 3: Power Demand Summary
    html += """
    <h2>⚡ Table 3: Power Demand Summary</h2>
    <div class="section">
        <p>Peak and average power demand at the wheels. Values exceeding 100 kW are highlighted.</p>
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    for cycle_key in cycles:
        display = DRIVE_CYCLES[cycle_key]['display_name']
        html += f'                <th colspan="2" class="cycle-header">{display}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    for _ in cycles:
        html += '                <th class="metric-header">Peak (kW)</th>\n'
        html += '                <th class="metric-header">Avg (kW)</th>\n'
    
    html += """            </tr>
"""
    
    for vehicle in vehicles:
        display_name = 'Base (No Trailer)' if 'Harvester' in vehicle else vehicle.replace('Scout Terra EREV + ', '')
        html += f'            <tr>\n                <td class="vehicle-name">{display_name}</td>\n'
        
        for cycle_key in cycles:
            matching = [r for r in cycle_results_100 if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                r = matching[0]
                peak_class = 'exceed' if r.peak_power_kw > 100 else ''
                avg_class = 'exceed' if r.avg_power_kw > 100 else ''
                html += f'                <td class="{peak_class}">{r.peak_power_kw:.1f}</td>\n'
                html += f'                <td class="{avg_class}">{r.avg_power_kw:.1f}</td>\n'
            else:
                html += '                <td>-</td>\n                <td>-</td>\n'
        
        html += '            </tr>\n'
    
    html += """        </table>
    </div>
"""
    
    # Table 4: Efficiency Results at 100% SOC
    html += """
    <h2>🔥 Table 4: Efficiency Results <span class="soc-label">100% SOC</span></h2>
    <div class="section">
        <p><strong>Overall mi/kWh</strong> shows total energy efficiency (battery + fuel combined). This is the most accurate measure of drive cycle energy intensity.</p>
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    for cycle_key in cycles:
        display = DRIVE_CYCLES[cycle_key]['display_name']
        html += f'                <th colspan="3" class="cycle-header">{display}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    for _ in cycles:
        html += '                <th class="metric-header">Overall mi/kWh</th>\n'
        html += '                <th class="metric-header">MPG</th>\n'
        html += '                <th class="metric-header">MPGe</th>\n'
    
    html += """            </tr>
"""
    
    for vehicle in vehicles:
        display_name = 'Base (No Trailer)' if 'Harvester' in vehicle else vehicle.replace('Scout Terra EREV + ', '')
        html += f'            <tr>\n                <td class="vehicle-name">{display_name}</td>\n'
        
        for cycle_key in cycles:
            matching = [r for r in cycle_results_100 if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                r = matching[0]
                mpg_display = f'{r.mpg:.1f}' if r.fuel_gallons > 0.01 else 'N/A'
                html += f'                <td>{r.overall_mi_per_kwh:.2f}</td>\n'
                html += f'                <td>{mpg_display}</td>\n'
                html += f'                <td>{r.mpge:.1f}</td>\n'
            else:
                html += '                <td>-</td>\n                <td>-</td>\n                <td>-</td>\n'
        
        html += '            </tr>\n'
    
    html += """        </table>
    </div>
"""
    
    # Table 5: Efficiency Results at 50% SOC
    html += """
    <h2>🔥 Table 5: Efficiency Results <span class="soc-label">50% SOC</span></h2>
    <div class="section">
        <p><strong>Overall mi/kWh</strong> shows total energy efficiency (battery + fuel combined). This is the most accurate measure of drive cycle energy intensity.</p>
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    for cycle_key in cycles:
        display = DRIVE_CYCLES[cycle_key]['display_name']
        html += f'                <th colspan="3" class="cycle-header">{display}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    for _ in cycles:
        html += '                <th class="metric-header">Overall mi/kWh</th>\n'
        html += '                <th class="metric-header">MPG</th>\n'
        html += '                <th class="metric-header">MPGe</th>\n'
    
    html += """            </tr>
"""
    
    for vehicle in vehicles:
        display_name = 'Base (No Trailer)' if 'Harvester' in vehicle else vehicle.replace('Scout Terra EREV + ', '')
        html += f'            <tr>\n                <td class="vehicle-name">{display_name}</td>\n'
        
        for cycle_key in cycles:
            matching = [r for r in cycle_results_50 if r.vehicle_name == vehicle and r.cycle_name == cycle_key]
            if matching:
                r = matching[0]
                mpg_display = f'{r.mpg:.1f}' if r.fuel_gallons > 0.01 else 'N/A'
                html += f'                <td>{r.overall_mi_per_kwh:.2f}</td>\n'
                html += f'                <td>{mpg_display}</td>\n'
                html += f'                <td>{r.mpge:.1f}</td>\n'
            else:
                html += '                <td>-</td>\n                <td>-</td>\n                <td>-</td>\n'
        
        html += '            </tr>\n'
    
    html += """        </table>
    </div>
"""
    
    # Table 6: Multi-Cycle Range Test
    html += """
    <h2>🛣️ Table 6: Multi-Cycle Range Test Results (100% SOC)</h2>
    <div class="section">
        <p>EPA-style multi-cycle range test (4×UDDS + 2×HWFET + 2×Constant 70mph).</p>
        <table>
            <tr>
                <th>Configuration</th>
                <th>Total Range (mi)</th>
                <th>EV Range (mi)</th>
                <th>Generator Range (mi)</th>
                <th>Fuel Used (gal)</th>
                <th>MPGe</th>
            </tr>
"""
    
    for r in sorted(range_results_multi, key=lambda x: (0 if 'Harvester' in x.vehicle_name else 1, x.vehicle_name)):
        display_name = 'Base (No Trailer)' if 'Harvester' in r.vehicle_name else r.vehicle_name.replace('Scout Terra EREV + ', '')
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td><strong>{r.total_range_miles:.0f}</strong></td>
                <td>{r.ev_range_miles:.0f}</td>
                <td>{r.generator_range_miles:.0f}</td>
                <td>{r.fuel_used_gallons:.1f}</td>
                <td>{r.mpge:.1f}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
"""
    
    # Build a lookup for max generator speed
    max_speed_lookup = {r.vehicle_name: r.max_speed_mph for r in generator_speed_results}
    
    # Table 7: 75mph Steady-State Range Test
    html += """
    <h2>🚗 Table 7: 75 MPH Steady-State Range Test (100% SOC)</h2>
    <div class="section">
        <p>Range at constant 75 mph on flat ground starting from 100% SOC. Peak power values exceeding 100 kW generator capacity are highlighted.</p>
        <p><strong>Note:</strong> "Generator-Assisted Range" shows miles traveled while generator is running. When power demand exceeds 100 kW, 
        the battery supplements the generator. All configurations deplete their 15-gallon fuel tank at roughly the same rate because 
        the generator output is capped at 100 kW, but high-drag configurations would be speed-limited to "Max Gen Speed" after battery depletion.</p>
        <table>
            <tr>
                <th>Configuration</th>
                <th>Total Range (mi)</th>
                <th>EV Range (mi)</th>
                <th>Gen-Assisted Range (mi)</th>
                <th>Peak Power (kW)</th>
                <th>Exceeds 100kW?</th>
                <th>Max Gen Speed (mph)</th>
            </tr>
"""
    
    for r in sorted(range_results_75mph, key=lambda x: (0 if 'Harvester' in x.vehicle_name else 1, x.vehicle_name)):
        display_name = 'Base (No Trailer)' if 'Harvester' in r.vehicle_name else r.vehicle_name.replace('Scout Terra EREV + ', '')
        exceed_class = 'exceed' if r.exceeds_generator else ''
        exceed_text = 'Yes ⚠️' if r.exceeds_generator else 'No ✓'
        max_speed = max_speed_lookup.get(r.vehicle_name, 0)
        speed_class = '' if max_speed >= 75 else 'exceed'
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td><strong>{r.total_range_miles:.0f}</strong></td>
                <td>{r.ev_range_miles:.0f}</td>
                <td>{r.generator_range_miles:.0f}</td>
                <td class="{exceed_class}">{r.peak_power_kw:.1f}</td>
                <td class="{exceed_class}">{exceed_text}</td>
                <td class="{speed_class}">{max_speed:.0f}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
"""
    
    # Charts
    html += f"""
    <h2>📈 Charts</h2>
    
    <div class="chart-container">
        <h3>Maximum Generator-Only Sustainable Speed</h3>
        <img src="{chart_files.get('max_speed', 'max_speed_comparison.png')}" alt="Max Speed Comparison">
    </div>
    
    <div class="chart-container">
        <h3>Power Demand vs Generator Capacity</h3>
        <img src="{chart_files.get('power', 'power_demand_comparison.png')}" alt="Power Demand Comparison">
    </div>
    
    <div class="chart-container">
        <h3>MPGe Efficiency Comparison</h3>
        <img src="{chart_files.get('mpge', 'mpge_comparison.png')}" alt="MPGe Comparison">
    </div>
    
    <div class="chart-container">
        <h3>Generator Utilization</h3>
        <img src="{chart_files.get('generator', 'generator_utilization.png')}" alt="Generator Utilization">
    </div>
    
    <div class="chart-container">
        <h3>SOC Profiles by Drive Cycle (100% SOC Start)</h3>
        <img src="{chart_files.get('soc_100', 'soc_profiles_100.png')}" alt="SOC Profiles 100%">
    </div>
    
    <div class="chart-container">
        <h3>SOC Profiles by Drive Cycle (50% SOC Start)</h3>
        <img src="{chart_files.get('soc_50', 'soc_profiles_50.png')}" alt="SOC Profiles 50%">
    </div>
    
    <div class="chart-container">
        <h3>Multi-Cycle Range Comparison</h3>
        <img src="{chart_files.get('range', 'range_comparison.png')}" alt="Range Comparison">
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #95a5a6; text-align: center;">
        <p>Scout Terra Generator Sizing Study • Road Load Simulator</p>
    </footer>
</body>
</html>
"""
    
    # Write HTML file
    report_path = os.path.join(output_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


if __name__ == '__main__':
    main()
