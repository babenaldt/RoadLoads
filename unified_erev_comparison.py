#!/usr/bin/env python3
"""
Unified EREV Generator Sizing Study
====================================
Combined analysis of Ram 1500 Ramcharger and Scout Terra EREV configurations.
Generates comparative HTML report with direct vehicle-to-vehicle comparisons.

Features:
- Side-by-side comparison of Ramcharger (130 kW) vs Scout Terra (100 kW)
- Batch simulation across multiple drive cycles
- Range testing (multi-cycle and highway 75mph)
- Maximum generator-sustained speed analysis
- Comprehensive charts and tables

Author: GitHub Copilot
Date: January 2026
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
STUDY_NAME = "unified_erev_comparison_" + datetime.now().strftime('%Y-%m-%d')
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


# =============================================================================
# Vehicle Management
# =============================================================================
@dataclass
class VehicleGroup:
    """Group of vehicles with their generator capacity."""
    name: str
    display_name: str
    generator_kw: float
    presets: List[Dict[str, Any]]
    color: str  # For charts


def load_vehicle_presets(
    json_path: str = 'vehicle_presets.json',
    include_ramcharger: bool = True,
    include_scout: bool = True
) -> Dict[str, VehicleGroup]:
    """Load vehicle presets and organize by type."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    groups = {}
    
    if include_ramcharger:
        ram_presets = [
            p for p in data['presets'] 
            if 'Ram 1500 REV' in p.get('name', '')
        ]
        if ram_presets:
            groups['ramcharger'] = VehicleGroup(
                name='ramcharger',
                display_name='Ram 1500 REV',
                generator_kw=130.0,
                presets=ram_presets,
                color='#C41E3A'  # Ram red
            )
    
    if include_scout:
        scout_presets = [
            p for p in data['presets'] 
            if 'Scout Terra EREV' in p.get('name', '')
        ]
        if scout_presets:
            groups['scout'] = VehicleGroup(
                name='scout',
                display_name='Scout Terra EREV',
                generator_kw=100.0,
                presets=scout_presets,
                color='#2C5F2D'  # Scout green
            )
    
    return groups


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


def get_short_vehicle_name(full_name: str) -> str:
    """Get short name for charts."""
    if 'Ram 1500 REV' in full_name:
        if '+' not in full_name:
            return 'Ram 1500 REV'
        else:
            return full_name.replace('Ram 1500 REV + ', 'REV+')
    elif 'Scout Terra EREV' in full_name:
        if '+' not in full_name:
            return 'Scout Terra'
        else:
            return full_name.replace('Scout Terra EREV + ', 'Scout+')
    return full_name


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
    
    # mi/kWh (battery only)
    if battery_energy_kwh > 0:
        metrics['mi_per_kwh'] = distance_miles / battery_energy_kwh
    
    # Overall mi/kWh (total energy efficiency)
    fuel_energy_kwh = fuel_gallons * GASOLINE_ENERGY_KWH_PER_GALLON
    total_energy_kwh = battery_energy_kwh + fuel_energy_kwh
    if total_energy_kwh > 0:
        metrics['overall_mi_per_kwh'] = distance_miles / total_energy_kwh
    
    # MPG (fuel only)
    if fuel_gallons > 0:
        metrics['mpg'] = distance_miles / fuel_gallons
    
    # MPGe (combined efficiency per EPA)
    if total_energy_kwh > 0:
        kwh_per_gallon_gasoline_equivalent = 33.7
        metrics['mpge'] = (distance_miles / total_energy_kwh) * kwh_per_gallon_gasoline_equivalent
    
    return metrics


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class CycleResult:
    """Results from a single drive cycle simulation."""
    vehicle_group: str  # 'ramcharger' or 'scout'
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
    cycle_failed: bool
    min_soc_required: float  # Minimum starting SOC needed to complete cycle
    usable_battery_pct: float  # For converting to customer-facing SOC


@dataclass
class RangeResult:
    """Results from range testing."""
    vehicle_group: str
    vehicle_name: str
    test_type: str  # 'multi_cycle' or 'highway_75mph'
    starting_soc: float
    total_range_miles: float
    ev_range_miles: float
    generator_range_miles: float
    fuel_used_gallons: float
    battery_energy_kwh: float
    mpge: float
    peak_power_kw: float = 0.0
    exceeds_generator: bool = False


@dataclass
class GeneratorSpeedResult:
    """Maximum speed achievable on generator alone."""
    vehicle_group: str
    vehicle_name: str
    max_speed_mph: float
    power_at_max_speed_kw: float
    power_at_75mph_kw: float
    can_sustain_75mph: bool


# =============================================================================
# Simulation Runners
# =============================================================================
def run_single_cycle_simulation(
    vehicle: VehicleParams,
    vehicle_group: str,
    vehicle_name: str,
    cycle_key: str,
    cycle_info: Dict[str, str],
    starting_soc: float = 100.0
) -> CycleResult:
    """Run simulation for a single vehicle/cycle combination."""
    cycle_file = cycle_info['file']
    display_name = cycle_info['display_name']
    
    # Load cycle and calculate road load
    cycle = load_drive_cycle(cycle_file)
    road_load = calculate_road_load(vehicle, cycle)
    
    # Peak and average power
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
    
    # Calculate minimum SOC required to complete cycle (customer-visible percentage)
    if erev_result.cycle_failed:
        min_soc_required = -1.0  # Flag for "cannot complete"
    else:
        # Calculate energy needed beyond what generator can provide
        # Power demand in watts
        power_demand_w = road_load.power
        generator_capacity_w = vehicle.generator_power_kw * 1000.0
        
        # Calculate energy deficit (power beyond generator capacity)
        power_deficit_w = np.maximum(0, power_demand_w - generator_capacity_w)
        
        # Integrate to get total energy deficit
        # Account for drivetrain efficiency - battery must provide more than the wheel demand
        dt = np.diff(np.array([p.time for p in cycle]), prepend=0)
        energy_deficit_j = np.sum((power_deficit_w / vehicle.drivetrain_efficiency) * dt)
        energy_deficit_kwh = energy_deficit_j / 3.6e6
        
        # Convert to percentage of usable battery capacity
        usable_capacity_kwh = vehicle.battery_capacity * (vehicle.usable_battery_pct / 100.0)
        if usable_capacity_kwh > 0:
            min_soc_required = (energy_deficit_kwh / usable_capacity_kwh) * 100.0
        else:
            min_soc_required = 0.0
    
    return CycleResult(
        vehicle_group=vehicle_group,
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
        distance_trace=erev_result.distance_miles_trace,
        cycle_failed=erev_result.cycle_failed,
        min_soc_required=min_soc_required,
        usable_battery_pct=vehicle.usable_battery_pct
    )


def run_multi_cycle_range_test(
    vehicle: VehicleParams,
    vehicle_group: str,
    vehicle_name: str,
    starting_soc: float = 100.0
) -> RangeResult:
    """Run multi-cycle range test."""
    result = estimate_erev_range_multi_cycle(vehicle, starting_soc=starting_soc)
    
    return RangeResult(
        vehicle_group=vehicle_group,
        vehicle_name=vehicle_name,
        test_type='multi_cycle',
        starting_soc=starting_soc,
        total_range_miles=result.get('range_miles', 0.0),
        ev_range_miles=result.get('ev_only_miles', 0.0),
        generator_range_miles=result.get('generator_miles', 0.0),
        fuel_used_gallons=result.get('fuel_used_gallons', 0.0),
        battery_energy_kwh=result.get('battery_energy_kwh', 0.0),
        mpge=result.get('mpge', 0.0)
    )


def calculate_max_generator_speed(
    vehicle: VehicleParams,
    vehicle_group: str,
    vehicle_name: str,
    generator_power_kw: float
) -> GeneratorSpeedResult:
    """Calculate maximum steady-state speed on generator alone."""
    generator_power_w = generator_power_kw * 1000.0
    
    def road_load_power(v_mps):
        """Calculate road load power at given speed."""
        p_aero = 0.5 * AIR_DENSITY * vehicle.drag_coefficient * vehicle.frontal_area * (v_mps ** 3)
        p_rolling = vehicle.rolling_resistance * vehicle.mass * GRAVITY * v_mps
        p_total = (p_aero + p_rolling) / vehicle.drivetrain_efficiency
        return p_total
    
    def power_balance(v_mps):
        """Power balance equation to solve."""
        return road_load_power(v_mps) - generator_power_w
    
    # Find max speed
    try:
        max_speed_mps = brentq(power_balance, 1.0, 60.0)
    except ValueError:
        max_speed_mps = 60.0
    
    max_speed_mph = max_speed_mps * MPS_TO_MPH
    power_at_max = road_load_power(max_speed_mps) / 1000.0
    
    # Calculate power at 75 mph
    speed_75mph_mps = 75.0 / MPS_TO_MPH
    power_at_75mph = road_load_power(speed_75mph_mps) / 1000.0
    
    return GeneratorSpeedResult(
        vehicle_group=vehicle_group,
        vehicle_name=vehicle_name,
        max_speed_mph=max_speed_mph,
        power_at_max_speed_kw=power_at_max,
        power_at_75mph_kw=power_at_75mph,
        can_sustain_75mph=(power_at_75mph <= generator_power_kw)
    )


# =============================================================================
# Chart Generation - Cross-Vehicle Comparisons
# =============================================================================
def generate_cross_vehicle_mpge_chart(
    results: List[CycleResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Compare base vehicle MPGe across cycles (Ramcharger vs Scout)."""
    cycles = list(DRIVE_CYCLES.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(cycles))
    width = 0.2
    
    # Get base vehicle and Streamlined RV results for each group
    bar_idx = 0
    for group_key, group in vehicle_groups.items():
        # Base vehicle
        base_name = group.presets[0]['name']
        base_results = []
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.cycle_name == cycle_key and r.vehicle_name == base_name), None)
            if result:
                base_results.append(result.mpge)
            else:
                base_results.append(0)
        
        offset = width * (bar_idx - 1.5)
        bars = ax.bar(x + offset, base_results, width, label=f'{group.display_name} (Base)',
                     color=group.color, alpha=0.8)
        bar_idx += 1
        
        # Streamlined RV
        rv_name = next((p['name'] for p in group.presets if 'Streamlined RV' in p['name']), None)
        if rv_name:
            rv_results = []
            for cycle_key in cycles:
                result = next((r for r in results if r.vehicle_group == group_key and 
                              r.cycle_name == cycle_key and r.vehicle_name == rv_name), None)
                if result:
                    rv_results.append(result.mpge)
                else:
                    rv_results.append(0)
            
            offset = width * (bar_idx - 1.5)
            bars_rv = ax.bar(x + offset, rv_results, width, label=f'{group.display_name} (Stream. RV)',
                           color=group.color, alpha=0.5)
            bar_idx += 1
    
    ax.set_xlabel('Drive Cycle', fontsize=12)
    ax.set_ylabel('MPGe', fontsize=12)
    ax.set_title('Base Vehicle Efficiency Comparison: Ram 1500 REV vs Scout Terra', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cross_vehicle_mpge.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'cross_vehicle_mpge.png'


def generate_cross_vehicle_range_chart(
    range_results: List[RangeResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Compare base vehicle range (Ramcharger vs Scout)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(vehicle_groups))
    width = 0.6
    
    ev_ranges = []
    gen_ranges = []
    labels = []
    colors = []
    
    for group_key, group in vehicle_groups.items():
        # Find base vehicle range result
        base_name = group.presets[0]['name']
        result = next((r for r in range_results if r.vehicle_group == group_key and
                      r.vehicle_name == base_name and r.test_type == 'multi_cycle'), None)
        
        if result:
            ev_ranges.append(result.ev_range_miles)
            gen_ranges.append(result.generator_range_miles)
            labels.append(f'{group.display_name}\n({int(group.generator_kw)} kW gen)')
            colors.append(group.color)
        else:
            ev_ranges.append(0)
            gen_ranges.append(0)
            labels.append(group.display_name)
            colors.append(group.color)
    
    # Stacked bars - create separate bars for each vehicle to show in legend
    from matplotlib.patches import Patch
    
    for i in range(len(x)):
        ax.bar(i, ev_ranges[i], width, color=colors[i], alpha=0.6)
        ax.bar(i, gen_ranges[i], width, bottom=ev_ranges[i], color=colors[i], alpha=1.0)
    
    # Add total labels
    for i in range(len(x)):
        total = ev_ranges[i] + gen_ranges[i]
        if total > 0:
            ax.text(i, total, f'{total:.0f} mi', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
    
    # Create custom legend with all colors
    legend_elements = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        vehicle_name = label.split('\n')[0]  # Get just the vehicle name
        legend_elements.append(Patch(facecolor=color, alpha=0.6, label=f'{vehicle_name} - EV Range'))
        legend_elements.append(Patch(facecolor=color, alpha=1.0, label=f'{vehicle_name} - Generator Range'))
    
    ax.set_ylabel('Range (miles)', fontsize=12)
    ax.set_title('Multi-Cycle Range Comparison: Base Vehicles', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cross_vehicle_range.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'cross_vehicle_range.png'


def generate_cross_vehicle_power_chart(
    results: List[CycleResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Compare power demand vs generator capacity for base and Streamlined RV."""
    cycles = list(DRIVE_CYCLES.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(cycles))
    width = 0.2
    
    # Plot peak power for each vehicle group's base vehicle and Streamlined RV
    bar_idx = 0
    for group_key, group in vehicle_groups.items():
        # Base vehicle
        base_name = group.presets[0]['name']
        peak_powers_base = []
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.cycle_name == cycle_key and r.vehicle_name == base_name), None)
            if result:
                peak_powers_base.append(result.peak_power_kw)
            else:
                peak_powers_base.append(0)
        
        offset = width * (bar_idx - 1.5)
        ax.bar(x + offset, peak_powers_base, width, label=f'{group.display_name} (Base)',
              color=group.color, alpha=0.8)
        bar_idx += 1
        
        # Streamlined RV
        rv_name = next((p['name'] for p in group.presets if 'Streamlined RV' in p['name']), None)
        if rv_name:
            peak_powers_rv = []
            for cycle_key in cycles:
                result = next((r for r in results if r.vehicle_group == group_key and 
                              r.cycle_name == cycle_key and r.vehicle_name == rv_name), None)
                if result:
                    peak_powers_rv.append(result.peak_power_kw)
                else:
                    peak_powers_rv.append(0)
            
            offset = width * (bar_idx - 1.5)
            ax.bar(x + offset, peak_powers_rv, width, label=f'{group.display_name} (Stream. RV)',
                  color=group.color, alpha=0.5)
            bar_idx += 1
    
    # Add generator capacity lines
    for group_key, group in vehicle_groups.items():
        ax.axhline(y=group.generator_kw, color=group.color, linestyle='--', linewidth=2,
                  label=f'{group.display_name} Generator ({int(group.generator_kw)} kW)')
    
    ax.set_xlabel('Drive Cycle', fontsize=12)
    ax.set_ylabel('Peak Power (kW)', fontsize=12)
    ax.set_title('Peak Power Demand vs Generator Capacity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cross_vehicle_power.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'cross_vehicle_power.png'


def generate_max_speed_comparison_chart(
    speed_results: List[GeneratorSpeedResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Compare maximum generator-sustained speeds for all configurations."""
    # Organize results by vehicle group
    fig, ax = plt.subplots(figsize=(16, 7))
    
    all_results = []
    all_labels = []
    all_colors = []
    
    for group_key, group in vehicle_groups.items():
        group_results = [r for r in speed_results if r.vehicle_group == group_key]
        # Sort: base first, then others
        group_results.sort(key=lambda x: ('+' in x.vehicle_name, x.vehicle_name))
        
        for result in group_results:
            all_results.append(result)
            short_name = get_short_vehicle_name(result.vehicle_name)
            all_labels.append(short_name)
            all_colors.append(group.color)
    
    if not all_results:
        return ''
    
    x = np.arange(len(all_results))
    speeds = [r.max_speed_mph for r in all_results]
    
    bars = ax.bar(x, speeds, color=all_colors, alpha=0.8)
    
    # Add 75 mph reference line
    ax.axhline(y=75, color='red', linestyle='--', linewidth=2, label='75 mph target')
    
    # Add value labels for all speeds
    for i, (bar, result) in enumerate(zip(bars, all_results)):
        height = bar.get_height()
        # Color code labels: red if below 75, normal otherwise
        color = 'red' if height < 75 else 'black'
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.0f}',
               ha='center', va='bottom', fontsize=8, color=color, fontweight='bold' if height < 75 else 'normal')
    
    # Add padding at top for text labels
    y_max = max(speeds) if speeds else 120
    ax.set_ylim(0, max(y_max * 1.1, 130))
    
    ax.set_ylabel('Maximum Speed (mph)', fontsize=12)
    ax.set_title('Maximum Generator-Sustained Speed Comparison (All Configurations)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cross_vehicle_max_speed.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'cross_vehicle_max_speed.png'


# =============================================================================
# Chart Generation - Individual Vehicle Groups
# =============================================================================
def generate_efficiency_bar_chart(
    results: List[CycleResult],
    vehicle_group: str,
    output_dir: str,
    metric: str,
    title: str,
    ylabel: str
) -> str:
    """Generate efficiency bar chart for one vehicle group."""
    group_results = [r for r in results if r.vehicle_group == vehicle_group]
    if not group_results:
        return ''
    
    cycles = list(DRIVE_CYCLES.keys())
    vehicles = list(set(r.vehicle_name for r in group_results))
    vehicles.sort()
    
    short_labels = [get_short_vehicle_name(v) for v in vehicles]
    
    x = np.arange(len(cycles))
    width = 0.12
    n_vehicles = len(vehicles)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        values = []
        for cycle_key in cycles:
            result = next((r for r in group_results if r.vehicle_name == vehicle and 
                          r.cycle_name == cycle_key), None)
            if result:
                values.append(getattr(result, metric))
            else:
                values.append(0)
        
        offset = width * (i - n_vehicles/2)
        ax.bar(x + offset, values, width, label=label)
    
    ax.set_xlabel('Drive Cycle', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{vehicle_group}_{metric}_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_power_demand_chart(
    results: List[CycleResult],
    vehicle_group: str,
    generator_capacity: float,
    output_dir: str
) -> str:
    """Generate peak vs average power chart for one vehicle group."""
    group_results = [r for r in results if r.vehicle_group == vehicle_group]
    if not group_results:
        return ''
    
    cycles = list(DRIVE_CYCLES.keys())
    vehicles = list(set(r.vehicle_name for r in group_results))
    vehicles.sort()
    
    short_labels = [get_short_vehicle_name(v) for v in vehicles]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(cycles))
    width = 0.12
    n_vehicles = len(vehicles)
    
    # Peak power
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        peak_powers = []
        for cycle_key in cycles:
            result = next((r for r in group_results if r.vehicle_name == vehicle and 
                          r.cycle_name == cycle_key), None)
            if result:
                peak_powers.append(result.peak_power_kw)
            else:
                peak_powers.append(0)
        
        offset = width * (i - n_vehicles/2)
        axes[0].bar(x + offset, peak_powers, width, label=label)
    
    axes[0].axhline(y=generator_capacity, color='red', linestyle='--', linewidth=2,
                   label=f'Generator ({int(generator_capacity)} kW)')
    axes[0].set_xlabel('Drive Cycle', fontsize=12)
    axes[0].set_ylabel('Peak Power (kW)', fontsize=12)
    axes[0].set_title('Peak Power Demand', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Average power
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        avg_powers = []
        for cycle_key in cycles:
            result = next((r for r in group_results if r.vehicle_name == vehicle and 
                          r.cycle_name == cycle_key), None)
            if result:
                avg_powers.append(result.avg_power_kw)
            else:
                avg_powers.append(0)
        
        offset = width * (i - n_vehicles/2)
        axes[1].bar(x + offset, avg_powers, width, label=label)
    
    axes[1].axhline(y=generator_capacity, color='red', linestyle='--', linewidth=2,
                   label=f'Generator ({int(generator_capacity)} kW)')
    axes[1].set_xlabel('Drive Cycle', fontsize=12)
    axes[1].set_ylabel('Average Power (kW)', fontsize=12)
    axes[1].set_title('Average Power Demand', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{vehicle_group}_power_demand.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_range_comparison_chart(
    range_results: List[RangeResult],
    vehicle_group: str,
    output_dir: str
) -> str:
    """Generate range comparison chart for one vehicle group."""
    group_results = [r for r in range_results if r.vehicle_group == vehicle_group 
                     and r.test_type == 'multi_cycle']
    if not group_results:
        return ''
    
    group_results_sorted = sorted(group_results, key=lambda x: x.vehicle_name)
    short_labels = [get_short_vehicle_name(r.vehicle_name) for r in group_results_sorted]
    
    x = np.arange(len(group_results_sorted))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ev_ranges = [r.ev_range_miles for r in group_results_sorted]
    gen_ranges = [r.generator_range_miles for r in group_results_sorted]
    
    bars1 = ax.bar(x, ev_ranges, width, label='EV Range', color='#2ecc71')
    bars2 = ax.bar(x, gen_ranges, width, bottom=ev_ranges, label='Generator Range', color='#e74c3c')
    
    # Add total labels
    for i, r in enumerate(group_results_sorted):
        total = r.total_range_miles
        ax.text(i, total, f'{total:.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Range (miles)', fontsize=12)
    ax.set_title('Multi-Cycle Range Test Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{vehicle_group}_range.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_soc_profile_charts(
    results: List[CycleResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Generate SOC profile charts for each cycle, comparing both vehicles."""
    cycles = list(DRIVE_CYCLES.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, cycle_key in enumerate(cycles):
        ax = axes[idx]
        cycle_results = [r for r in results if r.cycle_name == cycle_key]
        
        # Group results by vehicle group
        for group_key, group in vehicle_groups.items():
            base_name = group.presets[0]['name']
            
            # Plot base vehicle with solid line
            base_result = next((r for r in cycle_results if r.vehicle_group == group_key and 
                              r.vehicle_name == base_name), None)
            if base_result:
                # Convert to customer-facing SOC
                min_soc_limit = 100.0 - base_result.usable_battery_pct
                customer_soc = (base_result.soc_timeline - min_soc_limit) / base_result.usable_battery_pct * 100.0
                customer_soc = np.clip(customer_soc, 0, 100)
                
                ax.plot(base_result.distance_trace, customer_soc, 
                       label=f'{group.display_name} (Base)',
                       linestyle='-', linewidth=2.5, color=group.color, alpha=0.9)
            
            # Plot trailers with dashed lines (lighter)
            trailer_results = [r for r in cycle_results if r.vehicle_group == group_key and 
                             r.vehicle_name != base_name]
            for r in trailer_results:
                short_name = get_short_vehicle_name(r.vehicle_name)
                # Convert to customer-facing SOC
                min_soc_limit = 100.0 - r.usable_battery_pct
                customer_soc = (r.soc_timeline - min_soc_limit) / r.usable_battery_pct * 100.0
                customer_soc = np.clip(customer_soc, 0, 100)
                
                ax.plot(r.distance_trace, customer_soc, 
                       label=short_name,
                       linestyle='--', linewidth=1.2, color=group.color, alpha=0.4)
        
        ax.set_xlabel('Distance (miles)', fontsize=11)
        ax.set_ylabel('State of Charge (%)', fontsize=11)
        ax.set_title(f'{DRIVE_CYCLES[cycle_key]["display_name"]} - SOC Profile', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cross_vehicle_soc_profiles.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'cross_vehicle_soc_profiles.png'


def generate_generator_utilization_chart(
    results: List[CycleResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Generate cross-vehicle generator runtime and energy comparison for base vehicles."""
    cycles = list(DRIVE_CYCLES.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(cycles))
    width = 0.35
    
    # Generator runtime chart - base vehicles only
    for i, (group_key, group) in enumerate(vehicle_groups.items()):
        base_name = group.presets[0]['name']
        runtimes = []
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.vehicle_name == base_name and r.cycle_name == cycle_key), None)
            if result:
                runtimes.append(result.generator_runtime_min)
            else:
                runtimes.append(0)
        
        offset = width * (i - 0.5)
        axes[0].bar(x + offset, runtimes, width, label=f'{group.display_name}',
                   color=group.color, alpha=0.8)
    
    axes[0].set_xlabel('Drive Cycle', fontsize=12)
    axes[0].set_ylabel('Generator Runtime (min)', fontsize=12)
    axes[0].set_title('Base Vehicle Generator Runtime', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Generator energy chart - base vehicles only
    for i, (group_key, group) in enumerate(vehicle_groups.items()):
        base_name = group.presets[0]['name']
        energies = []
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.vehicle_name == base_name and r.cycle_name == cycle_key), None)
            if result:
                energies.append(result.generator_energy_kwh)
            else:
                energies.append(0)
        
        offset = width * (i - 0.5)
        axes[1].bar(x + offset, energies, width, label=f'{group.display_name}',
                   color=group.color, alpha=0.8)
    
    axes[1].set_xlabel('Drive Cycle', fontsize=12)
    axes[1].set_ylabel('Generator Energy (kWh)', fontsize=12)
    axes[1].set_title('Base Vehicle Generator Energy Output', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cross_vehicle_generator_utilization.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'cross_vehicle_generator_utilization.png'


def generate_individual_soc_charts(
    results: List[CycleResult],
    vehicle_group: str,
    output_dir: str
) -> str:
    """Generate SOC profile charts for one vehicle group."""
    group_results = [r for r in results if r.vehicle_group == vehicle_group]
    if not group_results:
        return ''
    
    cycles = list(DRIVE_CYCLES.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, cycle_key in enumerate(cycles):
        ax = axes[idx]
        cycle_results = [r for r in group_results if r.cycle_name == cycle_key]
        
        for r in cycle_results:
            short_name = get_short_vehicle_name(r.vehicle_name)
            is_base = '+' not in r.vehicle_name
            
            # Convert to customer-facing SOC
            min_soc_limit = 100.0 - r.usable_battery_pct
            customer_soc = (r.soc_timeline - min_soc_limit) / r.usable_battery_pct * 100.0
            customer_soc = np.clip(customer_soc, 0, 100)
            
            ax.plot(r.distance_trace, customer_soc, 
                   label=short_name,
                   linestyle='-' if is_base else '--',
                   linewidth=2.5 if is_base else 1.5,
                   alpha=0.9 if is_base else 0.6)
        
        ax.set_xlabel('Distance (miles)', fontsize=11)
        ax.set_ylabel('State of Charge (%)', fontsize=11)
        ax.set_title(f'{DRIVE_CYCLES[cycle_key]["display_name"]} - SOC Profile', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    filename = f'{vehicle_group}_soc_profiles.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_individual_generator_utilization(
    results: List[CycleResult],
    vehicle_group: str,
    generator_capacity: float,
    output_dir: str
) -> str:
    """Generate generator utilization charts for one vehicle group."""
    group_results = [r for r in results if r.vehicle_group == vehicle_group]
    if not group_results:
        return ''
    
    cycles = list(DRIVE_CYCLES.keys())
    vehicles = list(set(r.vehicle_name for r in group_results))
    vehicles.sort()
    
    short_labels = [get_short_vehicle_name(v) for v in vehicles]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(cycles))
    width = 0.12
    n_vehicles = len(vehicles)
    
    # Get cycle durations for percentage calculation
    cycle_durations = {}
    for cycle_key, cycle_info in DRIVE_CYCLES.items():
        cycle_points = load_drive_cycle(cycle_info['file'])
        times = np.array([p.time for p in cycle_points])
        cycle_durations[cycle_key] = times[-1] / 60  # Convert to minutes
    
    # Generator runtime chart (as percentage of cycle duration)
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        runtime_percentages = []
        for cycle_key in cycles:
            result = next((r for r in group_results if r.vehicle_name == vehicle and 
                          r.cycle_name == cycle_key), None)
            if result and cycle_durations.get(cycle_key, 0) > 0:
                runtime_pct = (result.generator_runtime_min / cycle_durations[cycle_key]) * 100
                runtime_percentages.append(runtime_pct)
            else:
                runtime_percentages.append(0)
        
        offset = width * (i - n_vehicles/2)
        axes[0].bar(x + offset, runtime_percentages, width, label=label)
    
    axes[0].set_xlabel('Drive Cycle', fontsize=12)
    axes[0].set_ylabel('Generator Runtime (% of cycle)', fontsize=12)
    axes[0].set_title('Generator Runtime by Configuration', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Generator energy chart
    for i, (vehicle, label) in enumerate(zip(vehicles, short_labels)):
        energies = []
        for cycle_key in cycles:
            result = next((r for r in group_results if r.vehicle_name == vehicle and 
                          r.cycle_name == cycle_key), None)
            if result:
                energies.append(result.generator_energy_kwh)
            else:
                energies.append(0)
        
        offset = width * (i - n_vehicles/2)
        axes[1].bar(x + offset, energies, width, label=label)
    
    axes[1].set_xlabel('Drive Cycle', fontsize=12)
    axes[1].set_ylabel('Generator Energy (kWh)', fontsize=12)
    axes[1].set_title('Generator Energy Output by Configuration', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{vehicle_group}_generator_utilization.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_drive_cycle_comparison_table() -> Dict[str, Dict[str, float]]:
    """Analyze drive cycles and return statistics."""
    cycle_stats = {}
    
    for cycle_key, cycle_info in DRIVE_CYCLES.items():
        # Load cycle points
        cycle_points = load_drive_cycle(cycle_info['file'])
        
        # Extract arrays
        times = np.array([p.time for p in cycle_points])
        speeds_mps = np.array([p.speed for p in cycle_points])
        grades = np.array([p.grade for p in cycle_points])
        
        # Calculate distance by integrating speed
        distances = np.zeros(len(times))
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            distances[i] = distances[i-1] + speeds_mps[i] * dt
        
        # Convert speeds to mph
        speeds_mph = speeds_mps * MPS_TO_MPH
        # Grades are already stored as percentages in the CSV files
        grades_pct = grades
        
        # Calculate acceleration (m/s²)
        time_diff = np.diff(times)
        speed_diff = np.diff(speeds_mps)
        acceleration = np.divide(speed_diff, time_diff, where=time_diff>0, out=np.zeros_like(speed_diff))
        
        stats = {
            'distance_miles': distances[-1] / 1609.34,
            'duration_min': times[-1] / 60,
            'avg_speed_mph': np.mean(speeds_mph[speeds_mph > 0]) if np.any(speeds_mph > 0) else 0,
            'max_speed_mph': np.max(speeds_mph),
            'avg_grade_pct': np.mean(grades_pct),
            'max_grade_pct': np.max(grades_pct),
            'min_grade_pct': np.min(grades_pct),
            'max_accel_mps2': np.max(acceleration) if len(acceleration) > 0 else 0,
            'idle_time_pct': (np.sum(speeds_mph < 0.5) / len(speeds_mph)) * 100
        }
        
        cycle_stats[cycle_key] = stats
    
    return cycle_stats


def generate_generator_sizing_analysis_chart(
    results: List[CycleResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> str:
    """Generate comprehensive generator sizing analysis chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Peak Power vs Generator Capacity (scatter)
    ax1 = axes[0, 0]
    
    # Define marker styles for each drive cycle
    cycle_markers = {
        'davis_dam_charger': 'o',      # circle
        'el_rancho_to_frisco': 's',    # square
        'highway_75mph': '^',          # triangle up
        'US06': 'D'                    # diamond
    }
    
    for group_key, group in vehicle_groups.items():
        group_results = [r for r in results if r.vehicle_group == group_key]
        
        # Plot each cycle with different marker
        for cycle_key, marker in cycle_markers.items():
            cycle_results = [r for r in group_results if r.cycle_name == cycle_key]
            if cycle_results:
                peak_powers = [r.peak_power_kw for r in cycle_results]
                avg_powers = [r.avg_power_kw for r in cycle_results]
                
                ax1.scatter(peak_powers, avg_powers, 
                           color=group.color, marker=marker, alpha=0.6, s=150,
                           edgecolors='black', linewidths=0.5)
        
        # Add generator capacity reference lines
        ax1.axvline(x=group.generator_kw, color=group.color, linestyle='--', 
                   linewidth=2, alpha=0.7)
        ax1.axhline(y=group.generator_kw, color=group.color, linestyle=':', 
                   linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('Peak Power Demand (kW)', fontsize=11)
    ax1.set_ylabel('Average Power Demand (kW)', fontsize=11)
    ax1.set_title('Power Demand Analysis: Peak vs Average', fontsize=12, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Vehicle groups
    for group_key, group in vehicle_groups.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=group.color, markersize=10,
                                     label=group.display_name))
    
    # Drive cycles
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='gray', markersize=10,
                                 label='Davis Dam'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor='gray', markersize=10,
                                 label='I-70 Climb'))
    legend_elements.append(Line2D([0], [0], marker='^', color='w', 
                                 markerfacecolor='gray', markersize=10,
                                 label='Highway 75 MPH'))
    legend_elements.append(Line2D([0], [0], marker='D', color='w', 
                                 markerfacecolor='gray', markersize=10,
                                 label='US06'))
    
    # Generator capacity lines
    for group_key, group in vehicle_groups.items():
        legend_elements.append(Line2D([0], [0], color=group.color, linestyle='--',
                                     linewidth=2, label=f'{group.display_name} Gen Cap ({group.generator_kw} kW)'))
    
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Average Power Generator Capacity Utilization
    ax2 = axes[0, 1]
    cycles = list(DRIVE_CYCLES.keys())
    
    # Get base and streamlined RV configurations
    configs_to_plot = []
    for group_key, group in vehicle_groups.items():
        configs_to_plot.append((group_key, group.presets[0]['name'], 'Base'))
        rv_name = next((p['name'] for p in group.presets if 'Streamlined RV' in p['name']), None)
        if rv_name:
            configs_to_plot.append((group_key, rv_name, 'Stream. RV'))
    
    x = np.arange(len(cycles))
    width = 0.2
    
    for idx, (group_key, vehicle_name, label_suffix) in enumerate(configs_to_plot):
        group = vehicle_groups[group_key]
        utilizations = []
        
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.vehicle_name == vehicle_name and r.cycle_name == cycle_key), None)
            if result:
                # Utilization = (avg power / generator capacity) * 100
                utilization = (result.avg_power_kw / group.generator_kw) * 100
                utilizations.append(utilization)
            else:
                utilizations.append(0)
        
        offset = width * (idx - 1.5)
        alpha = 0.8 if label_suffix == 'Base' else 0.5
        ax2.bar(x + offset, utilizations, width, 
               label=f'{group.display_name} ({label_suffix})',
               color=group.color, alpha=alpha)
    
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='100% Capacity')
    ax2.set_xlabel('Drive Cycle', fontsize=11)
    ax2.set_ylabel('Avg Generator Utilization (%)', fontsize=11)
    ax2.set_title('Average Power vs Generator Capacity', fontsize=12, fontweight='bold')
    ax2.set_xticks(np.arange(len(cycles)))
    ax2.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles], rotation=30, ha='right')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Peak Generator Utilization Percentage
    ax3 = axes[1, 0]
    
    for idx, (group_key, vehicle_name, label_suffix) in enumerate(configs_to_plot):
        group = vehicle_groups[group_key]
        utilizations = []
        
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.vehicle_name == vehicle_name and r.cycle_name == cycle_key), None)
            if result:
                # Utilization = (peak power / generator capacity) * 100
                utilization = (result.peak_power_kw / group.generator_kw) * 100
                utilizations.append(utilization)
            else:
                utilizations.append(0)
        
        offset = width * (idx - 1.5)
        alpha = 0.8 if label_suffix == 'Base' else 0.5
        ax3.bar(x + offset, utilizations, width, 
               label=f'{group.display_name} ({label_suffix})',
               color=group.color, alpha=alpha)
    
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='100% Capacity')
    ax3.set_xlabel('Drive Cycle', fontsize=11)
    ax3.set_ylabel('Peak Generator Utilization (%)', fontsize=11)
    ax3.set_title('Peak Power vs Generator Capacity', fontsize=12, fontweight='bold')
    ax3.set_xticks(np.arange(len(cycles)))
    ax3.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles], rotation=30, ha='right')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Chart 4: Generator Runtime Comparison (as percentage)
    ax4 = axes[1, 1]
    
    # Get cycle durations for percentage calculation
    cycle_durations = {}
    for cycle_key, cycle_info in DRIVE_CYCLES.items():
        cycle_points = load_drive_cycle(cycle_info['file'])
        times = np.array([p.time for p in cycle_points])
        cycle_durations[cycle_key] = times[-1] / 60  # Convert to minutes
    
    for idx, (group_key, vehicle_name, label_suffix) in enumerate(configs_to_plot):
        group = vehicle_groups[group_key]
        runtime_percentages = []
        
        for cycle_key in cycles:
            result = next((r for r in results if r.vehicle_group == group_key and 
                          r.vehicle_name == vehicle_name and r.cycle_name == cycle_key), None)
            if result and cycle_durations.get(cycle_key, 0) > 0:
                runtime_pct = (result.generator_runtime_min / cycle_durations[cycle_key]) * 100
                runtime_percentages.append(runtime_pct)
            else:
                runtime_percentages.append(0)
        
        offset = width * (idx - 1.5)
        alpha = 0.8 if label_suffix == 'Base' else 0.5
        ax4.bar(x + offset, runtime_percentages, width, 
               label=f'{group.display_name} ({label_suffix})',
               color=group.color, alpha=alpha)
    
    ax4.set_xlabel('Drive Cycle', fontsize=11)
    ax4.set_ylabel('Generator Runtime (% of cycle)', fontsize=11)
    ax4.set_title('Generator Operating Time', fontsize=12, fontweight='bold')
    ax4.set_xticks(np.arange(len(cycles)))
    ax4.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles], rotation=30, ha='right')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'generator_sizing_analysis.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'generator_sizing_analysis.png'


def generate_complete_results_charts(
    results: List[CycleResult],
    vehicle_groups: Dict[str, VehicleGroup],
    output_dir: str
) -> Dict[str, str]:
    """Generate bar charts from complete results table data."""
    cycles = list(DRIVE_CYCLES.keys())
    chart_files = {}
    
    # Use matplotlib's tab10 color palette (same as generator runtime charts)
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Chart 1: Efficiency Comparison - Separate subplots for each vehicle
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    for idx, (group_key, group) in enumerate(vehicle_groups.items()):
        ax = axes[idx]
        configs = []
        
        for preset in group.presets:
            configs.append({
                'name': get_short_vehicle_name(preset['name']),
                'vehicle_name': preset['name']
            })
        
        x = np.arange(len(cycles))
        width = 0.12
        n_configs = len(configs)
        
        for i, config in enumerate(configs):
            efficiencies = []
            for cycle_key in cycles:
                result = next((r for r in results if r.vehicle_group == group_key and
                              r.vehicle_name == config['vehicle_name'] and r.cycle_name == cycle_key), None)
                if result:
                    efficiencies.append(result.overall_mi_per_kwh)
                else:
                    efficiencies.append(0)
            
            offset = width * (i - n_configs/2)
            ax.bar(x + offset, efficiencies, width, label=config['name'], color=distinct_colors[i], alpha=0.9)
        
        ax.set_xlabel('Drive Cycle', fontsize=11)
        ax.set_ylabel('Overall Efficiency (mi/kWh)', fontsize=11)
        ax.set_title(f'{group.display_name} - Efficiency Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'complete_results_efficiency.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    chart_files['efficiency'] = 'complete_results_efficiency.png'
    
    # Chart 2: Fuel Consumption - Separate subplots for each vehicle
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    for idx, (group_key, group) in enumerate(vehicle_groups.items()):
        ax = axes[idx]
        configs = []
        
        for preset in group.presets:
            configs.append({
                'name': get_short_vehicle_name(preset['name']),
                'vehicle_name': preset['name']
            })
        
        x = np.arange(len(cycles))
        width = 0.12
        n_configs = len(configs)
        
        for i, config in enumerate(configs):
            fuel_used = []
            for cycle_key in cycles:
                result = next((r for r in results if r.vehicle_group == group_key and
                              r.vehicle_name == config['vehicle_name'] and r.cycle_name == cycle_key), None)
                if result:
                    fuel_used.append(result.fuel_gallons)
                else:
                    fuel_used.append(0)
            
            offset = width * (i - n_configs/2)
            ax.bar(x + offset, fuel_used, width, label=config['name'], color=distinct_colors[i], alpha=0.9)
        
        ax.set_xlabel('Drive Cycle', fontsize=11)
        ax.set_ylabel('Fuel Consumed (gallons)', fontsize=11)
        ax.set_title(f'{group.display_name} - Fuel Consumption', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'complete_results_fuel.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    chart_files['fuel'] = 'complete_results_fuel.png'
    
    # Chart 3: Minimum SOC Required - Separate subplots for each vehicle
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    for idx, (group_key, group) in enumerate(vehicle_groups.items()):
        ax = axes[idx]
        configs = []
        
        for preset in group.presets:
            configs.append({
                'name': get_short_vehicle_name(preset['name']),
                'vehicle_name': preset['name']
            })
        
        x = np.arange(len(cycles))
        width = 0.12
        n_configs = len(configs)
        
        for i, config in enumerate(configs):
            min_socs = []
            can_start_at_zero = []
            cycle_failed = []
            
            for cycle_key in cycles:
                result = next((r for r in results if r.vehicle_group == group_key and
                              r.vehicle_name == config['vehicle_name'] and r.cycle_name == cycle_key), None)
                if result and result.cycle_failed:
                    # Cycle cannot be completed - show at 100% with special marker
                    min_socs.append(100)
                    can_start_at_zero.append(False)
                    cycle_failed.append(True)
                elif result and result.min_soc_required < 1.0:
                    # Can start at 0% SOC (generator can handle entire cycle)
                    min_socs.append(result.min_soc_required)
                    can_start_at_zero.append(True)
                    cycle_failed.append(False)
                elif result:
                    # Needs battery assistance
                    min_socs.append(result.min_soc_required)
                    can_start_at_zero.append(False)
                    cycle_failed.append(False)
                else:
                    min_socs.append(0)
                    can_start_at_zero.append(False)
                    cycle_failed.append(False)
            
            offset = width * (i - n_configs/2)
            bars = ax.bar(x + offset, min_socs, width, label=config['name'], 
                         color=distinct_colors[i], alpha=0.9)
            
            # Add hatching for cycles that can start at 0% SOC
            for j, (bar, zero_ok, failed) in enumerate(zip(bars, can_start_at_zero, cycle_failed)):
                if failed:
                    bar.set_hatch('///')
                    bar.set_edgecolor('black')
                    bar.set_alpha(0.4)
                elif zero_ok:
                    bar.set_hatch('...')
                    bar.set_edgecolor('green')
                    bar.set_linewidth(2)
        
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
        
        # Add legend entries for special cases
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor='gray', edgecolor='green', hatch='...', linewidth=2, label='Can start at 0% SOC'))
        handles.append(Patch(facecolor='gray', edgecolor='black', hatch='///', alpha=0.4, label='Cannot complete'))
        
        ax.set_xlabel('Drive Cycle', fontsize=11)
        ax.set_ylabel('Minimum Starting SOC Required (%)', fontsize=11)
        ax.set_title(f'{group.display_name} - Battery Capacity Required', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([DRIVE_CYCLES[c]['display_name'] for c in cycles])
        ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'complete_results_min_soc.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    chart_files['min_soc'] = 'complete_results_min_soc.png'
    
    return chart_files


# =============================================================================
# HTML Report Generation
# =============================================================================
def generate_html_report(
    vehicle_groups: Dict[str, VehicleGroup],
    cycle_results: List[CycleResult],
    range_results: List[RangeResult],
    speed_results: List[GeneratorSpeedResult],
    output_dir: str
) -> str:
    """Generate comprehensive HTML report with cross-vehicle comparisons."""
    
    cycles = list(DRIVE_CYCLES.keys())
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified EREV Generator Comparison Study</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1600px;
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
            margin-top: 30px;
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
        .comparison-box {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .comparison-box h3 {{
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
        .ramcharger-header {{
            background: #C41E3A !important;
        }}
        .scout-header {{
            background: #2C5F2D !important;
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
        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1200px) {{
            .grid-2col {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <h1>⚡ Unified EREV Generator Comparison Study</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary-box">
        <h3>Study Overview</h3>
        <p>Direct comparison of EREV platforms across demanding drive cycles and towing configurations.</p>
        <ul>
"""
    
    for group_key, group in vehicle_groups.items():
        html += f"""            <li><strong>{group.display_name}:</strong> {int(group.generator_kw)} kW generator, {len(group.presets)} configurations tested</li>
"""
    
    html += f"""        </ul>
        <ul>
            <li><strong>Drive Cycles:</strong> {len(cycles)} cycles (Davis Dam, I-70 Climb, Highway 75 MPH, US06)</li>
            <li><strong>Range Tests:</strong> Multi-cycle EPA-style + Highway 75 MPH steady-state</li>
            <li><strong>Starting SOC:</strong> 100%</li>
        </ul>
    </div>
    
    <h2>🏆 Head-to-Head Comparison: Base Vehicles</h2>
    
    <div class="comparison-box">
        <h3>Key Differences</h3>
        <div class="grid-2col">
"""
    
    for group_key, group in vehicle_groups.items():
        if group.presets:
            base = group.presets[0]
            html += f"""            <div>
                <h4>{group.display_name}</h4>
                <ul>
                    <li>Generator: {int(group.generator_kw)} kW</li>
                    <li>Battery: {base['battery']:.1f} kWh</li>
                    <li>Mass: {base['mass']:.0f} kg ({base['mass']*2.20462:.0f} lb)</li>
                    <li>Drag: Cd {base['cd']:.3f}, Area {base['frontal_area']:.2f} m²</li>
                    <li>Fuel Tank: {base['fuel_tank_gallons']:.1f} gal</li>
                </ul>
            </div>
"""
    
    html += """        </div>
    </div>
    
    <h2>🛣️ Drive Cycle Comparison</h2>
    <div class="section">
        <p>Characteristics of the drive cycles used in this study:</p>
        <table>
            <tr>
                <th>Drive Cycle</th>
                <th>Distance (mi)</th>
                <th>Duration (min)</th>
                <th>Avg Speed (mph)</th>
                <th>Max Speed (mph)</th>
                <th>Avg Grade (%)</th>
                <th>Max Grade (%)</th>
                <th>Min Grade (%)</th>
                <th>Max Accel (m/s²)</th>
                <th>Idle Time (%)</th>
            </tr>
"""
    
    # Generate drive cycle stats
    cycle_stats = generate_drive_cycle_comparison_table()
    
    for cycle_key in cycles:
        stats = cycle_stats[cycle_key]
        display_name = DRIVE_CYCLES[cycle_key]['display_name']
        html += f"""            <tr>
                <td class="vehicle-name">{display_name}</td>
                <td>{stats['distance_miles']:.1f}</td>
                <td>{stats['duration_min']:.1f}</td>
                <td>{stats['avg_speed_mph']:.1f}</td>
                <td>{stats['max_speed_mph']:.1f}</td>
                <td>{stats['avg_grade_pct']:.2f}</td>
                <td>{stats['max_grade_pct']:.1f}</td>
                <td>{stats['min_grade_pct']:.1f}</td>
                <td>{stats['max_accel_mps2']:.2f}</td>
                <td>{stats['idle_time_pct']:.1f}</td>
            </tr>
"""
    
    html += """        </table>
    </div>
    
    <h2>🔧 Generator Sizing Analysis</h2>
    <div class="section">
        <p>Comprehensive analysis of generator capacity adequacy across all test scenarios:</p>
    </div>
    
    <div class="chart-container">
        <img src="generator_sizing_analysis.png" alt="Generator Sizing Analysis">
    </div>
    
    <div class="chart-container">
        <h3>Base Vehicle Efficiency Comparison</h3>
        <img src="cross_vehicle_mpge.png" alt="Cross-Vehicle MPGe">
    </div>
    
    <div class="chart-container">
        <h3>Base Vehicle Range Comparison</h3>
        <img src="cross_vehicle_range.png" alt="Cross-Vehicle Range">
    </div>
    
    <div class="chart-container">
        <h3>Power Demand vs Generator Capacity</h3>
        <img src="cross_vehicle_power.png" alt="Cross-Vehicle Power">
    </div>
    
    <div class="chart-container">
        <h3>Maximum Generator-Sustained Speed (All Configurations)</h3>
        <img src="cross_vehicle_max_speed.png" alt="Cross-Vehicle Max Speed">
    </div>
    
    <div class="chart-container">
        <h3>Base Vehicle Generator Utilization</h3>
        <img src="cross_vehicle_generator_utilization.png" alt="Cross-Vehicle Generator">
    </div>
    
    <h2>📊 Detailed Results by Vehicle</h2>
"""
    
    # Generate detailed sections for each vehicle group
    for group_key, group in vehicle_groups.items():
        html += f"""
    <h3>{group.display_name} - Detailed Analysis</h3>
    
    <div class="section">
        <h4>Vehicle Configurations</h4>
        <table>
            <tr>
                <th>Configuration</th>
                <th>Mass (kg)</th>
                <th>Mass (lb)</th>
                <th>Cd</th>
                <th>Frontal Area (m²)</th>
                <th>Generator (kW)</th>
            </tr>
"""
        
        for preset in group.presets:
            mass_kg = preset['mass']
            mass_lb = mass_kg * 2.20462
            config_name = get_short_vehicle_name(preset['name'])
            
            html += f"""            <tr>
                <td class="vehicle-name">{config_name}</td>
                <td>{mass_kg:.0f}</td>
                <td>{mass_lb:.0f}</td>
                <td>{preset['cd']:.3f}</td>
                <td>{preset['frontal_area']:.2f}</td>
                <td>{preset['generator_power_kw']:.0f}</td>
            </tr>
"""
        
        html += """        </table>
    </div>
    
    <div class="chart-container">
        <img src="{group_key}_mpge_comparison.png" alt="{group_name} MPGe">
    </div>
    
    <div class="chart-container">
        <img src="{group_key}_power_demand.png" alt="{group_name} Power">
    </div>
    
    <div class="chart-container">
        <img src="{group_key}_range.png" alt="{group_name} Range">
    </div>
    
    <div class="chart-container">
        <img src="{group_key}_soc_profiles.png" alt="{group_name} SOC">
    </div>
    
    <div class="chart-container">
        <img src="{group_key}_generator_utilization.png" alt="{group_name} Generator">
    </div>
""".format(group_key=group_key, group_name=group.display_name)
    
    # Add comprehensive data table
    html += """
    <h2>📋 Complete Results Table</h2>
    <div class="section">
        <p><strong>Note:</strong> Cells highlighted in <span class="deficit">red</span> indicate power demand exceeds generator capacity or cycle cannot be completed.</p>
        <table>
            <tr>
                <th rowspan="2">Configuration</th>
"""
    
    for cycle_key in cycles:
        html += f"""                <th colspan="5">{DRIVE_CYCLES[cycle_key]['display_name']}</th>
"""
    
    html += """            </tr>
            <tr>
"""
    
    for _ in cycles:
        html += """                <th>mi/kWh</th>
                <th>Peak kW</th>
                <th>Avg kW</th>
                <th>Fuel (gal)</th>
                <th>Min SOC (%)</th>
"""
    
    html += """            </tr>
"""
    
    # Add data rows
    for group_key, group in vehicle_groups.items():
        generator_capacity = group.generator_kw
        for preset in group.presets:
            vehicle_name = preset['name']
            short_name = get_short_vehicle_name(vehicle_name)
            
            html += f"""            <tr>
                <td class="vehicle-name">{short_name}</td>
"""
            
            for cycle_key in cycles:
                result = next((r for r in cycle_results if r.vehicle_group == group_key and
                              r.vehicle_name == vehicle_name and r.cycle_name == cycle_key), None)
                
                if result:
                    # Check if peak or avg exceeds generator capacity
                    peak_exceeds = result.peak_power_kw > generator_capacity
                    avg_exceeds = result.avg_power_kw > generator_capacity
                    cannot_complete = result.cycle_failed
                    
                    # Format Min SOC cell
                    if cannot_complete:
                        min_soc_cell = '<td class="deficit">N/A</td>'
                    else:
                        min_soc_class = 'deficit' if result.min_soc_required > 80 else ''
                        min_soc_cell = f'<td class="{min_soc_class}">{result.min_soc_required:.1f}</td>'
                    
                    html += f"""                <td>{result.overall_mi_per_kwh:.2f}</td>
                <td class="{'deficit' if peak_exceeds else ''}">{result.peak_power_kw:.1f}</td>
                <td class="{'deficit' if avg_exceeds else ''}">{result.avg_power_kw:.1f}</td>
                <td>{result.fuel_gallons:.2f}</td>
                {min_soc_cell}
"""
                else:
                    html += """                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
"""
            
            html += """            </tr>
"""
    
    html += """        </table>
    </div>
    
    <h2>📊 Complete Results Visualizations</h2>
    <div class="section">
        <p>Comprehensive bar charts comparing all configurations across drive cycles.</p>
        
        <div class="chart-container">
            <img src="complete_results_efficiency.png" alt="Efficiency Comparison">
        </div>
        
        <div class="chart-container">
            <img src="complete_results_fuel.png" alt="Fuel Consumption Comparison">
        </div>
        
        <div class="chart-container">
            <img src="complete_results_min_soc.png" alt="Minimum SOC Required">
        </div>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #95a5a6; text-align: center;">
        <p>Unified EREV Generator Comparison Study • Road Load Simulator</p>
    </footer>
</body>
</html>
"""
    
    # Write report
    report_path = os.path.join(output_dir, 'unified_report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run the unified EREV comparison study."""
    parser = argparse.ArgumentParser(
        description='Unified EREV Generator Sizing Comparison Study'
    )
    parser.add_argument(
        '--vehicles',
        nargs='+',
        choices=['ramcharger', 'scout', 'both'],
        default=['both'],
        help='Which vehicles to include in the study'
    )
    
    args = parser.parse_args()
    
    # Determine which vehicles to include
    include_ramcharger = 'ramcharger' in args.vehicles or 'both' in args.vehicles
    include_scout = 'scout' in args.vehicles or 'both' in args.vehicles
    
    print("=" * 70)
    print("Unified EREV Generator Comparison Study")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, STUDY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load vehicle presets
    print("\nLoading vehicle presets...")
    vehicle_groups = load_vehicle_presets(
        include_ramcharger=include_ramcharger,
        include_scout=include_scout
    )
    
    if not vehicle_groups:
        print("ERROR: No vehicles loaded!")
        return
    
    for group_key, group in vehicle_groups.items():
        print(f"  {group.display_name}: {len(group.presets)} configurations, {int(group.generator_kw)} kW generator")
    
    # ==========================================================================
    # Run single-cycle simulations
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Running single-cycle simulations (100% SOC)...")
    print("-" * 70)
    
    cycle_results = []
    total_sims = sum(len(g.presets) for g in vehicle_groups.values()) * len(DRIVE_CYCLES)
    current_sim = 0
    
    for group_key, group in vehicle_groups.items():
        for preset in group.presets:
            vehicle = preset_to_vehicle_params(preset)
            vehicle_name = preset['name']
            
            for cycle_key, cycle_info in DRIVE_CYCLES.items():
                current_sim += 1
                print(f"  [{current_sim}/{total_sims}] {get_short_vehicle_name(vehicle_name)} on {cycle_info['display_name']}...")
                
                result = run_single_cycle_simulation(
                    vehicle=vehicle,
                    vehicle_group=group_key,
                    vehicle_name=vehicle_name,
                    cycle_key=cycle_key,
                    cycle_info=cycle_info,
                    starting_soc=STARTING_SOC_100
                )
                cycle_results.append(result)
    
    # ==========================================================================
    # Run multi-cycle range tests
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Running multi-cycle range tests...")
    print("-" * 70)
    
    range_results = []
    
    for group_key, group in vehicle_groups.items():
        for preset in group.presets:
            vehicle = preset_to_vehicle_params(preset)
            vehicle_name = preset['name']
            
            print(f"  {get_short_vehicle_name(vehicle_name)}...")
            
            result = run_multi_cycle_range_test(
                vehicle=vehicle,
                vehicle_group=group_key,
                vehicle_name=vehicle_name,
                starting_soc=STARTING_SOC_100
            )
            range_results.append(result)
    
    # ==========================================================================
    # Calculate maximum generator-sustained speeds
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Calculating maximum generator-sustained speeds...")
    print("-" * 70)
    
    speed_results = []
    
    for group_key, group in vehicle_groups.items():
        for preset in group.presets:
            vehicle = preset_to_vehicle_params(preset)
            vehicle_name = preset['name']
            
            result = calculate_max_generator_speed(
                vehicle=vehicle,
                vehicle_group=group_key,
                vehicle_name=vehicle_name,
                generator_power_kw=group.generator_kw
            )
            speed_results.append(result)
            
            print(f"  {get_short_vehicle_name(vehicle_name)}: {result.max_speed_mph:.1f} mph "
                  f"({result.power_at_75mph_kw:.1f} kW @ 75mph)")
    
    # ==========================================================================
    # Generate charts
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Generating charts...")
    print("-" * 70)
    
    # Cross-vehicle comparison charts
    print("  Cross-vehicle comparisons...")
    generate_cross_vehicle_mpge_chart(cycle_results, vehicle_groups, output_dir)
    generate_cross_vehicle_range_chart(range_results, vehicle_groups, output_dir)
    generate_cross_vehicle_power_chart(cycle_results, vehicle_groups, output_dir)
    generate_max_speed_comparison_chart(speed_results, vehicle_groups, output_dir)
    generate_generator_utilization_chart(cycle_results, vehicle_groups, output_dir)
    generate_generator_sizing_analysis_chart(cycle_results, vehicle_groups, output_dir)
    
    print("  Complete results charts...")
    complete_results_charts = generate_complete_results_charts(cycle_results, vehicle_groups, output_dir)
    
    # Individual vehicle group charts
    for group_key, group in vehicle_groups.items():
        print(f"  {group.display_name} detailed charts...")
        generate_efficiency_bar_chart(
            cycle_results, group_key, output_dir, 'mpge',
            f'{group.display_name} - MPGe Comparison', 'MPGe'
        )
        generate_power_demand_chart(
            cycle_results, group_key, group.generator_kw, output_dir
        )
        generate_range_comparison_chart(range_results, group_key, output_dir)
        generate_individual_soc_charts(cycle_results, group_key, output_dir)
        generate_individual_generator_utilization(
            cycle_results, group_key, group.generator_kw, output_dir
        )
    
    # ==========================================================================
    # Generate HTML report
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Generating HTML report...")
    print("-" * 70)
    
    report_path = generate_html_report(
        vehicle_groups=vehicle_groups,
        cycle_results=cycle_results,
        range_results=range_results,
        speed_results=speed_results,
        output_dir=output_dir
    )
    
    print(f"\n✅ Study complete!")
    print(f"📄 Report: {report_path}")
    print(f"📁 All outputs: {output_dir}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    for group_key, group in vehicle_groups.items():
        print(f"\n{group.display_name}:")
        
        # Base vehicle stats
        base_name = group.presets[0]['name']
        base_results = [r for r in cycle_results if r.vehicle_group == group_key and 
                       r.vehicle_name == base_name]
        
        if base_results:
            avg_mpge = np.mean([r.mpge for r in base_results])
            avg_peak = np.mean([r.peak_power_kw for r in base_results])
            print(f"  Base vehicle average: {avg_mpge:.1f} MPGe, {avg_peak:.1f} kW peak power")
        
        # Range
        base_range = next((r for r in range_results if r.vehicle_group == group_key and
                          r.vehicle_name == base_name and r.test_type == 'multi_cycle'), None)
        if base_range:
            print(f"  Multi-cycle range: {base_range.total_range_miles:.0f} mi "
                  f"({base_range.ev_range_miles:.0f} EV + {base_range.generator_range_miles:.0f} gen)")
        
        # Max speed
        base_speed = next((r for r in speed_results if r.vehicle_group == group_key and
                          r.vehicle_name == base_name), None)
        if base_speed:
            print(f"  Max generator speed: {base_speed.max_speed_mph:.1f} mph "
                  f"({'✅' if base_speed.can_sustain_75mph else '⚠️'} @ 75mph)")


if __name__ == '__main__':
    main()
