#!/usr/bin/env python3
"""
FDMNES Configuration Space Explorer

Automates exploration of FDMNES calculation parameters by:
1. Parsing TOML configuration with ITERATE keywords
2. Generating parameter combinations
3. Running main calculations and convolutions
4. Organizing outputs and generating plots
"""

import os
import sys
import toml
import shutil
import subprocess
import itertools
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple


def parse_toml_config(config_path: str) -> Dict[str, Any]:
    """
    Parse TOML configuration file and extract ITERATE parameters
    
    Returns:
        Dictionary with structure path, main/conv parameters, and templates
    """
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    result = {
        "structure": config.get("structure", ""),
        "main_params": {},
        "conv_params": {},
        "main_template": "",
        "conv_template": "",
        "run_main": False,
        "run_conv": False,
        "existing_main_dir": config.get("existing_main_dir", None),
        "existing_main_dirs": config.get("existing_main_dirs", None),
        "main_dir_root": config.get("main_dir_root", None),
        "plot_options": config.get("plot_options", {
            "individual": True,
            "combined": False
        }),
        "fdmnes_executable": config.get("fdmnes_executable", 
                                       "~/FDMNES/parallel_fdmnes/mpirun_fdmnes"),
        "num_cores": config.get("num_cores", 20),
        "output_dir": config.get("output_dir", "exploration_runs")
    }
    
    # Parse main calculation parameters
    if "fdmnes_input" in config:
        result["run_main"] = True
        main_text = config["fdmnes_input"]
        result["main_template"] = main_text
        result["main_params"] = parse_iterate_params(main_text)
    
    # Parse convolution parameters
    if "fdmnes_conv" in config:
        result["run_conv"] = True
        conv_text = config["fdmnes_conv"]
        result["conv_template"] = conv_text
        result["conv_params"] = parse_iterate_params(conv_text)
    
    return result


def parse_iterate_params(text: str) -> Dict[str, List]:
    """
    Extract parameters marked with ITERATE from input text
    
    Returns:
        Dictionary mapping parameter names to their possible values
    """
    params = {}
    lines = text.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if line contains ITERATE
        if "ITERATE" in line:
            # Extract card name (first word)
            card_name = line.split()[0]
            
            # Check next line for values
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                # Check for discrete values [...]
                if next_line.startswith('[') and next_line.endswith(']'):
                    values_str = next_line[1:-1]
                    values = [float(v.strip()) for v in values_str.split(',')]
                    params[card_name] = values
                    i += 2
                    continue
                
                # Check for range[start, end, step]
                elif next_line.startswith('range[') and next_line.endswith(']'):
                    range_str = next_line[6:-1]  # Remove 'range[' and ']'
                    parts = [float(p.strip()) for p in range_str.split(',')]
                    if len(parts) == 3:
                        start, end, step = parts
                        values = list(np.arange(start, end + step/2, step))
                        params[card_name] = values
                        i += 2
                        continue
            
            # No values specified - boolean card (with/without)
            params[card_name] = [True, False]
        
        i += 1
    
    return params


def generate_parameter_combinations(params: Dict[str, List]) -> List[Dict]:
    """
    Generate all combinations of parameters
    
    Returns:
        List of dictionaries with parameter values and run names
    """
    if not params:
        return [{"values": {}, "name": "baseline"}]
    
    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]
    
    combinations = []
    for i, combo in enumerate(itertools.product(*param_values)):
        values = dict(zip(param_names, combo))
        
        # Generate descriptive name
        name_parts = [f"run_{i:03d}"]
        for param, value in values.items():
            if isinstance(value, bool):
                name_parts.append(f"{param}_{'on' if value else 'off'}")
            else:
                # Format number cleanly
                if isinstance(value, float) and value == int(value):
                    name_parts.append(f"{param}_{int(value)}")
                else:
                    name_parts.append(f"{param}_{value:.2f}".replace('.', 'p'))
        
        combinations.append({
            "values": values,
            "name": "_".join(name_parts)
        })
    
    return combinations


def generate_fdmnes_input(template: str, params: Dict[str, Any]) -> str:
    """
    Generate FDMNES input file from template and parameters
    
    Returns:
        Modified input text with parameters substituted
    """
    result_lines = []
    lines = template.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check if line contains ITERATE
        if "ITERATE" in line:
            # Extract card name
            card_name = stripped.split()[0]
            
            if card_name in params:
                value = params[card_name]
                
                # Handle boolean cards
                if isinstance(value, bool):
                    if value:
                        # Keep the card (remove ITERATE)
                        result_lines.append(card_name)
                    else:
                        # Comment out the card
                        result_lines.append(f"! {card_name}")
                    
                    # Skip next line if it contains parameter value
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if (next_line.startswith('[') or 
                            next_line.startswith('range[')):
                            i += 1
                
                else:
                    # Numeric value - replace with actual value
                    result_lines.append(card_name)
                    result_lines.append(str(value))
                    
                    # Skip next line if it contains the placeholder
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if (next_line.startswith('[') or 
                            next_line.startswith('range[')):
                            i += 1
            else:
                # Keep line as-is if not in params
                result_lines.append(line)
        else:
            # Keep non-ITERATE lines as-is
            result_lines.append(line)
        
        i += 1
    
    return '\n'.join(result_lines)


def create_working_directory(base_dir: Path, run_name: str, structure_file: str) -> Path:
    """
    Create working directory for a run with necessary files
    
    Returns:
        Path to the working directory
    """
    work_dir = base_dir / run_name / "working"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy structure file if it exists
    if structure_file and os.path.exists(structure_file):
        shutil.copy2(structure_file, work_dir)
    
    return work_dir


def write_fdmfile(work_dir: Path, input_file: Path, conv_file: Path = None):
    """
    Write fdmfile.txt with list of calculations to run
    """
    fdmfile = work_dir / "fdmfile.txt"
    
    if conv_file:
        lines = ["2"]  # Two calculations
        lines.append(str(input_file.absolute()))
        lines.append(str(conv_file.absolute()))
    else:
        lines = ["1"]  # One calculation
        lines.append(str(input_file.absolute()))
    
    fdmfile.write_text('\n'.join(lines) + '\n')


def run_fdmnes(work_dir: Path, num_cores: int = 20, 
               fdmnes_executable: str = "~/FDMNES/parallel_fdmnes/mpirun_fdmnes") -> bool:
    """
    Execute FDMNES calculation in the working directory
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        os.path.expanduser(fdmnes_executable),
        "-n", str(num_cores)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Error running FDMNES in {work_dir}:")
            print(result.stderr)
            return False
        
        return True
    
    except subprocess.TimeoutExpired:
        print(f"FDMNES calculation timed out in {work_dir}")
        return False
    except Exception as e:
        print(f"Error running FDMNES: {e}")
        return False


def plot_stick_spectrum(data_file: str, output_file: str, title: str = ""):
    """
    Create PNG plot from stick spectrum data file
    """
    try:
        # Read file and detect data start
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Find where numerical data starts
        data_start = 0
        for i, line in enumerate(lines):
            # Skip lines that start with headers or contain '='
            if '=' in line or 'Energy' in line.strip() or 'FDMNES' in line:
                data_start = i + 1
                continue
            # Try to parse as numbers
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    float(parts[0])
                    float(parts[1])
                    # Found the start of data
                    data_start = i
                    break
            except (ValueError, IndexError):
                data_start = i + 1
        
        # Read data from the detected start position
        data = np.loadtxt(data_file, skiprows=data_start)
        
        if len(data) == 0:
            print(f"No data found in {data_file}")
            return False
        
        if len(data.shape) == 1:
            # Single column - just intensities
            x = np.arange(len(data))
            y = data
        else:
            # Multiple columns - assume first is energy, second is intensity
            x = data[:, 0]
            y = data[:, 1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=1.5)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity')
        plt.title(title if title else 'Stick Spectrum')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save as PNG
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error plotting {data_file}: {e}")
        return False


def plot_conv_spectrum(data_file: str, output_file: str, title: str = ""):
    """
    Create PNG plot from convolution spectrum data file
    """
    # Same as stick spectrum but different default title
    return plot_stick_spectrum(
        data_file, 
        output_file, 
        title if title else "Convolution Spectrum"
    )


def create_combined_plot(data_files: List[Tuple[str, str]], output_file: str, title: str = "Combined Spectra"):
    """
    Create a combined plot with multiple spectra overlaid
    
    Args:
        data_files: List of tuples (file_path, label) for each spectrum
        output_file: Output PNG file path
        title: Plot title
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Use a colormap for different line colors
        colors = plt.cm.get_cmap('tab10')
        
        for i, (data_file, label) in enumerate(data_files):
            # Read file and detect data start
            with open(data_file, 'r') as f:
                lines = f.readlines()
            
            # Find where numerical data starts
            data_start = 0
            for j, line in enumerate(lines):
                if '=' in line or 'Energy' in line.strip() or 'FDMNES' in line:
                    data_start = j + 1
                    continue
                try:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        float(parts[0])
                        float(parts[1])
                        data_start = j
                        break
                except (ValueError, IndexError):
                    data_start = j + 1
            
            # Read data
            data = np.loadtxt(data_file, skiprows=data_start)
            
            if len(data) == 0:
                print(f"No data found in {data_file}")
                continue
            
            if len(data.shape) == 1:
                x = np.arange(len(data))
                y = data
            else:
                x = data[:, 0]
                y = data[:, 1]
            
            # Plot with label and color
            plt.plot(x, y, label=label, linewidth=1.5, color=colors(i % 10), alpha=0.8)
        
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity')
        plt.title(title)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save as PNG
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error creating combined plot: {e}")
        return False


def organize_outputs(run_dir: Path, run_name: str, plot_options: Dict[str, bool] = None):
    """
    Organize output files and generate plots
    """
    # Default plot options
    if plot_options is None:
        plot_options = {"individual": True, "combined": False}
    
    # Create output directories
    output_txt_dir = run_dir / "output_txt"
    output_conv_dir = run_dir / "output_conv"
    
    # Only create plot directories if we're making individual plots
    if plot_options.get("individual", True):
        plot_stick_dir = run_dir / "plots" / "stick"
        plot_conv_dir = run_dir / "plots" / "conv"
        plot_stick_dir.mkdir(parents=True, exist_ok=True)
        plot_conv_dir.mkdir(parents=True, exist_ok=True)
    
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    output_conv_dir.mkdir(parents=True, exist_ok=True)
    
    # Move and plot stick outputs
    work_dir = run_dir / "working"
    for txt_file in work_dir.glob("*.txt"):
        # Skip fdmfile, error files, bav files, and convolution files
        if (txt_file.name == "fdmfile.txt" or 
            "error" in txt_file.name.lower() or
            "bav" in txt_file.name.lower() or
            "conv" in txt_file.name.lower()):
            continue
            
        # This is likely a stick output
        shutil.copy2(txt_file, output_txt_dir)
        
        # Generate plot if individual plots are requested
        if plot_options.get("individual", True):
            plot_file = plot_stick_dir / f"{txt_file.stem}.png"
            plot_stick_spectrum(str(txt_file), str(plot_file), f"{run_name} - {txt_file.stem}")
    
    # Also check the actual output_txt directory for outputs
    for txt_file in output_txt_dir.glob("*.txt"):
        # Skip error, bav, and other auxiliary files
        if ("error" in txt_file.name.lower() or 
            "bav" in txt_file.name.lower() or
            "_bav" in txt_file.name):
            continue
        if plot_options.get("individual", True):
            plot_file = plot_stick_dir / f"{txt_file.stem}.png"
            if not plot_file.exists():
                plot_stick_spectrum(str(txt_file), str(plot_file), f"{run_name} - {txt_file.stem}")
    
    # Move and plot convolution outputs
    # First check for files with "conv" in the name in working directory
    for conv_file in work_dir.glob("*conv*.txt"):
        if "error" not in conv_file.name.lower():
            shutil.copy2(conv_file, output_conv_dir)
            
            # Generate plot if individual plots are requested
            if plot_options.get("individual", True):
                plot_file = plot_conv_dir / f"{conv_file.stem}.png"
                plot_conv_spectrum(str(conv_file), str(plot_file), f"{run_name} - {conv_file.stem}")
    
    # For convolution runs, the output_conv directory will have the convolution results
    # These may not have "conv" in the filename
    if output_conv_dir.exists():
        for conv_file in output_conv_dir.glob("*.txt"):
            if "error" not in conv_file.name.lower() and "bav" not in conv_file.name.lower():
                if plot_options.get("individual", True):
                    plot_file = plot_conv_dir / f"{conv_file.stem}.png"
                    if not plot_file.exists():
                        plot_conv_spectrum(str(conv_file), str(plot_file), f"{run_name} - {conv_file.stem}")


def generate_summary_csv(base_dir: Path, all_runs: List[Dict]):
    """
    Generate CSV summary of all runs with parameters and output locations
    """
    csv_file = base_dir / "exploration_summary.csv"
    
    # Collect all parameter names
    all_params = set()
    for run in all_runs:
        all_params.update(run["values"].keys())
    
    param_names = sorted(all_params)
    
    # Write CSV
    with open(csv_file, 'w') as f:
        # Header
        headers = ["run_name"] + param_names + ["output_dir", "status"]
        f.write(','.join(headers) + '\n')
        
        # Data rows
        for run in all_runs:
            row = [run["name"]]
            
            # Parameter values
            for param in param_names:
                value = run["values"].get(param, "")
                if isinstance(value, bool):
                    value = "on" if value else "off"
                row.append(str(value))
            
            # Output location and status
            row.append(str(run.get("output_dir", "")))
            row.append(run.get("status", "pending"))
            
            f.write(','.join(row) + '\n')
    
    print(f"Summary written to {csv_file}")


def main():
    """
    Main execution function
    """
    if len(sys.argv) != 2:
        print("Usage: python fdmnes_explorer.py explore.toml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found")
        sys.exit(1)
    
    print(f"Parsing configuration from {config_file}...")
    config = parse_toml_config(config_file)
    
    # Create base output directory
    base_dir = Path(config["output_dir"])
    
    all_runs = []
    
    # Check if we're using existing main calculations
    if config["existing_main_dir"] or config["existing_main_dirs"] or config["main_dir_root"]:
        print("\nUsing existing main calculations...")
        
        # Collect existing main directories
        existing_dirs = []
        
        if config["main_dir_root"]:
            # Root directory containing main calculation subdirectories
            root_path = Path(config["main_dir_root"])
            if root_path.exists() and root_path.is_dir():
                # Find all subdirectories that contain output_txt (indicating a main calculation)
                for subdir in sorted(root_path.iterdir()):
                    if subdir.is_dir() and (subdir / "output_txt").exists():
                        existing_dirs.append(subdir)
                        print(f"  Found main calculation: {subdir.name}")
                
                if not existing_dirs:
                    print(f"Warning: No main calculation directories found in {root_path}")
            else:
                print(f"Warning: {root_path} does not exist or is not a directory")
        
        if config["existing_main_dir"]:
            # Single directory specified
            main_path = Path(config["existing_main_dir"])
            if main_path.exists():
                existing_dirs.append(main_path)
            else:
                print(f"Warning: {main_path} does not exist")
        
        if config["existing_main_dirs"]:
            # Multiple directories specified
            for dir_path in config["existing_main_dirs"]:
                main_path = Path(dir_path)
                if main_path.exists():
                    existing_dirs.append(main_path)
                else:
                    print(f"Warning: {main_path} does not exist")
        
        # Add existing directories to all_runs
        for main_dir in existing_dirs:
            all_runs.append({
                "name": main_dir.name,
                "values": {},  # No parameter values since these are pre-existing
                "output_dir": str(main_dir),
                "status": "completed"
            })
        
        print(f"Found {len(all_runs)} existing main calculation directories")
    
    # Generate and run main calculations
    elif config["run_main"]:
        print("\nGenerating main calculation combinations...")
        main_combos = generate_parameter_combinations(config["main_params"])
        print(f"Found {len(main_combos)} main calculation combinations")
        
        for combo in main_combos:
            print(f"\nRunning: {combo['name']}")
            
            # Create working directory
            run_dir = base_dir / combo["name"]
            work_dir = create_working_directory(run_dir, "", config["structure"])
            
            # Generate input file
            input_text = generate_fdmnes_input(config["main_template"], combo["values"])
            
            # Create output directory first
            output_dir = run_dir / 'output_txt'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Add Filout directive with absolute path
            output_name = (output_dir / combo['name']).absolute()
            input_text = f"Filout\n{output_name}\n\n" + input_text
            
            # Add structure file reference with absolute path
            if config["structure"]:
                cif_name = Path(config["structure"]).name
                input_text += f"\n\nCif_file\n{(work_dir / cif_name).absolute()}\n"
            
            input_text += "\nEnd\n"
            
            # Write input file
            input_file = run_dir / "input_file" / f"{combo['name']}.txt"
            input_file.parent.mkdir(parents=True, exist_ok=True)
            input_file.write_text(input_text)
            
            # Write fdmfile
            write_fdmfile(work_dir, input_file)
            
            # Run FDMNES
            success = run_fdmnes(work_dir, config["num_cores"], config["fdmnes_executable"])
            
            # Organize outputs
            if success:
                organize_outputs(run_dir, combo["name"], config["plot_options"])
                combo["status"] = "completed"
            else:
                combo["status"] = "failed"
            
            combo["output_dir"] = str(run_dir)
            all_runs.append(combo)
    
    # Generate and run convolution calculations
    if config["run_conv"]:
        print("\nGenerating convolution combinations...")
        conv_combos = generate_parameter_combinations(config["conv_params"])
        print(f"Found {len(conv_combos)} convolution combinations")
        
        # For each main calculation, run all convolution combinations
        for main_run in all_runs:
            if main_run.get("status") != "completed":
                continue
            
            main_output = Path(main_run["output_dir"]) / "output_txt"
            # Filter out bav and error files - only get the actual calculation output
            main_files = [f for f in main_output.glob("*.txt") 
                         if "bav" not in f.name.lower() and 
                            "_bav" not in f.name and
                            "error" not in f.name.lower()]
            
            if not main_files:
                print(f"No output files found for {main_run['name']}")
                continue
            
            for conv_combo in conv_combos:
                conv_name = f"conv_{conv_combo['name']}"
                print(f"\nRunning convolution: {main_run['name']} - {conv_name}")
                
                # Use the existing main calculation directory
                run_dir = Path(main_run["output_dir"])
                work_dir = run_dir / "working"
                
                # Generate convolution input
                conv_text = generate_fdmnes_input(config["conv_template"], conv_combo["values"])
                
                # Create output directory if it doesn't exist
                conv_output_dir = run_dir / 'output_conv'
                conv_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Add Calculation and Conv_out directives with absolute paths
                conv_text = f"Calculation\n{main_files[0].absolute()}\n\n" + \
                           f"Conv_out\n{(conv_output_dir / conv_name).absolute()}\n\n" + \
                           "Convolution\n\n" + conv_text + "\n\nEnd\n"
                
                # Write convolution input in input_conv directory
                conv_file = run_dir / "input_conv" / f"{conv_name}.txt"
                conv_file.parent.mkdir(parents=True, exist_ok=True)
                conv_file.write_text(conv_text)
                
                # Write fdmfile for convolution only
                write_fdmfile(work_dir, conv_file)
                
                # Run FDMNES
                success = run_fdmnes(work_dir, config["num_cores"], config["fdmnes_executable"])
                
                # Organize outputs (this will add to existing plots/conv directory)
                if success:
                    organize_outputs(run_dir, f"{main_run['name']}_{conv_name}", config["plot_options"])
                    status = "completed"
                else:
                    status = "failed"
            
            # After running all convolutions for this main run, create combined plot if requested
            if config["plot_options"].get("combined", False):
                print(f"\nCreating combined convolution plot for {main_run['name']}...")
                
                run_dir = Path(main_run["output_dir"])
                conv_output_dir = run_dir / 'output_conv'
                plot_dir = run_dir / 'plots'
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Collect all convolution output files
                conv_files = []
                for conv_file in sorted(conv_output_dir.glob("conv_*.txt")):
                    if "error" not in conv_file.name.lower() and "bav" not in conv_file.name.lower():
                        # Extract parameter info from filename for label
                        label = conv_file.stem.replace("conv_", "")
                        conv_files.append((str(conv_file), label))
                
                if conv_files:
                    combined_plot_file = plot_dir / "combined_convolutions.png"
                    create_combined_plot(
                        conv_files, 
                        str(combined_plot_file),
                        f"{main_run['name']} - All Convolutions"
                    )
                    print(f"Combined plot saved to {combined_plot_file}")
    
    # Generate summary
    generate_summary_csv(base_dir, all_runs)
    
    print("\nExploration complete!")
    print(f"Results saved in {base_dir}")


if __name__ == "__main__":
    main()