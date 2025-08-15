# FDMNES Explorer

A Python tool for automated exploration of FDMNES calculation parameter space. This tool enables systematic parameter variation for both main calculations and convolutions, with automatic output organization and visualization.

## Features

- **Parameter Space Exploration**: Systematically vary FDMNES parameters using discrete values, ranges, or boolean flags
- **Automated Execution**: Run multiple FDMNES calculations in parallel with MPI support
- **Smart Organization**: Automatically organize outputs into clear directory structures
- **Visualization**: Generate individual and/or combined plots for easy comparison
- **Flexible Configuration**: Use TOML files for easy parameter specification
- **Convolution Support**: Run convolutions on existing or new main calculations
- **Batch Processing**: Process multiple existing calculations with new convolution parameters

## Installation

### From Source

```bash
git clone https://github.com/Joint-Photon-Sciences-Institute/FDMNES-Explorer.git
cd FDMNES-Explorer
pip install -e .
```

### Requirements

- Python 3.7+
- FDMNES (with MPI support)
- NumPy
- Matplotlib
- TOML

```bash
pip install numpy matplotlib toml
```

## Quick Start

1. Create a TOML configuration file (see `examples/` for templates)
2. Run the explorer:

```bash
python -m fdmnes_explorer your_config.toml
```

Or if installed:

```bash
fdmnes-explorer your_config.toml
```

## Configuration

### Basic Structure

```toml
# FDMNES execution settings
fdmnes_executable = "~/FDMNES/parallel_fdmnes/mpirun_fdmnes"
num_cores = 20
output_dir = "exploration_runs"

# Structure file
structure = "path/to/structure.cif"

# Main calculation parameters with ITERATE keywords
fdmnes_input = """
Range
-10.0 .1 -5.0 .005 0.0 .05 2.0 0.01 3.2 .1 7.0

Green

Screening ITERATE
[0.2, 0.3, 0.4]

Spinorbit
"""
```

### Parameter Iteration Syntax

The `ITERATE` keyword marks parameters for systematic variation:

#### 1. Discrete Values
```toml
Screening ITERATE
[0.2, 0.3, 0.4]
```

#### 2. Range Values
```toml
Gamma_hole ITERATE
range[0.5, 2.0, 0.5]
```

#### 3. Boolean Parameters (with/without card)
```toml
Spinorbit ITERATE
```

### Working with Existing Calculations

Run convolutions on existing main calculations:

```toml
# Option 1: Auto-discover all subdirectories
main_dir_root = "/path/to/exploration_runs"

# Option 2: Specific directory
existing_main_dir = "exploration_runs/run_000"

# Option 3: Multiple directories
existing_main_dirs = [
    "exploration_runs/run_000",
    "exploration_runs/run_001"
]

# Convolution parameters to iterate
fdmnes_conv = """
Gamma_hole ITERATE
[0.5, 1.0, 1.5, 2.0]
"""
```

### Plotting Options

```toml
[plot_options]
individual = true  # Create individual plots for each calculation
combined = true    # Create combined plot with all spectra overlaid
```

## Output Structure

The explorer creates an organized directory structure:

```
exploration_runs/
├── run_000_Screening_0p2_Spinorbit_on/
│   ├── working/           # FDMNES working directory
│   ├── input_file/        # Main calculation input
│   ├── input_conv/        # Convolution inputs
│   ├── output_txt/        # Main calculation outputs
│   ├── output_conv/       # Convolution outputs
│   └── plots/
│       ├── stick/         # Main calculation plots
│       ├── conv/          # Individual convolution plots
│       └── combined_convolutions.png  # Combined plot
├── run_001_Screening_0p3_Spinorbit_on/
│   └── ...
└── exploration_summary.csv  # Summary of all runs with parameters
```

## Examples

### Example 1: Full Parameter Exploration

```toml
# See examples/full_exploration.toml
fdmnes_input = """
Green
Screening ITERATE
range[0.2, 0.6, 0.1]

Spinorbit ITERATE
"""

fdmnes_conv = """
Gamma_hole ITERATE
[0.5, 1.0, 1.5]
"""
```

This will create:
- 10 main calculations (5 Screening values × 2 Spinorbit states)
- 30 total convolutions (10 main × 3 Gamma_hole values)

### Example 2: Convolution-Only on Existing Data

```toml
# See examples/convolution_only.toml
main_dir_root = "previous_exploration_runs"

fdmnes_conv = """
E_cut ITERATE
range[-10, 10, 5]
"""
```

## Advanced Usage

### Custom FDMNES Executable

```toml
fdmnes_executable = "/custom/path/to/fdmnes"
num_cores = 40
```

### Combine Multiple Parameter Types

```toml
fdmnes_input = """
# Boolean parameter
Excited ITERATE

# Discrete values
Radius ITERATE
[5.0, 5.5, 6.0]

# Range
Screening ITERATE
range[0.1, 0.5, 0.1]
"""
```

## Workflow

1. **Parse Configuration**: Read TOML file and identify ITERATE parameters
2. **Generate Combinations**: Create all parameter combinations using itertools.product
3. **Create Directories**: Set up organized directory structure
4. **Generate Inputs**: Create FDMNES input files with substituted parameters
5. **Run Calculations**: Execute FDMNES with MPI parallelization
6. **Organize Outputs**: Move outputs to appropriate directories
7. **Generate Plots**: Create individual and/or combined visualizations
8. **Create Summary**: Generate CSV file with all parameters and results

## Troubleshooting

### FDMNES Not Found
Ensure the `fdmnes_executable` path is correct and accessible:
```toml
fdmnes_executable = "~/FDMNES/parallel_fdmnes/mpirun_fdmnes"
```

### MPI Issues
Adjust the number of cores based on your system:
```toml
num_cores = 4  # Reduce if you have fewer cores
```

### Memory Issues with Large Parameter Spaces
Consider running smaller batches by splitting your parameter ranges.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FDMNES Explorer in your research, please cite:

```bibtex
@software{fdmnes_explorer,
  title = {FDMNES Explorer: Automated Parameter Space Exploration for FDMNES},
  author = {Joint Photon Sciences Institute},
  year = {2024},
  url = {https://github.com/Joint-Photon-Sciences-Institute/FDMNES-Explorer}
}
```

## Acknowledgments

This tool is designed to work with FDMNES, developed by Yves Joly and the FDMNES team.