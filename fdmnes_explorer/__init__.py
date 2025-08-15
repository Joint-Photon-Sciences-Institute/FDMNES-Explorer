"""
FDMNES Explorer - Automated parameter space exploration for FDMNES calculations
"""

__version__ = "1.0.0"
__author__ = "Joint Photon Sciences Institute"

from .explorer import (
    parse_toml_config,
    parse_iterate_params,
    generate_parameter_combinations,
    generate_fdmnes_input,
    run_fdmnes,
    plot_stick_spectrum,
    plot_conv_spectrum,
    create_combined_plot,
    organize_outputs,
    main
)

__all__ = [
    'parse_toml_config',
    'parse_iterate_params', 
    'generate_parameter_combinations',
    'generate_fdmnes_input',
    'run_fdmnes',
    'plot_stick_spectrum',
    'plot_conv_spectrum',
    'create_combined_plot',
    'organize_outputs',
    'main'
]