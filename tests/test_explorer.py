#!/usr/bin/env python3
"""
Test suite for FDMNES explorer
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
import toml
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdmnes_explorer.explorer import (
    parse_iterate_params,
    generate_parameter_combinations,
    generate_fdmnes_input
)

class TestTOMLParser(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_parse_discrete_values(self):
        """Test parsing ITERATE with discrete values"""
        config = """
structure = "/path/to/structure.cif"

fdmnes_input = \"\"\"
Screening ITERATE
[0.2, 0.3, 0.4]

Edge
M4
\"\"\"
"""
        toml_path = Path(self.test_dir) / "test.toml"
        toml_path.write_text(config)
        
        from fdmnes_explorer import parse_toml_config
        result = parse_toml_config(str(toml_path))
        
        self.assertIn("Screening", result["main_params"])
        self.assertEqual(result["main_params"]["Screening"], [0.2, 0.3, 0.4])
    
    def test_parse_range_values(self):
        """Test parsing ITERATE with range syntax"""
        config = """
structure = "/path/to/structure.cif"

fdmnes_conv = \"\"\"
E_cut ITERATE
range[-0.5, 2, 0.1]

Gamma_hole ITERATE
range[0, 1, 0.01]
\"\"\"
"""
        toml_path = Path(self.test_dir) / "test.toml"
        toml_path.write_text(config)
        
        from fdmnes_explorer import parse_toml_config
        result = parse_toml_config(str(toml_path))
        
        self.assertIn("E_cut", result["conv_params"])
        # Check that range is expanded correctly
        e_cut_values = result["conv_params"]["E_cut"]
        self.assertAlmostEqual(e_cut_values[0], -0.5)
        self.assertAlmostEqual(e_cut_values[-1], 2.0, places=1)
    
    def test_parse_boolean_card(self):
        """Test parsing ITERATE for boolean cards (with/without)"""
        config = """
structure = "/path/to/structure.cif"

fdmnes_input = \"\"\"
Spinorbit ITERATE

Green

Edge
M4
\"\"\"
"""
        toml_path = Path(self.test_dir) / "test.toml"
        toml_path.write_text(config)
        
        from fdmnes_explorer import parse_toml_config
        result = parse_toml_config(str(toml_path))
        
        self.assertIn("Spinorbit", result["main_params"])
        self.assertEqual(result["main_params"]["Spinorbit"], [True, False])


class TestParameterCombinations(unittest.TestCase):
    def test_generate_combinations(self):
        """Test generation of all parameter combinations"""
        from fdmnes_explorer import generate_parameter_combinations
        
        params = {
            "Screening": [0.2, 0.3],
            "Spinorbit": [True, False],
            "E_cut": [-0.5, 0.0]
        }
        
        combos = generate_parameter_combinations(params)
        
        # Should have 2 * 2 * 2 = 8 combinations
        self.assertEqual(len(combos), 8)
        
        # Check naming convention
        first = combos[0]
        self.assertIn("Screening", first["name"])
        self.assertIn("Spinorbit", first["name"])
    
    def test_boolean_naming(self):
        """Test that boolean parameters get on/off in names"""
        from fdmnes_explorer import generate_parameter_combinations
        
        params = {"Spinorbit": [True, False]}
        combos = generate_parameter_combinations(params)
        
        names = [c["name"] for c in combos]
        self.assertTrue(any("Spinorbit_on" in n for n in names))
        self.assertTrue(any("Spinorbit_off" in n for n in names))


class TestInputGeneration(unittest.TestCase):
    def test_generate_main_input(self):
        """Test generation of main FDMNES input file"""
        from fdmnes_explorer import generate_fdmnes_input
        
        template = """Energpho

Range
-1 .1 0 0.01 1.5 0.1 5

Green

Screening ITERATE
VALUE_PLACEHOLDER

Spinorbit ITERATE

Edge
M4"""
        
        params = {"Screening": 0.3, "Spinorbit": True}
        result = generate_fdmnes_input(template, params)
        
        self.assertIn("Screening", result)
        self.assertIn("0.3", result)
        self.assertIn("Spinorbit", result)
        self.assertNotIn("! Spinorbit", result)
    
    def test_comment_out_boolean_card(self):
        """Test commenting out boolean cards when False"""
        from fdmnes_explorer import generate_fdmnes_input
        
        template = """Green

Spinorbit ITERATE

Edge
M4"""
        
        params = {"Spinorbit": False}
        result = generate_fdmnes_input(template, params)
        
        self.assertIn("! Spinorbit", result)
        self.assertNotIn("\nSpinorbit\n", result)


class TestPlotGeneration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_plot_stick_data(self):
        """Test plotting of stick spectrum data"""
        from fdmnes_explorer import plot_stick_spectrum
        
        # Create sample data file
        data_file = Path(self.test_dir) / "test.txt"
        data_file.write_text("""# Energy   Intensity
-10.0  0.5
-5.0   1.0
0.0    0.8
5.0    0.3
10.0   0.1""")
        
        output_file = Path(self.test_dir) / "test.png"
        plot_stick_spectrum(str(data_file), str(output_file))
        
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()