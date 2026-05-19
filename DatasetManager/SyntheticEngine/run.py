"""
SyntheticEngine/run.py
========================
CLI entry point for the Synthetic Data Generation Engine.
Usage:
  python run.py --config config/default_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add current dir to path to ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))

from pipeline.synthetic_pipeline import SyntheticPipeline

def setup_logging(level=logging.INFO):
    script_dir = Path(__file__).resolve().parent
    log_file = script_dir / "synthetic_engine.log"
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Engine for YOLO Pose")
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir / "config" / "default_config.yaml"
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config YAML")
    parser.add_argument("--samples", type=int, help="Override samples_per_source")
    parser.add_argument("--workers", type=int, help="Override number of workers (-1 for all cores)")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to the script directory
        alt_path = script_dir / args.config
        if alt_path.exists():
            config_path = alt_path
        else:
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Setup logging based on config debug flag
    debug_mode = config.get('debug', False)
    level = logging.DEBUG if debug_mode else logging.INFO
    setup_logging(level=level)
    log = logging.getLogger("SyntheticEngine")
    
    # Process color_to_label: convert string "[255, 255, 255]" to actual tuple
    if 'color_to_label' in config:
        new_c2l = {}
        for k, v in config['color_to_label'].items():
            if isinstance(k, str) and k.startswith("[") and k.endswith("]"):
                # Convert "[B, G, R]" to (B, G, R)
                try:
                    tuple_val = tuple(map(int, k.strip("[]").split(",")))
                    new_c2l[tuple_val] = v
                except ValueError:
                    new_c2l[k] = v
            else:
                new_c2l[k] = v
        config['color_to_label'] = new_c2l

    # Overrides
    if args.samples:
        config['augmentation']['samples_per_source'] = args.samples
    if args.workers:
        config['augmentation']['workers'] = args.workers

    log.info("Loaded config from %s (Debug: %s)", config_path, debug_mode)
    
    try:
        pipeline = SyntheticPipeline(config)
        pipeline.run()
        log.info("Synthetic generation completed successfully.")
    except Exception as e:
        log.exception("Pipeline failed with error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
