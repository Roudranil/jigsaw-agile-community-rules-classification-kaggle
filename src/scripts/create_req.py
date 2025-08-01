#!/usr/bin/env python3
"""
Script to create a robust requirements file with exact versions from current environment.
Converts package==version format to pip install package==version format.
Handles PyTorch specially based on CUDA version detection.
"""

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

import pkg_resources


def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages and their versions from current environment."""
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    return installed_packages


def get_cuda_version() -> Optional[str]:
    """Detect CUDA version from system."""
    try:
        # Try nvidia-smi first
        output = subprocess.check_output(
            ["nvidia-smi"], universal_newlines=True, stderr=subprocess.DEVNULL
        )
        match = re.search(r"CUDA Version: ([\d.]+)", output)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        # Fallback to nvcc --version
        output = subprocess.check_output(
            ["nvcc", "--version"], universal_newlines=True, stderr=subprocess.DEVNULL
        )
        match = re.search(r"release ([\d.]+)", output)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def get_pytorch_install_command(torch_version: str, cuda_version: Optional[str]) -> str:
    """Generate proper PyTorch installation command based on CUDA version."""
    base_url = "https://download.pytorch.org/whl"

    if not cuda_version:
        return f"pip install torch=={torch_version} torchvision torchaudio"

    # Normalize CUDA version to major.minor
    cuda_parts = cuda_version.split(".")
    if len(cuda_parts) >= 2:
        cuda_major_minor = f"{cuda_parts[0]}.{cuda_parts[1]}"
    else:
        cuda_major_minor = cuda_parts[0]

    # Map CUDA versions to PyTorch wheel URLs based on official PyTorch documentation
    cuda_mapping = {
        "11.7": "cu117",
        "11.8": "cu118",
        "12.1": "cu121",
        "12.2": "cu122",
        "12.4": "cu124",
    }

    if cuda_major_minor in cuda_mapping:
        cuda_suffix = cuda_mapping[cuda_major_minor]
        return f"pip install torch=={torch_version} torchvision torchaudio --index-url {base_url}/{cuda_suffix}"
    else:
        # Default to CPU if CUDA version not supported
        print(
            f"Warning: CUDA version {cuda_version} not directly supported, defaulting to CPU version"
        )
        return f"pip install torch=={torch_version} torchvision torchaudio"


def parse_requirement_line(line: str) -> Optional[tuple]:
    """Parse a requirements.txt line to extract package name and version."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Handle various requirement formats
    patterns = [
        r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)==([\d.]+(?:[a-zA-Z0-9._-]*)?)$",  # exact version
        r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)\s*$",  # package name only
        r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)>=?([\d.]+(?:[a-zA-Z0-9._-]*)?)$",  # minimum version
    ]

    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            if len(match.groups()) == 2:
                return (match.group(1), match.group(2))
            else:
                return (match.group(1), None)

    return None


def create_robust_requirements(input_file: str, output_file: str) -> None:
    """Create robust requirements file with exact versions and pip install commands."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    # Get current environment info
    installed_packages = get_installed_packages()
    cuda_version = get_cuda_version()

    print(
        f"Detected CUDA version: {cuda_version if cuda_version else 'None (CPU only)'}"
    )
    print(f"Found {len(installed_packages)} installed packages in current environment")

    # Read input requirements
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_lines = []
    skipped_lines = []
    processed_packages = []

    # Add header comment
    output_lines.append("# Robust requirements file generated with exact versions")
    output_lines.append("# Generated from current environment packages")
    if cuda_version:
        output_lines.append(f"# CUDA version detected: {cuda_version}")
    output_lines.append("")

    for line_num, line in enumerate(lines, 1):
        parsed = parse_requirement_line(line)
        if not parsed:
            if line.strip() and not line.strip().startswith("#"):
                skipped_lines.append(f"Line {line_num}: {line.strip()}")
            continue

        pkg_name, specified_version = parsed
        pkg_key = pkg_name.lower().replace("_", "-")

        # Check if package is installed in current environment
        if pkg_key in installed_packages:
            actual_version = installed_packages[pkg_key]

            # Special handling for PyTorch
            if pkg_key in ["torch", "pytorch"]:
                install_cmd = get_pytorch_install_command(actual_version, cuda_version)
                output_lines.append(install_cmd)
                processed_packages.append(
                    f"{pkg_name}: {actual_version} (PyTorch with CUDA support)"
                )
            else:
                install_cmd = f"pip install {pkg_name}=={actual_version}"
                output_lines.append(install_cmd)
                processed_packages.append(f"{pkg_name}: {actual_version}")
        else:
            # Package not found in environment
            if specified_version:
                install_cmd = f"pip install {pkg_name}=={specified_version}"
                output_lines.append(install_cmd)
                skipped_lines.append(
                    f"Package {pkg_name} not found in environment, using specified version {specified_version}"
                )
            else:
                output_lines.append(
                    f"# pip install {pkg_name}  # Version not specified and package not found in environment"
                )
                skipped_lines.append(
                    f"Package {pkg_name} not found in environment and no version specified"
                )

    # Write output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    # Print summary
    print(f"\n✅ Robust requirements written to: {output_file}")
    print(f"📦 Processed {len(processed_packages)} packages:")
    for pkg in processed_packages:
        print(f"   - {pkg}")

    if skipped_lines:
        print(f"\n⚠️  Warnings/Skipped items:")
        for item in skipped_lines:
            print(f"   - {item}")


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print(
            "Usage: python pin_requirements.py <input_requirements.txt> <output_file.txt>"
        )
        print("\nExample:")
        print("  python pin_requirements.py requirements.txt robust_requirements.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        create_robust_requirements(input_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
