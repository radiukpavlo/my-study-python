# markdown_to_docx_converter.py
"""
Universal Markdown to DOCX Converter Script

This script converts any Markdown file to DOCX format using Pandoc via pypandoc.
It automatically handles Pandoc installation and downloads Pandoc binary if needed.
The script is universal - it accepts any input .md file and produces any output .docx file.

Usage:
1. From terminal: python markdown_to_docx_converter.py input.md output.docx
2. From Jupyter Notebook: 
   import markdown_to_docx_converter
   markdown_to_docx_converter.convert_md_to_docx('input.md', 'output.docx')
"""

import subprocess
import sys
import importlib.util
from pathlib import Path
import argparse
import os

def ensure_pandoc():
    """
    Ensures that pypandoc is installed and Pandoc is available.
    
    This function checks if pypandoc is installed, installs it if missing,
    and verifies that Pandoc binary is available. If Pandoc is not found
    in the system PATH, it automatically downloads and installs it.
    
    Returns:
        module: The pypandoc module for use in conversion operations
    """
    # Check if pypandoc is installed, install if missing
    if importlib.util.find_spec("pypandoc") is None:
        print("Installing pypandoc...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pypandoc"])
    
    # Import pypandoc after ensuring it's installed
    import pypandoc
    
    try:
        # Try to get Pandoc version to verify it's available
        pypandoc.get_pandoc_version()
        print("Pandoc is already available in the system.")
    except OSError:
        # If Pandoc is not found in PATH, download and install it
        print("Pandoc not found. Downloading and installing...")
        pypandoc.download_pandoc()
        print("Pandoc successfully installed.")
    
    return pypandoc

def validate_input_file(file_path):
    """
    Validates that the input Markdown file exists and has the correct extension.
    
    Args:
        file_path (str or Path): Path to the input Markdown file
        
    Returns:
        Path: Validated input file path
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't have .md extension
    """
    input_path = Path(file_path)
    
    # Check if file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if file has .md extension
    if input_path.suffix.lower() != '.md':
        raise ValueError(f"Input file must have .md extension, got: {input_path.suffix}")
    
    return input_path

def ensure_output_directory(file_path):
    """
    Ensures that the directory for the output file exists and is writable.
    
    Args:
        file_path (str or Path): Path to the output file
        
    Returns:
        Path: Validated output file path
    """
    output_path = Path(file_path)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if directory is writable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_path.parent}")
    
    # Check if file has .docx extension
    if output_path.suffix.lower() != '.docx':
        raise ValueError(f"Output file must have .docx extension, got: {output_path.suffix}")
    
    return output_path

def remove_existing_file(file_path):
    """
    Safely removes an existing file if it exists.
    
    Args:
        file_path (Path): Path to the file to remove
        
    Raises:
        PermissionError: If the file cannot be removed (e.g., it's open)
    """
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"Removed existing file: {file_path}")
        except PermissionError:
            raise PermissionError(f"Cannot overwrite existing file: {file_path}. File may be open in another application.")
        except Exception as e:
            raise Exception(f"Error removing existing file: {e}")

def convert_md_to_docx(input_file, output_file, include_toc=True):
    """
    Converts a Markdown file to DOCX format using Pandoc.
    
    This is the main conversion function that can be called directly from
    Jupyter Notebook or other Python scripts.
    
    Args:
        input_file (str or Path): Path to the input Markdown file (.md)
        output_file (str or Path): Path to the output DOCX file (.docx)
        include_toc (bool): Whether to include a table of contents (default: True)
        
    Returns:
        Path: Path to the created DOCX file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file extensions are incorrect
        PermissionError: If output file cannot be written
        Exception: For other conversion errors
    """
    try:
        print(f"Starting conversion: {input_file} → {output_file}")
        
        # Validate input file
        validated_input = validate_input_file(input_file)
        
        # Validate output file and directory
        validated_output = ensure_output_directory(output_file)
        
        # Remove existing output file if it exists
        remove_existing_file(validated_output)
        
        # Ensure Pandoc is available
        pypandoc = ensure_pandoc()
        
        # Read the Markdown content
        with open(validated_input, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        print(f"Successfully read {len(markdown_content)} characters from {validated_input}")
        
        # Prepare extra arguments for Pandoc
        extra_args = ["--standalone"]
        if include_toc:
            extra_args.append("--toc")  # Include table of contents
        
        # Convert Markdown to DOCX
        pypandoc.convert_text(
            markdown_content,
            to="docx",
            format="md",
            outputfile=str(validated_output),
            extra_args=extra_args
        )
        
        print(f"✅ Conversion successful!")
        print(f"DOCX file created: {validated_output.resolve()}")
        return validated_output
        
    except Exception as e:
        raise Exception(f"Conversion failed: {str(e)}")

def parse_arguments():
    """
    Parses command line arguments for terminal usage.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert Markdown files to DOCX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python markdown_to_docx_converter.py input.md output.docx
  python markdown_to_docx_converter.py README.md documentation.docx
  python markdown_to_docx_converter.py notes.md /path/to/output/report.docx
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input Markdown file (.md)'
    )
    
    parser.add_argument(
        'output_file',
        help='Output DOCX file (.docx)'
    )
    
    parser.add_argument(
        '--no-toc',
        action='store_true',
        help='Disable table of contents generation'
    )
    
    return parser.parse_args()

def main():
    """
    Main function that handles both terminal and module usage.
    
    When run from terminal, parses command line arguments.
    When imported as module, provides the convert_md_to_docx function.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Perform the conversion
        convert_md_to_docx(
            args.input_file,
            args.output_file,
            include_toc=not args.no_toc
        )
        
        print("\n🎉 Conversion completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Conversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

# Make the script executable from both terminal and Jupyter Notebook
if __name__ == "__main__":
    # This allows the script to be run directly from terminal or Jupyter Notebook
    main()

# For easy execution in Jupyter Notebook, you can also use:
# from markdown_to_docx_converter import convert_md_to_docx
# convert_md_to_docx('input.md', 'output.docx')
