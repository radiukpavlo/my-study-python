# markdown_to_docx_converter.py
"""
Universal Markdown to DOCX Converter Script

This script converts any Markdown file to DOCX format using Pandoc via pypandoc.
It automatically handles Pandoc installation and downloads Pandoc binary if needed.
The script is universal - it accepts any input .md file and produces any output .docx file,
and correctly processes non-ASCII characters like Cyrillic.

Usage:
1. From terminal: python markdown_to_docx_converter.py input.md output.docx
2. From Jupyter Notebook: 
   from markdown_to_docx_converter import convert_md_to_docx
   convert_md_to_docx('input.md', 'output.docx')
"""

import subprocess
import sys
import importlib.util
from pathlib import Path
import argparse
import os

def ensure_pandoc():
    """Ensures that pypandoc is installed and Pandoc is available."""
    if importlib.util.find_spec("pypandoc") is None:
        print("Installing pypandoc...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pypandoc"])
    
    import pypandoc
    
    try:
        pypandoc.get_pandoc_version()
        print("Pandoc is already available in the system.")
    except OSError:
        print("Pandoc not found. Downloading and installing...")
        pypandoc.download_pandoc()
        print("Pandoc successfully installed.")
    
    return pypandoc

def validate_input_file(file_path):
    """Validates that the input Markdown file exists and has the correct extension."""
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != '.md':
        raise ValueError(f"Input file must have .md extension, got: {input_path.suffix}")
    return input_path

def ensure_output_directory(file_path):
    """Ensures that the directory for the output file exists and is writable."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_path.parent}")
    if output_path.suffix.lower() != '.docx':
        raise ValueError(f"Output file must have .docx extension, got: {output_path.suffix}")
    return output_path

def remove_existing_file(file_path):
    """Safely removes an existing file if it exists."""
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
    Converts a Markdown file to DOCX format by calling the Pandoc executable directly.
    
    This is the most robust method, bypassing pypandoc wrapper bugs and reliably
    handling special characters like Cyrillic by letting Pandoc manage file I/O.
    
    Args:
        input_file (str or Path): Path to the input Markdown file (.md)
        output_file (str or Path): Path to the output DOCX file (.docx)
        include_toc (bool): Whether to include a table of contents (default: True)
        
    Returns:
        Path: Path to the created DOCX file
    """
    try:
        print(f"Starting conversion: {input_file} → {output_file}")
        
        pypandoc = ensure_pandoc()
        validated_input = validate_input_file(input_file)
        validated_output = ensure_output_directory(output_file)
        remove_existing_file(validated_output)

        # --- THE ULTIMATE FIX: CALL PANDOC DIRECTLY ---
        # We bypass the buggy pypandoc wrappers and build the command manually.
        # This is the most reliable way to ensure correct file handling.
        
        # Get the path to the pandoc executable
        pandoc_path = pypandoc.get_pandoc_path()
        
        # Build the list of command-line arguments
        command = [
            pandoc_path,
            str(validated_input),  # Input file
            '--from=markdown',
            '--to=docx',
            '--output', str(validated_output),  # Output file
            '--standalone',
        ]
        if include_toc:
            command.append('--toc')
        
        print(f"Executing command: {' '.join(command)}")
        
        # Run the Pandoc command
        result = subprocess.run(
            command, 
            capture_output=True, # Capture stdout/stderr
            text=True, # Decode stdout/stderr as text
            encoding='utf-8' # Be explicit about encoding for captured output
        )

        # Check for errors
        if result.returncode != 0:
            # If Pandoc failed, raise an error with its output
            error_message = f"Pandoc failed with exit code {result.returncode}:\n{result.stderr}"
            raise RuntimeError(error_message)

        print(f"✅ Conversion successful!")
        print(f"DOCX file created: {validated_output.resolve()}")
        return validated_output
        
    except Exception as e:
        raise Exception(f"Conversion failed: {str(e)}")

def parse_arguments():
    """Parses command line arguments for terminal usage."""
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
    parser.add_argument('input_file', help='Input Markdown file (.md)')
    parser.add_argument('output_file', help='Output DOCX file (.docx)')
    parser.add_argument('--no-toc', action='store_true', help='Disable table of contents generation')
    return parser.parse_args()

def main():
    """Main function that handles both terminal and module usage."""
    try:
        args = parse_arguments()
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

if __name__ == "__main__":
    main()