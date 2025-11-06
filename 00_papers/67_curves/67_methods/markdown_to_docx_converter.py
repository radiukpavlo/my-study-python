# markdown_to_docx_converter.py
"""
Markdown to DOCX Converter Script

This script converts a Markdown file to DOCX format using Pandoc via pypandoc.
It automatically handles Pandoc installation and downloads Pandoc binary if needed.
The script reads Markdown content from an external file and translates Ukrainian text to English.
"""

import subprocess
import sys
import importlib.util
from pathlib import Path
import textwrap
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

def read_markdown_file(file_path):
    """
    Reads Markdown content from an external file.
    
    Args:
        file_path (str or Path): Path to the Markdown file
        
    Returns:
        str: Content of the Markdown file
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        Exception: For other file reading errors
    """
    try:
        # Convert to Path object for better path handling
        markdown_path = Path(file_path)
        
        # Check if file exists
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        
        # Read the file content with UTF-8 encoding
        with open(markdown_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        print(f"Successfully read Markdown file: {markdown_path}")
        return content
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"Error reading Markdown file: {str(e)}")

def translate_ukrainian_to_english(text):
    """
    Translates Ukrainian text to English using Google Translate.
    
    Note: This function requires the googletrans library to be installed.
    If not available, it will install it automatically.
    
    Args:
        text (str): Text to translate from Ukrainian to English
        
    Returns:
        str: Translated text in English
    """
    try:
        # Check if googletrans is installed, install if missing
        if importlib.util.find_spec("googletrans") is None:
            print("Installing googletrans for translation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "googletrans==4.0.0-rc1"])
        
        from googletrans import Translator
        
        # Initialize translator
        translator = Translator()
        
        # Translate from Ukrainian (uk) to English (en)
        result = translator.translate(text, src='uk', dest='en')
        
        print("Text successfully translated from Ukrainian to English.")
        return result.text
        
    except Exception as e:
        print(f"Warning: Translation failed - {str(e)}")
        print("Returning original text without translation.")
        return text

def ensure_writable_directory(file_path):
    """
    Ensures that the directory for the output file is writable.
    
    Args:
        file_path (str or Path): Path to the file to be created
        
    Returns:
        Path: Validated and writable file path
    """
    output_path = Path(file_path)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if directory is writable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_path.parent}")
    
    return output_path

def convert_markdown_to_docx(markdown_content, output_path, include_toc=True):
    """
    Converts Markdown content to DOCX format using Pandoc.
    
    Args:
        markdown_content (str): Markdown text to convert
        output_path (str or Path): Path for the output DOCX file
        include_toc (bool): Whether to include a table of contents
    """
    # Ensure Pandoc is available
    pypandoc = ensure_pandoc()
    
    # Ensure output directory is writable
    output_file = ensure_writable_directory(output_path)
    
    # If file exists, try to remove it first (to avoid permission issues)
    if output_file.exists():
        try:
            output_file.unlink()
            print(f"Removed existing file: {output_file}")
        except PermissionError:
            raise PermissionError(f"Cannot overwrite existing file: {output_file}. File may be open in another application.")
        except Exception as e:
            raise Exception(f"Error removing existing file: {e}")
    
    # Prepare extra arguments for Pandoc
    extra_args = ["--standalone"]
    if include_toc:
        extra_args.append("--toc")  # Include table of contents
    
    try:
        # Convert Markdown to DOCX
        pypandoc.convert_text(
            markdown_content,
            to="docx",
            format="md",
            outputfile=str(output_file),
            extra_args=extra_args
        )
        
        print(f"✅ DOCX file successfully created: {output_file.resolve()}")
        return output_file
        
    except Exception as e:
        raise Exception(f"Pandoc conversion failed: {str(e)}")

def main():
    """
    Main function that orchestrates the conversion process.
    
    This function:
    1. Reads Markdown content from external file
    2. Translates Ukrainian text to English
    3. Converts the translated content to DOCX format
    """
    # Define input and output file paths
    input_markdown_file = "journals.md"
    output_docx_file = "journals_doc.docx"
    
    try:
        # Step 1: Read Markdown content from external file
        print("Reading Markdown content from external file...")
        markdown_text = read_markdown_file(input_markdown_file)
        
        # Step 2: Translate Ukrainian text to English
        # print("Translating text from Ukrainian to English...")
        # translated_text = translate_ukrainian_to_english(markdown_text)
        
        # Step 3: Convert translated Markdown to DOCX
        print("Converting translated Markdown to DOCX format...")
        created_file = convert_markdown_to_docx(markdown_text, output_docx_file, include_toc=True)
        
        print("Conversion process completed successfully!")
        print(f"Output file location: {created_file.resolve()}")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Please make sure 'input_markdown.md' exists in the current directory.")
    except PermissionError as e:
        print(f"❌ Permission Error: {e}")
        print("Possible solutions:")
        print("1. Close the DOCX file if it's open in another application")
        print("2. Run Jupyter Notebook as administrator (Windows)")
        print("3. Check file permissions in the output directory")
        print("4. Try changing the output file name or directory")
    except Exception as e:
        print(f"❌ Conversion Error: {e}")

# Make the script executable from Jupyter Notebook
if __name__ == "__main__":
    # This allows the script to be run directly from a Jupyter Notebook cell
    # Simply execute: !python markdown_to_docx_converter.py
    # Or run the main() function directly in a cell
    main()

# For easy execution in Jupyter Notebook, you can also run:
# main()