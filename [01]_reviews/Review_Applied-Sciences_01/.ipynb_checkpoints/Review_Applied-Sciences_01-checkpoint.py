import os
import sys
import docling
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def convert_pdf_to_xml(pdf_path, output_xml_path):
    """
    Convert a PDF research article to XML format using Docling.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_xml_path (str): Path to save the output XML file
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Print docling version and available modules for debugging
        print(f"Docling version: {docling.__version__}")
        print(f"Docling modules: {dir(docling)}")
        
        # Load the PDF document using Docling - the correct import may vary
        # Let's try different approaches based on the library structure
        print(f"Loading PDF from {pdf_path}...")
        
        try:
            # Try the document module if available
            if hasattr(docling, 'document'):
                doc = docling.document.parse_pdf(pdf_path)
            # Try alternate imports if document module isn't available
            elif hasattr(docling, 'pdf'):
                doc = docling.pdf.read(pdf_path)
            elif hasattr(docling, 'parsing'):
                doc = docling.parsing.parse_pdf(pdf_path)
            else:
                # Fallback to importing directly if specific modules aren't found
                from docling.core.document import Document
                doc = Document.from_pdf(pdf_path)
        except Exception as e:
            print(f"Error importing document module: {e}")
            # Try alternate import approaches
            try:
                from docling.pdf import parse_document
                doc = parse_document(pdf_path)
            except Exception as e2:
                print(f"Error with alternate import approach: {e2}")
                raise
        
        # Create the root XML element
        root = ET.Element("article")
        
        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        if hasattr(doc, 'metadata') and doc.metadata:
            for key, value in doc.metadata.items():
                meta_item = ET.SubElement(metadata, "meta", {"name": key})
                meta_item.text = str(value)
        
        # Process content
        content = ET.SubElement(root, "content")
        
        # Process title if available
        if hasattr(doc, 'title') and doc.title:
            title = ET.SubElement(content, "title")
            title.text = doc.title
        
        # Process abstract if available
        if hasattr(doc, 'abstract') and doc.abstract:
            abstract = ET.SubElement(content, "abstract")
            abstract.text = doc.abstract
        
        # Process sections if available
        if hasattr(doc, 'sections'):
            for section in doc.sections:
                sec_elem = ET.SubElement(content, "section")
                
                # Section heading
                if hasattr(section, 'heading') and section.heading:
                    heading = ET.SubElement(sec_elem, "heading")
                    heading.text = section.heading
                
                # Section text/paragraphs
                if hasattr(section, 'paragraphs') and section.paragraphs:
                    for para in section.paragraphs:
                        paragraph = ET.SubElement(sec_elem, "paragraph")
                        if hasattr(para, 'text'):
                            paragraph.text = para.text
                        else:
                            paragraph.text = str(para)
        
        # Process text blocks/paragraphs if sections aren't available
        if hasattr(doc, 'paragraphs') and doc.paragraphs:
            paragraphs_section = ET.SubElement(content, "paragraphs")
            for para in doc.paragraphs:
                paragraph = ET.SubElement(paragraphs_section, "paragraph")
                if hasattr(para, 'text'):
                    paragraph.text = para.text
                else:
                    paragraph.text = str(para)
        
        # Process figures if available
        if hasattr(doc, 'figures') and doc.figures:
            figures_section = ET.SubElement(content, "figures")
            for i, figure in enumerate(doc.figures):
                fig_elem = ET.SubElement(figures_section, "figure", {"id": f"fig_{i+1}"})
                
                if hasattr(figure, 'caption') and figure.caption:
                    caption = ET.SubElement(fig_elem, "caption")
                    caption.text = figure.caption
                
                if hasattr(figure, 'content_type') and figure.content_type:
                    fig_elem.set("type", figure.content_type)
        
        # Process tables if available
        if hasattr(doc, 'tables') and doc.tables:
            tables_section = ET.SubElement(content, "tables")
            for i, table in enumerate(doc.tables):
                table_elem = ET.SubElement(tables_section, "table", {"id": f"tab_{i+1}"})
                
                if hasattr(table, 'caption') and table.caption:
                    caption = ET.SubElement(table_elem, "caption")
                    caption.text = table.caption
                
                # Try to extract table content if available
                if hasattr(table, 'rows') and table.rows:
                    for row in table.rows:
                        row_elem = ET.SubElement(table_elem, "row")
                        for cell in row:
                            cell_elem = ET.SubElement(row_elem, "cell")
                            cell_elem.text = str(cell)
        
        # Process references if available
        if hasattr(doc, 'references') and doc.references:
            refs_section = ET.SubElement(root, "references")
            for i, ref in enumerate(doc.references):
                ref_elem = ET.SubElement(refs_section, "reference", {"id": f"ref_{i+1}"})
                if hasattr(ref, 'text'):
                    ref_elem.text = ref.text
                else:
                    ref_elem.text = str(ref)
        
        # Pretty print the XML and save to file
        xml_str = ET.tostring(root, encoding='utf-8')
        xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
        
        with open(output_xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_pretty)
        
        print(f"Successfully converted PDF to XML. Output saved to {output_xml_path}")
        return True
    
    except Exception as e:
        print(f"Error converting PDF to XML: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
    
    # Define input and output paths
    pdf_filename = 'applsci-3508831-peer-review-v1.pdf'
    pdf_path = os.path.join(current_dir, pdf_filename)
    
    # Create output filename based on input filename
    output_filename = os.path.splitext(pdf_filename)[0] + '.xml'
    output_path = os.path.join(current_dir, output_filename)
    
    # Convert PDF to XML
    success = convert_pdf_to_xml(pdf_path, output_path)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()