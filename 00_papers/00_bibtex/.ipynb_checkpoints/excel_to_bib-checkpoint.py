import pandas as pd
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


def excel_to_bib(excel_file: str, bib_file: str) -> None:
    """
    Read an Excel file of references and export entries to a .bib file.
    """
    df = pd.read_excel(excel_file, dtype=str).fillna('')
    db = BibDatabase()
    entries = []

    for _, row in df.iterrows():
        entry = {k: v for k, v in row.items() if v}
        entry['ID'] = entry.pop('citation_key', '')
        entry['ENTRYTYPE'] = entry.get('ENTRYTYPE', 'article')
        entries.append(entry)

    db.entries = entries
    writer = BibTexWriter()
    with open(bib_file, 'w', encoding='utf-8') as bf:
        bf.write(writer.write(db))