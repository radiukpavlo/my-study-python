import bibtexparser
import pandas as pd


def bib_to_excel(bib_file: str, excel_file: str) -> None:
    """
    Read a .bib file and export its entries to an Excel file.
    """
    with open(bib_file, 'r', encoding='utf-8') as bf:
        bib_db = bibtexparser.load(bf)

    records = []
    for entry in bib_db.entries:
        rec = {
            'citation_key': entry.get('ID'),
            'ENTRYTYPE': entry.get('ENTRYTYPE'),
            'author': entry.get('author'),
            'title': entry.get('title'),
            'year': entry.get('year'),
            'journal': entry.get('journal')
        }
        # include any additional fields dynamically
        for k, v in entry.items():
            if k not in rec and v:
                rec[k] = v
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_excel(excel_file, index=False)