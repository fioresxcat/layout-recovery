SURYA_LABELS = [
    'Blank',
    'Text',
    'TextInlineMath',
    'Code',
    'SectionHeader',
    'Caption',
    'Footnote',
    'Equation',
    'ListItem',
    'PageFooter',
    'PageHeader',
    'Picture',
    'Figure',
    'Table',
    'Form',
    'TableOfContents',
    'Handwriting'
]

PADDLE_DOCLAYOUT_L_LABELS = [
    'document title',
    'paragraph title', 
    'text',
    'page number',
    'abstract',
    'table of contents',
    'references',
    'footnotes',
    'header',
    'footer',
    'algorithm',
    'formula',
    'formula number',
    'image',
    'figure caption',
    'table',
    'table caption',
    'seal',
    'figure title',
    'figure',
    'header image',
    'footer image',
    'sidebar text'
]

FINAL_LABELS = [
    'blank',
    'text',                # regular paragraphs
    'table_of_contents',
    'title',              # document/section titles
    'header',             # page headers
    'footer',             # page footers
    'list',               # bullet points/numbered lists
    'table',              # tables with content
    'figure',             # images/diagrams
    'caption',            # for figures/tables
    'equation',           # mathematical formulas
    'footnote',           # references/notes
    'handwriting',        # handwriting
]