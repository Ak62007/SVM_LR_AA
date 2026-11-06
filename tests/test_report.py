from pypdf import PdfReader
from report.build_report import build_report


def test_report_exports_all_sections_and_derivations(experiment_output, tmp_path):
    markdown, pdf = build_report(experiment_output, tmp_path)
    text = markdown.read_text()
    for title in ['Introduction', 'Theory', 'Implementation', 'Verification',
                  'Computational comparison', 'Sensitivity study', 'Conclusion', 'Appendix']:
        assert title in text
    assert '**(28)**' in text
    reader = PdfReader(pdf)
    assert 10 <= len(reader.pages) < 50
    extracted = '\n'.join(page.extract_text() for page in reader.pages)
    for phrase in ['central path', 'Schur', 'generalized', 'iteration-limit', 'CLARABEL']:
        assert phrase.lower() in extracted.lower()
    assert '\u25a0' not in extracted
    assert all(page.mediabox.width > 500 for page in reader.pages)
