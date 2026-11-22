"""Build the final report in the style of the NLE (Natural Language Engineering) journal template.

    python docs/report/build_report.py

Writes (in docs/report/):
    figures/pipeline.png   diagram made here; the other figures are copied from results/
    report.tex             LaTeX source of the same content (compile with pdflatex, e.g. on Overleaf)
    report.pdf             PDF made from the HTML version with headless Chrome/Edge (if installed)
    build/report.html      intermediate file for the PDF (gitignored)

The text and all numbers come from content.py, which reads results/. The layout follows the
template: italic title, letter-spaced author names, abstract between two rules, centred numbered
section headings, tables with a double rule above and below and the caption above, figures with
the caption below, centred decimal points in tables, author-date citations, an alphabetical list of
references, and running headlines (short title on odd pages, short author name on even pages).
"""

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import content  # noqa: E402

BUILD = HERE / "build"
FIG = HERE / "figures"


# ------------------------------------------------------------------ figures
def make_figures():
    FIG.mkdir(exist_ok=True)
    for src, dst in (("comparison.png", "comparison.png"), ("lstm_attn_curves.png", "lstm_curves.png"),
                     ("transformer_deep_curves.png", "transformer_curves.png"),
                     ("lstm_attn_attention_1.png", "lstm_attention.png")):
        shutil.copyfile(ROOT / "results" / src, FIG / dst)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eef7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec="#33445f", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#33445f", lw=1))

    box(1, 17, 17, 12, "PHINC\nraw file\n(13,738 rows)")
    box(23, 17, 19, 12, "Cleaning and\nfixed split\n(seed 42)")
    box(48, 33, 24, 9, "LSTM + attention\n(from scratch)")
    box(48, 18.5, 24, 9, "Transformer\n(from scratch)")
    box(48, 4, 24, 9, "mBART-50, mT5-small\n(zero-shot, fine-tuned)")
    box(78, 17, 21, 12, "Shared evaluation\nBLEU + chrF\n+ error analysis", fc="#fdf1dc")
    arrow(18.3, 23, 22.7, 23)
    for y in (37.5, 23, 8.5):
        arrow(42.3, 23, 47.7, y)
        arrow(72.3, y, 77.7, 23)
    ax.text(57, 44.8, "train on the training split only", ha="center", fontsize=6.5, style="italic", color="#555")
    fig.savefig(FIG / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ inline markup
CITE = re.compile(r"\[@([^\]]+)\]")
TCITE = re.compile(r"\[\+([^\]]+)\]")
INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\[@[^\]]+\]|\[\+[^\]]+\]|\[\[[^\]]+\]\])")


class Numbering:
    """Numbers tables and figures in the order they appear."""

    def __init__(self, blocks):
        self.num = {}
        t = f = 0
        for kind, data in blocks:
            if kind == "table":
                t += 1
                self.num["tab:" + data["label"]] = t
            elif kind == "fig":
                f += 1
                self.num["fig:" + data["label"]] = f


def cite_text(keys, refs, textual=False):
    parts = []
    for k in keys.split(";"):
        short, year, _ = refs[k.strip()]
        parts.append((short, year))
    if textual:
        return "; ".join(f"{s} ({y})" for s, y in parts)
    return "(" + "; ".join(f"{s} {y}" for s, y in parts) + ")"


def inline_html(text, refs, num):
    out = []
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            out.append(f"<b>{inline_html(piece[2:-2], refs, num)}</b>")
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(f"<i>{inline_html(piece[1:-1], refs, num)}</i>")
        elif piece.startswith("`"):
            out.append(f"<code>{html.escape(piece[1:-1])}</code>")
        elif piece.startswith("[@"):
            out.append(html.escape(cite_text(piece[2:-1], refs)))
        elif piece.startswith("[+"):
            out.append(html.escape(cite_text(piece[2:-1], refs, textual=True)))
        elif piece.startswith("[["):
            key = piece[2:-2]
            name = "Table" if key.startswith("tab:") else "Figure"
            out.append(f"{name}&nbsp;{num[key]}")
        else:
            out.append(html.escape(piece.replace("--", "\u2013")))
    return "".join(out)


LATEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
             "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
             "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}


def tex_escape(s):
    return "".join(LATEX_ESC.get(ch, ch) for ch in s)


UNICODE_TEX = {"\u2013": "--", "\u2014": "---", "\u201c": "``", "\u201d": "''", "\u2019": "'", "\u2018": "`",
               "\u00b7": r"\textperiodcentered{}", "\u0107": r"\'c", "\u00a0": "~", "\u00d7": r"\ensuremath{\times}"}


def tex_text(s):
    s = tex_escape(s.replace("--", "\u2013"))
    for k, v in UNICODE_TEX.items():
        s = s.replace(k, v)
    return s


def inline_tex(text, refs, num):
    out = []
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            out.append(r"\textbf{" + inline_tex(piece[2:-2], refs, num) + "}")
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(r"\textit{" + inline_tex(piece[1:-1], refs, num) + "}")
        elif piece.startswith("`"):
            out.append(r"\texttt{" + tex_escape(piece[1:-1]) + "}")
        elif piece.startswith("[@"):
            out.append(tex_text(cite_text(piece[2:-1], refs)))
        elif piece.startswith("[+"):
            out.append(tex_text(cite_text(piece[2:-1], refs, textual=True)))
        elif piece.startswith("[["):
            key = piece[2:-2]
            name = "Table" if key.startswith("tab:") else "Figure"
            out.append(f"{name}~\\ref{{{key}}}")
        else:
            out.append(tex_text(piece))
    return "".join(out)


def centre_decimals(cell):
    """NLE style: a centred decimal point inside tables (36.76 -> 36\u00b776), numbers only."""
    plain = cell.replace("**", "")
    if re.fullmatch(r"[\d,]*\.\d+", plain):
        return cell.replace(".", "\u00b7")
    return cell


def parse_align(spec):
    """'lrr' -> ['l','r','r']; 'p1p2p3' -> ['p1','p2','p3']."""
    return ["p" if x.startswith("p") else x for x in re.findall(r"p\d|[lrc]", spec)]


P_WIDTHS = {  # relative widths of p-columns, used by both outputs
    "p1p2": [0.22, 0.78],
    "p1p2p3p4p5": [0.23, 0.20, 0.15, 0.23, 0.19],   # examples table
    "p1p2p3p4p6": [0.17, 0.22, 0.25, 0.18, 0.18],   # settings table
}


# ------------------------------------------------------------------ HTML
HTML_CSS = """
@page { size: A4; margin: 29mm 33mm 29mm 33mm; }
@page :first { @top-left { content: "Natural Language Engineering 1 (1): 1\\2013 %(npages)s. Course project submission"; font: 9pt Cambria, 'Times New Roman', serif; }
               @top-right { content: counter(page); font: 9pt Cambria, 'Times New Roman', serif; }
               @top-center { content: none; } }
@page :right { @top-center { content: "%(short_title)s"; font: italic 9pt Cambria, 'Times New Roman', serif; }
               @top-right { content: counter(page); font: 9pt Cambria, 'Times New Roman', serif; } }
@page :left  { @top-center { content: "%(short_author)s"; font: italic 9pt Cambria, 'Times New Roman', serif; }
               @top-left { content: counter(page); font: 9pt Cambria, 'Times New Roman', serif; } }
html { font-family: Cambria, 'Times New Roman', serif; font-size: 11pt; line-height: 1.32; color: #000; }
body { margin: 0; text-align: justify; hyphens: auto; }
h1.title { font-style: italic; font-weight: normal; font-size: 21pt; line-height: 1.2; text-align: center; margin: 14mm 0 7mm; }
.authors { text-align: center; letter-spacing: 0.16em; text-transform: uppercase; font-size: 11pt; margin: 0 0 1.5mm; }
.affil { text-align: center; font-style: italic; font-size: 9pt; margin: 0 0 4mm; }
.received { text-align: center; font-style: italic; font-size: 10pt; margin: 0 0 7mm; }
.abstract { border-top: 0.6pt solid #000; border-bottom: 0.6pt solid #000; padding: 3mm 0 4mm; margin: 0 0 7mm; }
.abstract h2 { text-align: center; font-size: 10.5pt; margin: 0 0 2mm; }
.abstract p { font-size: 9.5pt; line-height: 1.3; margin: 0; text-indent: 0; }
h2.sec { text-align: center; font-size: 12pt; margin: 7mm 0 3mm; break-after: avoid; }
h3.sub { text-align: center; font-size: 11pt; font-style: italic; margin: 5mm 0 2mm; break-after: avoid; }
p { margin: 0; text-indent: 1.5em; orphans: 3; widows: 3; }
h2.sec + p, h3.sub + p, .noindent { text-indent: 0; }
ul { margin: 2mm 0 2mm 0; padding-left: 7mm; }
li { margin: 0 0 1mm; text-indent: 0; }
code { font-family: 'Courier New', monospace; font-size: 9.3pt; }
table.t { border-collapse: collapse; margin: 0 auto; width: 100%%; font-size: 9.5pt; }
table.t thead tr:first-child th { border-top: 3pt double #000; }
table.t th { padding: 1.4mm 2mm; text-align: left; border-bottom: 0.6pt solid #000; }
table.t td { padding: 0.9mm 2mm; vertical-align: top; text-align: left; }
table.t tr:last-child td { border-bottom: 3pt double #000; padding-bottom: 1.6mm; }
table.t th.r, table.t td.r { text-align: right; }
table.t code { font-size: 8pt; overflow-wrap: anywhere; }
table.t td, table.t th { overflow-wrap: anywhere; hyphens: manual; }
table.fixed { table-layout: fixed; }
.tabbox { margin: 4mm 0 5mm; break-inside: avoid; }
.cap { text-align: center; font-size: 9.5pt; margin: 0 0 2mm; text-indent: 0; }
.cap i { font-style: italic; }
.figbox { margin: 4mm 0 5mm; text-align: center; break-inside: avoid; }
.figbox img { max-width: 100%%; }
.figcap { font-size: 9.5pt; text-align: center; margin-top: 1.5mm; text-indent: 0; }
.refs { font-size: 9.5pt; line-height: 1.25; }
.refs p { text-indent: -6mm; padding-left: 6mm; margin: 0 0 1.4mm; }
"""


def render_html(doc, npages="?"):
    refs, num = doc["references"], Numbering([b for b in doc["blocks"]
                                             if b[0] in ("table", "fig")]).num
    ih = lambda t: inline_html(t, refs, num)  # noqa: E731
    css = HTML_CSS % dict(npages=npages, short_title=doc["short_title"].replace('"', ""),
                          short_author=doc["short_author"].replace('"', ""))
    authors = ", ".join(doc["authors"][:-1]) + " and " + doc["authors"][-1]
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(doc['title'])}"
           f"</title><style>{css}</style></head><body>",
           f"<h1 class='title'>{html.escape(doc['title'])}</h1>",
           f"<div class='authors'>{html.escape(authors)}</div>",
           f"<div class='affil'>{html.escape(doc['affiliation'])}</div>",
           f"<div class='received'>( {html.escape(doc['received'])} )</div>",
           f"<div class='abstract'><h2>Abstract</h2><p>{ih(doc['abstract'])}</p></div>"]
    sec = sub = 0
    for kind, data in doc["blocks"]:
        if kind == "h1":
            sec += 1
            sub = 0
            out.append(f"<h2 class='sec'>{sec}&nbsp;&nbsp;{ih(data)}</h2>")
        elif kind == "h1_star":
            out.append(f"<h2 class='sec'>{ih(data)}</h2>")
        elif kind == "h2":
            sub += 1
            out.append(f"<h3 class='sub'>{sec}.{sub}&nbsp;&nbsp;{ih(data)}</h3>")
        elif kind == "p":
            out.append(f"<p>{ih(data)}</p>")
        elif kind == "bullets_head":
            out.append("<p class='noindent'>What the table shows:</p>")
        elif kind == "ul":
            out.append("<ul>" + "".join(f"<li>{ih(i)}</li>" for i in data) + "</ul>")
        elif kind == "table":
            out.append(render_table_html(data, num, ih))
        elif kind == "fig":
            n = num["fig:" + data["label"]]
            out.append(f"<div class='figbox'><img src='../{data['file']}' style='width:{int(data['width'] * 100)}%'>"
                       f"<div class='figcap'>Fig.&nbsp;{n}. {ih(data['caption'])}</div></div>")
    out.append("<h2 class='sec'>References</h2><div class='refs'>")
    for key in sorted(refs, key=lambda k: refs[k][2].lower()):
        out.append(f"<p>{ih(refs[key][2])}</p>")
    out.append("</div></body></html>")
    return "\n".join(out)


def render_table_html(t, num, ih):
    n = num["tab:" + t["label"]]
    cols = parse_align(t["align"])
    widths = P_WIDTHS.get(t["align"])
    fixed = widths is not None
    head = "".join(f"<th class='{'r' if a == 'r' else ''}'>{ih(c)}</th>" for a, c in zip(cols, t["cols"]))
    rows = []
    for r in t["rows"]:
        cells = "".join(f"<td class='{'r' if a == 'r' else ''}'>{ih(centre_decimals(c))}</td>"
                        for a, c in zip(cols, r))
        rows.append(f"<tr>{cells}</tr>")
    colgroup = ("<colgroup>" + "".join(f"<col style='width:{int(w * 100)}%'>" for w in widths) + "</colgroup>"
                if fixed else "")
    return (f"<div class='tabbox'><div class='cap'>Table&nbsp;{n}. <i>{ih(t['caption'])}</i></div>"
            f"<table class='t{' fixed' if fixed else ''}'>{colgroup}<thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# ------------------------------------------------------------------ LaTeX
TEX_HEAD = r"""\documentclass[11pt,a4paper]{article}
%% LaTeX source of the report, laid out like the Natural Language Engineering template.
%% The official nle.sty is not used (it is not freely available); the layout is reproduced with
%% standard packages. Compile with: pdflatex report.tex (twice). Works on Overleaf.
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,left=33mm,right=33mm,top=30mm,bottom=30mm]{geometry}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{array}
\usepackage{caption}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\captionsetup[table]{labelsep=period,font=small,justification=centering,position=top}
\captionsetup[figure]{labelsep=period,font=small,justification=centering,name=Fig.}
\titleformat{\section}[block]{\centering\bfseries\large}{\thesection}{1em}{}
\titleformat{\subsection}[block]{\centering\bfseries\itshape}{\thesubsection}{1em}{}
\titleformat{name=\section,numberless}[block]{\centering\bfseries\large}{}{0pt}{}
\titlespacing*{\section}{0pt}{2.5ex plus 1ex}{1.5ex}
\titlespacing*{\subsection}{0pt}{2ex plus 1ex}{1ex}
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0pt}
\newcommand{\shorttitle}{%(short_title)s}
\newcommand{\shortauthor}{%(short_author)s}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[CO]{\small\itshape\shorttitle}
\fancyhead[RO]{\small\thepage}
\fancyhead[LE]{\small\thepage}
\fancyhead[CE]{\small\itshape\shortauthor}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}%%
  \fancyhead[L]{\small\itshape Natural Language Engineering \upshape 1 (1): 1--%(npages)s. Course project submission}%%
  \fancyhead[R]{\small\thepage}}
\begin{document}
\thispagestyle{plain}
\begin{center}
{\LARGE\itshape %(title)s\par}
\vspace{1.2em}
{\large\textsc{%(authors)s}\par}
\vspace{0.4em}
{\small\itshape %(affiliation)s\par}
\vspace{0.8em}
{\itshape ( %(received)s )\par}
\end{center}
\vspace{0.4em}\hrule\vspace{0.8em}
\begin{center}\textbf{Abstract}\end{center}\vspace{-0.4em}
{\small %(abstract)s\par}
\vspace{0.8em}\hrule\vspace{0.4em}
"""


def render_latex(doc, npages="?"):
    refs, num = doc["references"], Numbering([b for b in doc["blocks"]
                                             if b[0] in ("table", "fig")]).num
    it = lambda t: inline_tex(t, refs, num)  # noqa: E731
    authors = ", ".join(doc["authors"][:-1]) + " and " + doc["authors"][-1]
    out = [TEX_HEAD % dict(short_title=tex_text(doc["short_title"]), short_author=tex_text(doc["short_author"]),
                           npages=npages, title=tex_text(doc["title"]), authors=tex_text(authors),
                           affiliation=tex_text(doc["affiliation"]), received=tex_text(doc["received"]),
                           abstract=it(doc["abstract"]))]
    for kind, data in doc["blocks"]:
        if kind == "h1":
            out.append(f"\\section{{{it(data)}}}")
        elif kind == "h1_star":
            out.append(f"\\section*{{{it(data)}}}")
        elif kind == "h2":
            out.append(f"\\subsection{{{it(data)}}}")
        elif kind == "p":
            out.append(it(data) + "\n")
        elif kind == "bullets_head":
            out.append("\\noindent What the table shows:\n")
        elif kind == "ul":
            out.append("\\begin{itemize}\\setlength\\itemsep{0pt}\n" + "\n".join(f"\\item {it(i)}" for i in data)
                       + "\n\\end{itemize}")
        elif kind == "table":
            out.append(render_table_tex(data, it))
        elif kind == "fig":
            out.append(f"\\begin{{figure}}[t]\\centering\n\\includegraphics[width={data['width']}\\textwidth]"
                       f"{{{data['file']}}}\n\\caption{{{it(data['caption'])}}}\\label{{fig:{data['label']}}}\n"
                       "\\end{figure}")
    out.append("\\section*{References}\n\\begin{list}{}{\\leftmargin=1.5em\\itemindent=-1.5em\\itemsep=2pt\\parsep=0pt}\n\\small")
    for key in sorted(refs, key=lambda k: refs[k][2].lower()):
        out.append(f"\\item {it(refs[key][2])}")
    out.append("\\end{list}\n\\end{document}\n")
    return "\n".join(out)


def render_table_tex(t, it):
    cols = parse_align(t["align"])
    widths = P_WIDTHS.get(t["align"])
    if widths:
        spec = "".join(f">{{\\raggedright\\arraybackslash\\scriptsize}}p{{{w * 0.98:.2f}\\textwidth}}" for w in widths) \
            if len(widths) == 2 else \
            "".join(f">{{\\raggedright\\arraybackslash\\scriptsize}}p{{{w * 0.94:.2f}\\textwidth}}" for w in widths)
        font = ""
    else:
        spec = "".join(cols)
        font = "\\small"
    head = " & ".join(f"\\textbf{{{it(c)}}}" for c in t["cols"])
    rows = " \\\\\n".join(" & ".join(it(centre_decimals(c)) for c in r) for r in t["rows"])
    return (f"\\begin{{table}}[t]\n\\caption{{\\textit{{{it(t['caption'])}}}}}\\label{{tab:{t['label']}}}\n"
            f"\\centering{font}\n\\begin{{tabular}}{{{spec}}}\n\\hline\\hline\n{head} \\\\\n\\hline\n{rows} \\\\\n"
            "\\hline\\hline\n\\end{tabular}\n\\end{table}")


# ------------------------------------------------------------------ PDF
def find_browser():
    for p in ("C:/Program Files/Google/Chrome/Application/chrome.exe",
              "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
              "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"):
        if Path(p).exists():
            return p
    return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")


def html_to_pdf(html_path, pdf_path):
    browser = find_browser()
    if not browser:
        print("No Chrome/Edge found: skipped the PDF (report.tex is still written).")
        return False
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    return pdf_path.exists()


def count_pages(pdf_path):
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def main():
    make_figures()
    doc = content.build()
    BUILD.mkdir(exist_ok=True)
    html_path, pdf_path = BUILD / "report.html", HERE / "report.pdf"
    html_path.write_text(render_html(doc), encoding="utf-8")
    npages = "?"
    if html_to_pdf(html_path, pdf_path):
        npages = str(count_pages(pdf_path))
        html_path.write_text(render_html(doc, npages), encoding="utf-8")  # second pass: page count
        html_to_pdf(html_path, pdf_path)
        print(f"report.pdf: {count_pages(pdf_path)} pages")
    (HERE / "report.tex").write_text(render_latex(doc, npages), encoding="utf-8")
    print("wrote docs/report/report.tex")


if __name__ == "__main__":
    main()
