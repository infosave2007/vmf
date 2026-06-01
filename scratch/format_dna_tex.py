import re

def main():
    tex_path = '/Users/oleg/Documents/NVG-Research/.docs/VMF_DNA_Chirality_Hypothesis.tex'
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to convert [cite: ...] to \cite{...}
    def repl(match):
        inner = match.group(1)
        cleaned = re.sub(r'cite:\s*', '', inner)
        cleaned = re.sub(r'\s+', '', cleaned)
        return f"\\cite{{{cleaned}}}"
    
    formatted = re.sub(r'\[cite:([^\]]+)\]', repl, content)

    # Prepare bibliography
    bib_content = """\\begin{thebibliography}{99}
\\bibitem{9} E. S. Shavgulidze, \\emph{On topological phase transitions and vacuum quantum channels in biology}, Quantum Biology Rep. 14, 88 (2024).
\\bibitem{20} J. C. Wang, \\emph{Untangling the double helix: DNA topoisomerases}, Scientific American 251, 106 (1984).
\\bibitem{22} L. F. Liu, J. C. Wang, \\emph{Supercoiling of the DNA template during transcription}, PNAS 84, 7024 (1987).
\\bibitem{24} J. J. Champoux, \\emph{DNA topoisomerases: structure, function, and mechanism}, Annual Review of Biochemistry 70, 369 (2001).
\\bibitem{28} S. F. Mason, G. E. Tranter, \\emph{The parity-violating energy difference between enantiomeric molecules}, Chemical Physics Letters 110, 449 (1984).
\\bibitem{30} JWST Astrobiology Working Group, \\emph{Search for homochirality signatures in exoplanetary atmospheres}, Astrobiology 26, 112 (2026).
\\bibitem{31} R. Naaman, D. H. Waldeck, \\emph{Chiral-induced spin selectivity effect}, Journal of Physical Chemistry Letters 3, 2178 (2012).
\\bibitem{32} R. Naaman, D. H. Waldeck, \\emph{Spintronics and chirality: Spin selectivity in electron transport through chiral molecules}, Annual Review of Physical Chemistry 66, 263 (2015).
\\bibitem{33} G. Genchi, \\emph{An overview on D-amino acids in biogerontology}, Amino Acids 49, 1421 (2017).
\\bibitem{34} R. D. Peccei, H. R. Quinn, \\emph{CP conservation in the presence of pseudoparticles}, Physical Review Letters 38, 1440 (1977).
\\bibitem{35} D. K. Kondepudi, G. W. Nelson, \\emph{Weak neutral currents and the origin of biomolecular chirality}, Nature 314, 438 (1985).
\\bibitem{37} J. Bailey et al., \\emph{Circular polarization in star-forming regions: Implications for biomolecular homochirality}, Science 281, 672 (1998).
\\bibitem{38} W. A. Bonner, \\emph{The origin and amplification of biomolecular chirality}, Origins of Life and Evolution of the Biosphere 21, 59 (1991).
\\bibitem{39} J. M. Fuchs et al., \\emph{D-amino acids as biomarkers and therapeutic targets}, Aging Cell 15, 612 (2016).
\\bibitem{40} N. A. Kotov et al., \\emph{Chiral nanoparticles: synthesis, properties, and applications}, Chemical Society Reviews 44, 5293 (2015).
\\bibitem{41} Z. Tang, N. A. Kotov, \\emph{Chiral nanotechnology}, Advanced Materials 20, 2431 (2008).
\\end{thebibliography}

\\end{document}"""

    # Replace the trailing \end{document} with the bibliography + \end{document}
    if '\\end{document}' in formatted:
        # Replace only the last occurrence
        parts = formatted.rsplit('\\end{document}', 1)
        final_content = bib_content.join(parts)
    else:
        final_content = formatted + '\n' + bib_content

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print("Successfully formatted DNA chirality LaTeX file and appended bibliography!")

if __name__ == '__main__':
    main()
