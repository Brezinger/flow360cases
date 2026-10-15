\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{achleitner-report}[2026/10/08 Achleitner Aerospace corporate technical report]
\newif\ifreportenglish
\DeclareOption{english}{\reportenglishtrue}
\DeclareOption{german}{\reportenglishfalse}
\ExecuteOptions{english}\ProcessOptions\relax
\LoadClass[a4paper,10pt,oneside]{article}
\RequirePackage{iftex}
\ifPDFTeX\ClassError{achleitner-report}{Use XeLaTeX or LuaLaTeX}{Open Sans requires fontspec.}\fi
\RequirePackage{fontspec}
\setmainfont{OpenSans}[Path=fonts/,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]
\setsansfont{OpenSans}[Path=fonts/,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]
\RequirePackage{amsmath,unicode-math}\setmathfont{Latin Modern Math}
\ifreportenglish\RequirePackage[english]{babel}\else\RequirePackage[provide=*,ngerman]{babel}\fi
\RequirePackage[left=25mm,right=25mm,top=31mm,bottom=25mm,headheight=30pt,headsep=9mm,footskip=12mm]{geometry}
\RequirePackage[table]{xcolor}
% Exact RGB values from the supplied Achleitner Aerospace colour guide.
\definecolor{reportblue}{HTML}{006B91}
\definecolor{reportgrey}{HTML}{E3E3E3}
\definecolor{reportmidgrey}{HTML}{B4B4B4}
\definecolor{reportdarkgrey}{HTML}{646464}
\definecolor{reportink}{HTML}{292929}
\RequirePackage{graphicx,eso-pic,tikz,fancyhdr,titlesec,caption,tabularx,longtable,array,enumitem,tocloft}
\usetikzlibrary{calc}
\RequirePackage[hidelinks,unicode]{hyperref}
\newcommand{\ifsreport}[2]{\ifreportenglish#1\else#2\fi}
\newcommand{\reporttitle}{Technical report}
\newcommand{\reportshorttitle}{\reporttitle}
\newcommand{\reporttype}{\ifsreport{Technical report}{Technischer Bericht}}
\newcommand{\reportid}{AA-YYYY-NNN}
\newcommand{\reportrevision}{A}
\newcommand{\reportdate}{08 October 2026}
\newcommand{\reportauthor}{Dr.-Ing. J. Achleitner}
\newcommand{\reportcompany}{Achleitner Aerospace GmbH}
\newcommand{\reportstatus}{\ifsreport{Draft}{Entwurf}}
\newcommand{\reportclassification}{\ifsreport{Confidential}{Vertraulich}}
\newcommand{\reportreviewer}{\ifsreport{Not yet reviewed}{Noch nicht geprüft}}
\newcommand{\reportapprover}{\ifsreport{Not yet approved}{Noch nicht freigegeben}}
\newcommand{\reportreviewdate}{--}
\newcommand{\reportapprovaldate}{--}
\newcommand{\reportproject}{[Project / customer]}
\newcommand{\reportcoverimage}{assets/cover-aircraft.png}
\newcommand{\reportconfiguration}{[Configuration]}
\newcommand{\reportoperatingpoint}{[Operating point]}
\newcommand{\reportmethod}{[Method]}
\newcommand{\reportrecipients}{[Recipients]}
\newcommand{\reportpurpose}{[Purpose]}
\newcommand{\reportwatermarkopacity}{0.035}
\newcommand{\reportchangelog}{\reportrevision & \reportdate & \ifsreport{Initial draft}{Erster Entwurf} & \ifsreport{All}{Alle} & J. Achleitner\\\hline}
\newif\ifreportcover
\AddToShipoutPictureBG{\begin{tikzpicture}[remember picture,overlay]
 \ifreportcover\else
 \node[opacity=\reportwatermarkopacity,inner sep=0pt,anchor=south east] at ($(current page.south east)+(20mm,20mm)$) {\includegraphics[width=170mm]{assets/signet.png}};
 \fi
\end{tikzpicture}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\includegraphics[width=42mm]{assets/logo-horizontal.png}}
\fancyhead[R]{\parbox[b]{105mm}{\raggedleft\fontsize{8}{10}\selectfont\reportshorttitle\\\color{reportdarkgrey}\reportid\enspace|\enspace\ifsreport{Rev.}{Rev.}~\reportrevision}}
\fancyfoot[L]{\fontsize{7.5}{10}\selectfont\color{reportdarkgrey}\reportclassification\enspace|\enspace\reportstatus}
\fancyfoot[R]{\fontsize{8}{10}\selectfont\color{reportdarkgrey}\reportdate\enspace|\enspace\thepage}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0.3pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{reportmidgrey}\leaders\hrule height\headrulewidth\hfill}}
\renewcommand{\footrule}{\hbox to\headwidth{\color{reportmidgrey}\leaders\hrule height\footrulewidth\hfill}}
\AtBeginDocument{\color{reportink}\hypersetup{pdftitle={\reporttitle},pdfauthor={\reportauthor},pdfsubject={\reporttype}}}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt plus 1pt minus 1pt}
\renewcommand{\baselinestretch}{1.12}\raggedbottom
\setlength{\emergencystretch}{3em}
\titleformat{\section}{\Large\bfseries\color{reportblue}}{\thesection\enspace|}{.5em}{}
\titleformat{\subsection}{\large\bfseries\color{reportblue}}{\thesubsection\enspace|}{.5em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{reportblue}}{\thesubsubsection\enspace|}{.5em}{}
\titlespacing*{\section}{0pt}{16pt}{8pt}\titlespacing*{\subsection}{0pt}{12pt}{6pt}\titlespacing*{\subsubsection}{0pt}{10pt}{5pt}
\setcounter{secnumdepth}{3}\setcounter{tocdepth}{3}
\DeclareCaptionFont{reportcaption}{\fontsize{9}{12}\selectfont\color{reportdarkgrey}}
\captionsetup{font=reportcaption,labelfont={bf,color=reportblue},labelsep=colon,justification=centering,singlelinecheck=false,skip=8pt}
\captionsetup[table]{position=top}
\setlist{leftmargin=7mm,labelsep=3mm,itemsep=3pt,topsep=4pt,parsep=0pt}
\renewcommand{\arraystretch}{1.5}\setlength{\tabcolsep}{5pt}
\renewcommand{\tabularxcolumn}[1]{m{#1}}\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\reporttablehead}{\rowcolor{reportblue}}
\newcommand{\headcell}[1]{\textcolor{white}{\textbf{#1}}}
\newenvironment{reporttable}[1]{\rowcolors{2}{white}{reportgrey!45}\tabularx{\linewidth}{#1}\hline}{\endtabularx}
\newcommand{\reportfronttitle}[1]{{\Large\bfseries\color{reportblue}#1\par}\vspace{10pt}}
\renewcommand{\cfttoctitlefont}{\Large\bfseries\color{reportblue}}
\renewcommand{\cftloftitlefont}{\Large\bfseries\color{reportblue}}
\renewcommand{\cftlottitlefont}{\Large\bfseries\color{reportblue}}
\tocloftpagestyle{fancy}
\setlength{\cftbeforetoctitleskip}{0pt}\setlength{\cftbeforeloftitleskip}{0pt}\setlength{\cftbeforelottitleskip}{0pt}
\setlength{\cftaftertoctitleskip}{18pt}\setlength{\cftafterloftitleskip}{18pt}\setlength{\cftafterlottitleskip}{18pt}
\renewcommand{\cftsecfont}{\normalfont}\renewcommand{\cftsecpagefont}{\normalfont}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftsecnumwidth}{8mm}\setlength{\cftsubsecindent}{8mm}\setlength{\cftsubsecnumwidth}{12mm}
\setlength{\cftsubsubsecindent}{20mm}\setlength{\cftsubsubsecnumwidth}{15mm}
% Alphabetically sorted nomenclature, implemented without external indexing tools.
\ExplSyntaxOn
\seq_new:N \g_report_abbr_seq
\seq_new:N \g_report_symbol_seq
\prop_new:N \g_report_names_prop
\prop_new:N \g_report_desc_prop
\prop_new:N \g_report_sort_prop
\prop_new:N \g_report_unit_prop
\prop_new:N \g_report_used_prop
\NewDocumentCommand{\newabbreviation}{mmm}{
 \seq_gput_right:Nn \g_report_abbr_seq {#1}
 \prop_gput:Nnn \g_report_names_prop {#1}{#2}
 \prop_gput:Nnn \g_report_desc_prop {#1}{#3}
 \prop_gput:Nnn \g_report_sort_prop {#1}{#2}
}
\NewDocumentCommand{\gls}{m}{
 \prop_if_in:NnTF \g_report_used_prop {#1}
 {\prop_item:Nn \g_report_names_prop {#1}}
 {\prop_item:Nn \g_report_desc_prop {#1}~(\prop_item:Nn \g_report_names_prop {#1})\prop_gput:Nnn \g_report_used_prop {#1}{1}}
}
\NewDocumentCommand{\abbr}{m}{\prop_item:Nn \g_report_names_prop {#1}}
\NewDocumentCommand{\sym}{m}{\prop_item:Nn \g_report_names_prop {#1}}
\NewDocumentCommand{\reportsymbol}{mmmmm}{
 \seq_gput_right:Nn \g_report_symbol_seq {#1}
 \prop_gput:Nnn \g_report_names_prop {#1}{\ensuremath{#2}}
 \prop_gput:Nnn \g_report_sort_prop {#1}{#3}
 \prop_gput:Nnn \g_report_desc_prop {#1}{#4}
 \prop_gput:Nnn \g_report_unit_prop {#1}{#5}
}
\cs_new_protected:Npn \report_sort:N #1 {
 \seq_sort:Nn #1 {
  \str_compare:eNeTF {\str_lowercase:f{\prop_item:Nn \g_report_sort_prop {##1}}} > {\str_lowercase:f{\prop_item:Nn \g_report_sort_prop {##2}}}
  {\sort_return_swapped:}{\sort_return_same:}
 }
}
\NewDocumentCommand{\reportabbreviationrows}{}{
 \report_sort:N \g_report_abbr_seq
 \seq_map_inline:Nn \g_report_abbr_seq {\prop_item:Nn \g_report_names_prop {##1} & \prop_item:Nn \g_report_desc_prop {##1} \\}
}
\NewDocumentCommand{\reportsymbolrows}{}{
 \report_sort:N \g_report_symbol_seq
 \seq_map_inline:Nn \g_report_symbol_seq {\prop_item:Nn \g_report_names_prop {##1} & \prop_item:Nn \g_report_desc_prop {##1} & \prop_item:Nn \g_report_unit_prop {##1} \\}
}
\ExplSyntaxOff
\newcommand{\reportabbreviationlist}{%
 \begingroup\renewcommand{\theHtable}{abbreviations.\arabic{table}}
 \begin{longtable}{@{}p{30mm}p{125mm}@{}}%
 \reporttablehead\headcell{\ifsreport{Abbreviation}{Abkürzung}} & \headcell{\ifsreport{Meaning}{Bedeutung}}\\\endhead
 \reportabbreviationrows\end{longtable}\addtocounter{table}{-1}\endgroup}
\newcommand{\reportsymbollist}{%
 \begingroup\renewcommand{\theHtable}{symbols.\arabic{table}}
 \begin{longtable}{@{}p{25mm}p{100mm}p{25mm}@{}}%
 \reporttablehead\headcell{Symbol} & \headcell{\ifsreport{Meaning}{Bedeutung}} & \headcell{\ifsreport{Unit}{Einheit}}\\\endhead
 \reportsymbolrows\end{longtable}\addtocounter{table}{-1}\endgroup}
\newcommand{\makereportcover}{%
 \clearpage\pagenumbering{roman}\reportcovertrue\thispagestyle{empty}%
 \begin{tikzpicture}[remember picture,overlay]
 \node[opacity=.055,anchor=north east,inner sep=0pt] at ($(current page.north east)+(20mm,-6mm)$) {\includegraphics[width=175mm]{assets/signet.png}};
 \node[anchor=north west,inner sep=0pt] at ($(current page.north west)+(25mm,-22mm)$) {\includegraphics[width=110mm]{assets/logo-horizontal.png}};
 \draw[reportblue,line width=1.5pt] ($(current page.north west)+(25mm,-56mm)$) -- ($(current page.north west)+(25mm,-137mm)$);
 \node[anchor=north west,inner sep=0pt,text width=146mm] at ($(current page.north west)+(31mm,-58mm)$) {\color{reportblue}\small\bfseries\MakeUppercase{\reporttype}\par\vspace{7mm}{\color{reportink}\fontsize{27}{34}\selectfont\bfseries\reporttitle\par}\vspace{5mm}{\color{reportdarkgrey}\normalsize\reportproject\par}};
 \node[anchor=north west,inner sep=0pt,text width=146mm] at ($(current page.north west)+(31mm,-145mm)$) {\small\reportid\enspace|\enspace\ifsreport{Revision}{Revision}~\reportrevision\enspace|\enspace\reportstatus\par\vspace{3mm}\reportauthor\par\reportdate};
 % Eindhoven-inspired angled image band; no university artwork is used.
 \begin{scope}
 \clip ($(current page.south west)+(0,110mm)$) -- ($(current page.south east)+(0,83mm)$) -- (current page.south east) -- (current page.south west) -- cycle;
 \fill[reportgrey!45] (current page.south west) rectangle ($(current page.south east)+(0,112mm)$);
 \ifx\reportcoverimage\empty\else
 \node[anchor=center,inner sep=0pt] at ($(current page.south)+(0,55mm)$) {\includegraphics[width=194mm,height=78mm,keepaspectratio]{\reportcoverimage}};
 \fi
 \end{scope}
 \draw[reportblue,line width=1.3pt] ($(current page.south west)+(0,110mm)$) -- ($(current page.south east)+(0,83mm)$);
 \fill[reportblue] (current page.south west) rectangle ($(current page.south east)+(0,13mm)$);
 \node[anchor=west,text=white,inner sep=0pt] at ($(current page.south west)+(25mm,6.5mm)$) {\fontsize{8}{10}\selectfont\reportcompany};
 \node[anchor=east,text=white,inner sep=0pt] at ($(current page.south east)+(-25mm,6.5mm)$) {\fontsize{8}{10}\selectfont\reportclassification};
 \end{tikzpicture}\null\clearpage\reportcoverfalse}
\newcommand{\reportdocumentcontrol}{%
 \reportfronttitle{\ifsreport{Document control}{Dokumenteninformation}}\phantomsection\addcontentsline{toc}{section}{\ifsreport{Document control}{Dokumenteninformation}}
 \begingroup\small
 \begin{reporttable}{|m{38mm}|Y|}\reporttablehead\headcell{\ifsreport{Field}{Merkmal}} & \headcell{\ifsreport{Document information}{Dokumenteninformation}}\\\hline
 \ifsreport{Document ID}{Dokumentnummer} & \reportid\\\hline
 \ifsreport{Title}{Titel} & \reporttitle\\\hline
 \ifsreport{Project / customer}{Projekt / Kunde} & \reportproject\\\hline
 Revision / \ifsreport{issue date}{Ausgabedatum} & \reportrevision\ / \reportdate\\\hline
 \ifsreport{Status / classification}{Status / Vertraulichkeit} & \reportstatus\ / \reportclassification\\\hline
 \ifsreport{Prepared by}{Erstellt von} & \reportauthor, \reportcompany\\\hline
 \ifsreport{Reviewed by / date}{Geprüft von / Datum} & \reportreviewer\ / \reportreviewdate\\\hline
 \ifsreport{Approved by / date}{Freigegeben von / Datum} & \reportapprover\ / \reportapprovaldate\\\hline
 \ifsreport{Distribution}{Verteiler} & \reportrecipients\\\hline
 \ifsreport{Configuration}{Konfiguration} & \reportconfiguration\\\hline
 \ifsreport{Operating point}{Betriebspunkt} & \reportoperatingpoint\\\hline
 \ifsreport{Method}{Methode} & \reportmethod\\\hline
 \ifsreport{Purpose and scope}{Ziel und Umfang} & \reportpurpose\\\hline
 \end{reporttable}
 \par\vspace{5mm}{\footnotesize\color{reportdarkgrey}\ifsreport{Review and approval entries must reflect the actual release status.}{Prüf- und Freigabeeinträge müssen dem tatsächlichen Freigabestatus entsprechen.}\par}
 \endgroup\clearpage}
\newcommand{\reportrevisionhistory}{%
 \reportfronttitle{\ifsreport{Change log}{Änderungsübersicht}}\phantomsection\addcontentsline{toc}{section}{\ifsreport{Change log}{Änderungsübersicht}}
 \begingroup\footnotesize\setlength{\tabcolsep}{4pt}
 \begin{longtable}{|p{10mm}|p{25mm}|p{65mm}|p{14mm}|p{30mm}|}\hline
 \reporttablehead\headcell{Rev.} & \headcell{\ifsreport{Date}{Datum}} & \headcell{\ifsreport{Description of change}{Beschreibung der Änderung}} & \headcell{\ifsreport{Pages}{Seiten}} & \headcell{\ifsreport{Author}{Verfasser}}\\\hline\endhead
 \reportchangelog
 \end{longtable}\addtocounter{table}{-1}\endgroup\clearpage}
\newcommand{\reportfrontmatter}{%
 \reportdocumentcontrol\reportrevisionhistory\tableofcontents\clearpage
 \phantomsection\addcontentsline{toc}{section}{\ifsreport{List of figures}{Abbildungsverzeichnis}}\listoffigures\clearpage
 \reportfronttitle{\ifsreport{List of abbreviations}{Abkürzungsverzeichnis}}\phantomsection\addcontentsline{toc}{section}{\ifsreport{List of abbreviations}{Abkürzungsverzeichnis}}\reportabbreviationlist\clearpage
 \reportfronttitle{\ifsreport{List of symbols}{Symbolverzeichnis}}\phantomsection\addcontentsline{toc}{section}{\ifsreport{List of symbols}{Symbolverzeichnis}}\reportsymbollist\clearpage
 \pagenumbering{arabic}}
\endinput
