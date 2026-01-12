%% Minimal stub of CEURART class for local compilation in environments
%% where the official ceurart.cls is not installed.
%% For final submission, please use the official CEUR-WS ceurart.cls.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ceurart}[2025/01/01 Minimal stub CEURART class]

\LoadClass[11pt]{article}

\RequirePackage{authblk}
\RequirePackage{geometry}
\RequirePackage{hyperref}
\RequirePackage{xcolor}

\geometry{margin=1in}

%% Store conference and copyright meta (optional)
\newcommand{\conference}[1]{\gdef\ceur@conference{#1}}
\newcommand{\copyrightyear}[1]{\gdef\ceur@copyrightyear{#1}}
\newcommand{\copyrightclause}[1]{\gdef\ceur@copyrightclause{#1}}
\newcommand{\ceur@conference}{}
\newcommand{\ceur@copyrightyear}{}
\newcommand{\ceur@copyrightclause}{}

%% CEURART uses \address; map it to authblk \affil.
\newcommand{\address}[2][]{\affil[#1]{#2}}

%% Corresponding author helpers (no-op in stub)
\newcommand{\cormark}[1][]{}
\newcommand{\cortext}[2][]{}

%% Keywords environment and separator
\newcommand{\sep}{; }
\newenvironment{keywords}{\par\noindent\textbf{Keywords:} }{\par}

%% Override \maketitle to show basic CEUR meta
\renewcommand{\maketitle}{%
  \begin{center}
    {\Large\bfseries \@title\par}
    \vspace{0.75em}
    {\normalsize
      \begin{tabular}{c}
      \@author
      \end{tabular}\par}
    \vspace{0.75em}
    {\small \ceur@conference\par}
    \vspace{0.25em}
    {\small \ceur@copyrightclause\par}
  \end{center}
  \vspace{1em}
}
