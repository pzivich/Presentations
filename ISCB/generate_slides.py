import numpy as np
import pandas as pd
from pylatex import (Document, Section, Subsection, Package, Command,
                     Figure, NoEscape, LineBreak, Itemize, Enumerate)

tex_file_name = "Zivich_Python_ISCB43"


class Frame:
    def __init__(self, title):
        doc.append(NoEscape(r'\begin{frame}{'+str(title)+'}'))

    def append(self, text):
        doc.append(text)

    def end(self):
        doc.append(NoEscape(r'\end{frame}'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


def center(document, text):
    document.append(NoEscape('{'))
    document.append(Command('centering'))
    document.append(text)
    document.append(NoEscape('}'))


#####################
# TeX setup
doc = Document(tex_file_name, documentclass="beamer")

doc.preamble.append(Command('usetheme', 'Copenhagen'))
doc.preamble.append(Command('usecolortheme', 'whale'))

doc.packages.append(Package('amsmath'))
doc.packages.append(Package('xcolor'))
doc.packages.append(Package('graphicx'))
doc.packages.append(Package('fontawesome5'))
doc.packages.append(Package('pythonhighlight'))

doc.preamble.append(Command('usefonttheme', 'serif', 'onlymath'))

doc.preamble.append(Command('setbeamercovered', 'transparent'))
doc.preamble.append(Command('setbeamertemplate', 'navigation symbols',
                            extra_arguments=" "))
doc.preamble.append(Command('setbeamertemplate', 'page number in head/foot',
                            extra_arguments=NoEscape(r'\insertframenumber')))
doc.preamble.append(Command('setbeamertemplate', 'headline',
                            extra_arguments=" "))

doc.preamble.append(Command('title', NoEscape(r'\huge Why I Use Python \\'
                                              r'\large (and Why You Should Too)'),
                            'Why I Use Python'))
doc.preamble.append(Command('author', NoEscape(r'Paul Zivich \\~\\ Institute of Global Health and Infectious Diseases\\'
                                               r'Causal Inference Research Laboratory \\'
                                               r'University of North Carolina at Chapel Hill'),
                            'Paul Zivich'))
doc.preamble.append(Command('date', NoEscape(r'August 25, 2022')))

################################
# Title

doc.append(NoEscape(r'\maketitle'))

################################
# Acknowledgements

with Frame(title="Acknowledgements") as f:
    f.append(NoEscape(r"Supported by NIH T32-AI007001."
                      r"\footnote[frame]{Footnotes are reserved asides for possible later discussion or questions}"
                      r"\\~\\~\\"))
    f.append(NoEscape(r"Python: \faPython \\~\\~\\"))
    center(document=f,
           text=NoEscape(r"\faEnvelope \quad pzivich@unc.edu \qquad	"
                         r"\faTwitter \quad @PausalZ \qquad"
                         r"\faGithub \quad pzivich\\"))
    f.append(NoEscape(r"~\\~\\ Slides and code at https://github.com/pzivich/Presentations"))

################################
# Outline

with Frame(title="Outline") as f:
    f.append(NoEscape(r"My background \\~\\"))
    f.append(NoEscape(r"Value add of \faPython\footnote[frame]{I am going to pick on R, please save angry emails till "
                      r"after the presentation} \\~\\"))
    f.append(NoEscape(r"Illustrative applications \\~\\"))
    f.append(NoEscape(r"Installation and Conclusions"))

################################
# My background

with Frame(title="About me") as f:
    f.append("An epidemiologist working in methods and infectious diseases")
    f.append(NoEscape(r"\\~\\"))
    f.append(NoEscape(r"Using \faPython\; since 2016"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item("Largely self-taught")
    f.append(NoEscape(r"~\\"))
    f.append(NoEscape("Active contributor"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"\texttt{zEpid}, "
                                  r"\texttt{delicatessen}\footnote[frame]{Zivich PN, et al. (2022) Delicatessen: "
                                  r"M-Estimation in Python. \textit{arXiv:2203.11300}}, "
                                  r"\texttt{MossSpider}\footnote[frame]{Zivich PN, et al. (2022) Targeted maximum "
                                  r"likelihood estimation of causal effects with interference: A simulation study. "
                                  r"\textit{Statistics in Medicine}}"))
        itemize.add_item(NoEscape(r"\texttt{lifelines}"))
    f.append(NoEscape(r"~\\"))
    f.append(NoEscape(r"\faPython\; is my primary software"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item("Also use R, SAS")

################################
# Claim

with Frame(title="A Software Philosophy") as f:
    f.append("To be a good epidemiologist / biostatistician / data scientist / someone who works with "
             "data, familiarity with multiple software languages is important")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("No software is complete for all tasks")
        itemize.add_item(NoEscape(r"\textit{Lingua franca} of fields will change"))
        itemize.add_item("Harder to be replaced")
    f.append(NoEscape(r"~\\ Why \faPython\; should be added to your repertoire"))

################################
# Python background

with Frame(title=NoEscape(r"What is \faPython ?")) as f:
    f.append("High-level programming language")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Interpreted"))
        itemize.add_item(NoEscape(r"Object-oriented"))
        itemize.add_item(NoEscape(r"Free, open-source"))
        itemize.add_item(NoEscape(r"Supported for all major platforms"))
        itemize.add_item(NoEscape(r"Scales to available hardware"))
    f.append(NoEscape(r"~\\Some advantages from my perspective"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Language features"))
        itemize.add_item(NoEscape(r"Cross-software interactions"))
        itemize.add_item(NoEscape(r"Popularity"))

################################
# Advantages of Specifics

with Frame(title=NoEscape(r"Advantage: language-specific features")) as f:
    f.append(NoEscape(r"Class objects \\~\\"))
    f.append(NoEscape(r"Namespaces and modules \\~\\"))
    f.append(NoEscape(r"Readability \\~\\"))
    f.append(NoEscape(r"Accuracy"))

with Frame(title=r"Class objects") as f:
    f.append("Object that hold")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("Functions, other objects")
        itemize.add_item("Each function can have unique parameters and docs")
        itemize.add_item("Store hidden parameters for testing")
    f.append(NoEscape(r"~\\"))
    f.append(NoEscape(r"\inputpython{generate_slides.py}{9}{15}"))

with Frame(title=NoEscape(r"Class objects")) as f:
    f.append(NoEscape(r"\includegraphics[width=0.9\linewidth]{images/r_tmle.PNG}"))

with Frame(title=r"Class objects") as f:
    f.append(NoEscape(r"\inputpython{stat_examples.py}{6}{13}"))

with Frame(title=r"Namespace of modules") as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/R-namespace.png}"
                      r"\end{center}"))
    f.append(NoEscape(r"~\\"))
    f.append(NoEscape(r"R's namespace conflicts in my work"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Network analysis in R: \texttt{sna}, \texttt{igraph}"))
        itemize.add_item(NoEscape(r"Non-overlapping functionalities"))
        itemize.add_item(NoEscape(r"Overlapping functionalities have conflicts"))

with Frame(title=r"Namespace of modules") as f:
    f.append(NoEscape(r"Not a problem in \faPython \\~\\"))
    f.append(NoEscape(r"\inputpython{numpy_example.py}{1}{5}"))

with Frame(title=NoEscape(r"Readability")) as f:
    f.append(NoEscape(r"\includegraphics[width=0.9\linewidth]{images/R_bad-loop.PNG}"))

with Frame(title=r"Readability") as f:
    f.append(NoEscape(r"\inputpython{loop_example.py}{1}{14}"))

with Frame(title=NoEscape(r"Accuracy")) as f:
    f.append(NoEscape(r"\includegraphics[width=1.0\linewidth]{images/r_floating_point.png}"))

with Frame(title=NoEscape(r"Accuracy"
                          r"\footnote[frame]{Julia also presents this correctly}")) as f:
    f.append(NoEscape(r"\includegraphics[width=1.0\linewidth]{images/python_floating_point.png}"))

################################
# Ability to Interact

with Frame(title=r"Advantage: cross-software interactions") as f:
    f.append("Python is a good glue language")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Easily interacts with other software"))
        with doc.create(Itemize()) as itemize_inner:
            itemize_inner.add_item(NoEscape(r"\texttt{C}, \texttt{C++}"))
        itemize.add_item(NoEscape(r"Interact with other software:"))
        with doc.create(Itemize()) as itemize_inner:
            itemize_inner.add_item(NoEscape(r"\texttt{R}: \texttt{RPy2}"))
            itemize_inner.add_item(NoEscape(r"\texttt{Stan}: \texttt{PyStan}"))
            itemize_inner.add_item(NoEscape(r"\texttt{Julia}: \texttt{PyJulia}"))
            itemize_inner.add_item(NoEscape(r"\texttt{SAS}: \texttt{SASPy}"))


with Frame(title="Advantage: cross-software interactions") as f:
    f.append(NoEscape(r"All slides made with \faPython\; and \LaTeX"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Using \texttt{pylatex}"))
    f.append(NoEscape(r"\inputpython{generate_slides.py}{33}{44}"))

################################
# Popularity across areas

with Frame(title=NoEscape(r"Advantage: popularity")) as f:
    f.append(NoEscape(r"\includegraphics[width=1.0\linewidth]{images/python_trends.png}"))


with Frame(title=NoEscape(r"Advantage: popularity")) as f:
    f.append("Combination of:")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("Programmers")
        itemize.add_item("Scientists")
        itemize.add_item("Statisticians")
    f.append(NoEscape(r"~\\"))
    f.append("Wide support for use-cases")

with Frame(title=NoEscape(r"Example: Black Holes")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.55\linewidth]{images/sagittarius-a.jpg}"
                      r"\end{center}"))
    f.append(NoEscape(r"Computations and image processing done using \faPython"
                      r"\footnote[frame]{Akiyama K, et al. (2022) First Sagittarius A* Event Horizon Telescope "
                      r"Results. I. The Shadow of the Supermassive Black Hole in the Center of the Milky Way. \textit{"
                      r"The Astrophysical Journal Letters} 930.2 (2022): L12}"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item("The telescope array generates >350 terabytes of data daily")

# with Frame(title=NoEscape(r"Example: Image Generation")) as f:
#     f.append("DALLE-2: image generation from text descriptions")
#     f.append(NoEscape(r"\begin{center}"
#                       r"\includegraphics[width=0.8\linewidth]{images/dalle2.png}"
#                       r"\end{center}"))

# with Frame(title=NoEscape(r"Example: Image Generation"
#                           r"\footnote[frame]{https://github.com/huggingface/diffusers/releases/tag/v0.2.3}")) as f:
#     f.append(NoEscape(r"\begin{center}"
#                       r"\includegraphics[width=0.75\linewidth]{images/python_dalle2.PNG}"
#                       r"\end{center}"))

################################
# Examples

with Frame(title="Illustrative Applications") as f:
    f.append("Examples")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Basic statistical applications"))
        itemize.add_item(NoEscape(r"Plasmode data simulation with GANs"))
        itemize.add_item(NoEscape(r"Scientific abstract text generator"))

# Basics
with Frame(title="Basics: Regression") as f:
    f.append(NoEscape(r"\inputpython{stat_examples.py}{15}{22}"))

with Frame(title="Basics: Inverse Probability Weighting") as f:
    f.append(NoEscape(r"\inputpython{stat_examples.py}{24}{36}"))
    f.append(NoEscape(r"\footnote[frame]{Can also be done using \texttt{zEpid}}"))

with Frame(title="Basics: Survival Analysis") as f:
    f.append(NoEscape(r"\inputpython{stat_examples.py}{38}{46}"))

# GAN for simulations
with Frame(title="Plasmode Simulations with GAN") as f:
    f.append("Generative adversarial neural network (GAN) to generate data")
    f.append(NoEscape(r"\footnote[frame]{Athey S et al. (2021). Using Wasserstein generative adversarial networks for "
                      r"the design of Monte Carlo simulations. \textit{Journal of Econometrics}}"))
    f.append(NoEscape(r"~\\~\\"))
    f.append("Generate new data from existing data")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Avoid arbitrary data generating decisions"))
        itemize.add_item(NoEscape(r"Reflect performance in your particular application"))
        itemize.add_item(NoEscape(r"Share data without re-identification"))
    f.append(NoEscape(r"~\\"))
    f.append("Less than 150 lines")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Compatible with arbitrary input data"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_dgm.png}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_flow.png}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_generated_i1.png}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_generated_i100.png}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_generated_i500.png}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_generated_i2000.png}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Plasmode Simulations with GAN")) as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.8\linewidth]{images/gan_generated_i10000.png}"
                      r"\end{center}"))

# RNN for text generation
with Frame(title="Text Generation with RNN") as f:
    f.append("Recurrent neural network (RNN) to generate abstracts")
    f.append(NoEscape(r"\footnote[frame]{Code available at https://github.com/pzivich/RNN-Abstract-Generator}"))
    f.append(NoEscape(r"\textsuperscript{,}"))
    f.append(NoEscape(r"\footnote[frame]{Sutskever I, Martens J, \& Hinton GE (2011). Generating text with recurrent "
                      r"neural networks. In \textit{ICML}}"))
    f.append(NoEscape(r"~\\~\\"))
    f.append("Generate abstracts focusing on causal inference")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Whole process in \faPython"))
        itemize.add_item(NoEscape(r"Train using published abstracts"))
        with doc.create(Itemize()) as itemize_inner:
            itemize_inner.add_item(NoEscape(r"Query PubMed"))
    f.append(NoEscape(r"~\\"))
    f.append("Less than 300 lines")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Written generally, so could be re-trained for other topics"))

with Frame(title=r"RNN overview") as f:
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.95\linewidth]{images/rnn_flow.png}"
                      r"\end{center}"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Input:} Randomized control trials have been criticized as \\~\\"))
    f.append(NoEscape(r"\textcolor{white}{\texttt{Output:} Randomized control trials have been criticized as a "
                      r"mediation "
                      r"analysis of social schools and other time series (asthma multiple compositions, "
                      r"and confounding. maps that each of the hazard ratio of their assumptions that may examine "
                      r"the effects of causal inference network (rct) and the causal relationship between disease "
                      r"construction and intervention (i.e., all motivating population status are presented to assess "
                      r"robustness framework for interested in the method to infer the substance use and investigating "
                      r"the associations between structured patients}"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Input:} Randomized control trials have been criticized as \\~\\"))
    f.append(NoEscape(r"\texttt{Output:} Randomized control trials have been criticized as a mediation "
                      r"analysis of social schools and other time series (asthma multiple compositions, "
                      r"and confounding. maps that each of the hazard ratio of their assumptions that may examine "
                      r"the effects of causal inference network (rct) and the causal relationship between disease "
                      r"construction and intervention (i.e., all motivating population status are presented to assess "
                      r"robustness framework for interested in the method to infer the substance use and investigating "
                      r"the associations between structured patients"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Input:} Inverse probability of treatment weights were \\~\\"))
    f.append(NoEscape(r"\textcolor{white}{\texttt{Output:} Inverse probability of treatment weights were associated "
                      r"with all "
                      r"interventions or as a causal association between long-term sources and covariate and "
                      r"the consumption of the behavioral research results. in this article, we describe the method "
                      r"of the results of responses and work in a variety of high-current for genetic variants are "
                      r"problematic in statisticians}"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Input:} Inverse probability of treatment weights were \\~\\"))
    f.append(NoEscape(r"\texttt{Output:} Inverse probability of treatment weights were associated with all "
                      r"interventions or as a causal association between long-term sources and covariate and "
                      r"the consumption of the behavioral research results. in this article, we describe the method "
                      r"of the results of responses and work in a variety of high-current for genetic variants are "
                      r"problematic in statisticians"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Input:} results were statistically significant (p=0. \\~\\"))
    f.append(NoEscape(r"\textcolor{white}{\texttt{Output:} Results were statistically significant (p=0.011)}"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Input:} results were statistically significant (p=0. \\~\\"))
    f.append(NoEscape(r"\texttt{Output:} Results were statistically significant (p=0.011)"))

with Frame(title=r"") as f:
    f.append(NoEscape(r"\texttt{Output:} a causal inference approach to interpret, and the a propensity score "
                      r"(ps)"))
    f.append(NoEscape(r"\\~\\~\\"))
    f.append(NoEscape(r"\texttt{Output:} controlling for pregnancy manifererisequally associated \\~\\~\\"))
    f.append(NoEscape(r"\texttt{Output:} Was protective but not statistically significant (p=0.02)\\"))
    f.append(NoEscape(r"\texttt{Output:} We found no evidence of a causal effect (p=0.012)"))

################################
# Installation & IDEs

with Frame(title=NoEscape(r"Getting Started with \faPython")) as f:
    f.append(r"https://www.python.org/downloads/")
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=0.95\linewidth]{images/python_org.PNG}"
                      r"\end{center}"))
    f.append(NoEscape(r"But often will want multiple versions of \faPython\; available"))

with Frame(title=NoEscape(r"A Better Way...")) as f:
    f.append(NoEscape(r"Use \texttt{pyenv} \footnote[frame]{A good introduction is available at "
                      r"https://realpython.com/intro-to-pyenv/} ~\\~\\"))
    f.append(NoEscape(r"\begin{center}"
                      r"\includegraphics[width=1.0\linewidth]{images/pyenv_versions.PNG}"
                      r"\end{center}"))

with Frame(title=NoEscape(r"Getting Started with \faPython")) as f:
    f.append(NoEscape(r"Integrated Development Environment (IDE)"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item("PyCharm")
        itemize.add_item("Jupyter Notebook")
        itemize.add_item("Atom")
        itemize.add_item("RStudio")

with Frame(title=NoEscape(r"Essential Packages")) as f:
    f.append("Basics")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("NumPy, SciPy, pandas")
    f.append("Statistics")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("statsmodels, lifelines")
    f.append("Visualization")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("matplotlib, seaborn")
    f.append("Machine learning")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("sci-kit learn, torch")

with Frame(title=NoEscape(r"Learning \faPython")) as f:
    f.append(NoEscape(r"A number of online resources \\~\\"))
    f.append("Some I've made:")
    with doc.create(Itemize()) as itemize:
        itemize.add_item("https://github.com/pzivich/Python-for-Epidemiologists")
        itemize.add_item("https://github.com/pzivich/publications-code")
        itemize.add_item(NoEscape(r"Smith MJ et al. (2022). Introduction to computational causal inference using "
                                  r"reproducible Stata, R, and Python code: A tutorial. "
                                  r"\textit{Statistics in Medicine}, 41(2), 407-432."))
    f.append(NoEscape(r"~\\"))
    f.append("What worked for me:")
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r"Replicate a completed project in \faPython"))
        itemize.add_item(NoEscape(r"Then start a project in \faPython"))

################################
# Conclusion
with Frame(title=r"Conclusions") as f:
    f.append(NoEscape(r"Be familiar with more than one software"))
    f.append(NoEscape(r"\\~\\"))
    f.append(NoEscape(r"Strongly consider \faPython \; as the next"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item("Language features")
        itemize.add_item("Cross-software capabilities")
        itemize.add_item("Popularity")
    f.append(NoEscape(r"~\\"))
    f.append(NoEscape(r"Uptake in epidemiology / biostatistics is low"))
    with doc.create(Itemize()) as itemize:
        itemize.add_item("More dominated by comp sci / data science")
        itemize.add_item("Lots of opportunity for contributions")

with Frame(title=r"") as f:
    f.append(NoEscape(r"\huge \centering Questions?"))

################################
# END OF DOCUMENT

doc.generate_pdf(clean_tex=True, clean=True, compiler='pdfLaTeX')
