import build
import os
import jinja2
import time

template_file = "report_template.tex"
template_file_fr = "report_template_fr.tex"

while True:
    x = input("On which year should the newsletterfocus ? ")
    try:
        build.set_year(int(x))
        break
    except ValueError:
        print("choose another year please ")

while True:
    y = input("In which language? (en,fr)")
    try:
        build.set_lang(y)
        break
    except ValueError:
        print("")

latex_outfile = "report_" + str(x) + "_"+ str(y) + ".tex"

build.compute_stat()
if y=='fr':
    build.create_latex(template_file_fr, latex_outfile, build.VARS)
if y=='en':
    build.create_latex(template_file, latex_outfile, build.VARS)
build.create_pdf(latex_outfile)
