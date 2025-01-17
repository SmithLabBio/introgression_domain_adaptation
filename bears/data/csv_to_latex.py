import pandas as pd
import numpy as np
from jinja2 import Template

template = Template(r"""
\begin{longtable}{ {{ "l" * df.shape[1] }} }

% Header for first page
\caption{ {{ caption }}} \\
\hline

% Column names for first page
{% for col in df.columns -%}
    \multicolumn{1}{c}{ {{col}} }
    {%- if not loop.last %} & {% else %} \\ {% endif %} 
{% endfor -%}

\hline 
\endfirsthead

% Header for all but first page
\multicolumn{4}{c}%
{{ '{{' }} \tablename\ \thetable{} -- continued from previous page {{ '}}' }} \\
\hline

% Column names for all but first page
{% for col in df.columns -%}
    \multicolumn{1}{c}{ {{col}} }
    {%- if not loop.last %} & {% else %} \\ {% endif %} 
{% endfor -%}

\hline 
\endhead

% Footer for all but last page 
\hline 
\multicolumn{4}{r}{{ '{{' }}Continued on next page{{ '}}' }} \\
\endfoot
\hline 
\endlastfoot

% Table data
{% for index, row in df.iterrows() %}
    {% for value in row %}
        {{- value -}}
        {% if not loop.last %} & {% endif %}
    {% endfor %}
    {% if not loop.last %} \\ {% endif +%}
{% endfor %}

\label{ {{ label }} }
\end{longtable}
""",
trim_blocks=True,
lstrip_blocks=True)


def makeTable(df, path, caption, label):
    out = template.render(df=df, caption=caption, label=label)
    with open(path, "w") as fh:
        fh.write(out)

included_df = pd.read_csv("/mnt/home/kc2824/bears/filtering/brown-included-in-final-analyses.txt", header=None)
df = pd.read_csv("selected-for-analysis.csv", na_filter=False)
df.drop("Publication Sample ID", axis=1, inplace=True)
df = df[df["BioSample ID"].isin(included_df.iloc[:,0])]
df.sort_values(by=["Publication", "Population", "BioSample ID", "SRA Run ID"], inplace=True)
makeTable(
    df=df,
    path="/mnt/home/kc2824/manuscript/sections/bear-samples.tex",
    caption="Bear samples used in our empirical tests for introgression.",
    label="si:samples")

df.to_csv("/mnt/home/kc2824/manuscript/supmat/bear-sample-accession-ids.csv", index=False)
# Make sure sample number matches expected
df.drop_duplicates(subset=["BioSample ID"], inplace=True)
print(len(df))