ThreeDistVis
Visualize a 1D dataset with three normal distributions, classified via Gaussian Mixture Model (GMM), for Power BI, Streamlit, and Tableau. Generates two graphs:

Red, green, blue histograms with known PDFs.
Grey histogram with estimated PDFs.

Setup

Clone the Repository:
git clone https://github.com/sbecker11/ThreeDistVis.git
cd ThreeDistVis


Create Anaconda Environment:
conda env create -f environment.yml
conda activate ThreeDistVis


Install the Package:
pip install -e .


Run the Script:
python -m ThreeDistVis.main



Outputs

distributions_data.csv: Dataset with values, labels, and colors.
images/powerbi.png: Visualization for Power BI.
images/streamlit.png: Visualization for Streamlit.
images/tableau.png: Visualization for Tableau.

Visualizations
Click thumbnails to view full-sized images:



Power BI
Streamlit
Tableau

Project Structure
```
├── distributions_data.csv
├── docs
│   ├── index.html
│   ├── power bi.png
│   ├── streamlit.png
│   └── tableau.png
├── environment.yml
├── images
│   ├── power bi.png
│   ├── streamlit.png
│   └── tableau.png
├── LICENSE
├── pyproject.toml
├── README.md
└── src
    └── ThreeDistVis
        ├── __init__.py
        ├── __pycache__
        └── main.py
```

Deployment
Deployed to GitHub Pages.
Prompt Engineer
The project originated from a discussion to compare data visualization tools—Power BI, Streamlit, and Tableau—and develop a Python-based solution to visualize and classify a synthetic dataset. The requirements included:

Dataset Generation: Create a 1D dataset with three normal distributions, each with distinct means (-0.5 to 0.5), variances, and added uniform noise.
Visualization: Produce two graphs:
A histogram of red, green, and blue samples per bucket, overlaid with smooth probability density functions (PDFs) based on known parameters.
A histogram of all samples in grey buckets, overlaid with estimated PDFs for classified groups in red, green, and blue.


Classification: Implement an algorithm (Gaussian Mixture Model) to classify samples into three distributions.
Tool-Specific Rendering: Generate one page per tool (Power BI, Streamlit, Tableau) as PNGs, with instructions for rendering in each.
Project Structure: Use modern Python tools (pyproject.toml, Anaconda environment), host on a public GitHub repository, and deploy to GitHub Pages under https://sbecker11.github.io/ThreeDistVis/.
Documentation: Include a README.md with thumbnail-sized images linking to full-sized versions.

The discussion evolved to refine the visualization format (PNG outputs), ensure reproducibility, and address potential issues like dependency management and deployment.
License
MIT License
q