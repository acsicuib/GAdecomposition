This repository includes the source code, the data and the plots for the paper entitled "A Genetic Decomposition Approach for Multi-objective Application Placement in the Compute Continuum", published in the journal "XXXX":

```
Carlos Guerrero, Isaac Lera, A Genetic Decomposition Approach for Multi-objective Application Placement in the Compute Continuum, XXXX, Volume X, 2026, Pages X-X, ISSN XXXX, https://doi.org/10.xxxx/xxxx.
```

If you would consider to cite the paper, you could use this bibtex record:

```bibtex
@article{GUERRERO2026,
title = {A Genetic Decomposition Approach for Multi-objective Application Placement in the Compute Continuum},
journal = {XXXX},
volume = {X},
pages = {X-X},
year = {2026},
issn = {XXXX},
doi = {https://doi.org/10.xxxx/xxxx},
url = {https://www.sciencedirect.com/science/article/pii/XXXX},
author = {Carlos Guerrero and Isaac Lera},
keywords = {Compute continuum, Genetic Algorithm, Problem Decomposition, Resource optimization, Multi-objective optimization, Application placement}
}
```

**Directory Structure**:

- `src/`: Contains all Python scripts for the simulation algorithm, NSGA-II execution, hierarchical scheduler, and plotting generators.
- `config/`: Contains the JSON configuration files representing the three experimental workloads (`simulation_config_A.json`, `simulation_config_B.json`, `simulation_config_C.json`).
- `results/`: Default output directory where each new simulation run generates its respective logs and metrics subdirectories.
- `open_results/`: Stores consolidated metric results (e.g., `open_results/article_data/`) specifically required for regenerating the plots and tables featured in the paper.

**Reproducibility**:

Three different experiments (A, B, and C) are studied in the article.

To reproduce the three experiments, execute the `main.py` script from the `src` directory, passing the respective configuration JSON file as an argument:

```bash
cd src
python main.py ../config/simulation_config_A.json
python main.py ../config/simulation_config_B.json
python main.py ../config/simulation_config_C.json
```

**Generating the Plots**:

To generate the JCR-ready PDFs corresponding to the paper's figures (multi-objective metric boxplots and Pareto front horizontal comparisons), simply execute the plotting scripts from the `src` directory. 

They will automatically read the metrics stored in `../open_results/article_data/` and generate the PDFs in that same folder:

```bash
cd src
python generate_pdf_for_bloxplot.py
python generate_pdf_for_visual_comparison_paretos.py
```
