# BKSVD: Python Implementation

This repository contains a Python implementation of the "Bayesian K-SVD for H and E blind color deconvolution" (BKSVD) algorithm. The original MATLAB code was developed by Fernando Pérez-Bueno et al. and is available [here](https://github.com/vipgugr/BKSVD).

This project provides tools for stain separation and image normalization in histological image analysis, as described in the original publication:


> Pérez-Bueno, F., Serra, J. G., Vega, M., Mateos, J., Molina, R., & Katsaggelos, A. K. (2022). Bayesian K-SVD for H and E blind color deconvolution. Applications to stain normalization, data augmentation and cancer classification. *Computerized Medical Imaging and Graphics*, *95*, 102048. https://doi.org/10.1016/j.compmedimag.2022.102048

## Original Project

The original MATLAB code and data can be found at: [https://github.com/vipgugr/BKSVD](https://github.com/vipgugr/BKSVD)

## Requirements

The required Python packages are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

This repository includes a jupyter notebook `Example.ipynb` that shows the basic usage for this code. 

It is also possible to run the example by executing the `main.py` script:

```bash
python main.py
```

This will process the sample images in the `data` directory and display the original and normalized images using `matplotlib`.

## Data
The data included in this repo is a small sample to provide an example. Please notice that the patches have a very low quality and present jpg artifacts. We recommend to explore the algorithm using your own data.

The patched image is extracted from the SCAN algorithm dataset
Salvi, Massimo; Michielli, Nicola; Molinari, Filippo (2020), “SCAN algorithm dataset”, Mendeley Data, V1, doi: 10.17632/sc878z8pm3.1

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{PEREZBUENO2022102048,
title = {Bayesian K-SVD for H and E blind color deconvolution. Applications to stain normalization, data augmentation and cancer classification},
journal = {Computerized Medical Imaging and Graphics},
pages = {102048},
year = {2022},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2022.102048},
url = {https://www.sciencedirect.com/science/article/pii/S0895611122000210},
author = {Fernando Pérez-Bueno and Juan G. Serra and Miguel Vega and Javier Mateos and Rafael Molina and Aggelos K. Katsaggelos},
keywords = {Bayesian modelling, Histological images, Blind Color Deconvolution, Stain Normalization}
}
```

## IA disclaimer

IA has been used to generate this code from the original MATLAB implementation. Notice that this is still under review and might contain errors or not reproduce exactly the same implementation as explained in the paper. Use it with caution.