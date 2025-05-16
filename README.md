# ESCAPE: A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification

<table>
    <tr>
        <td>
            Sebastian Ojeda<sup>1</sup>, Rafael Velasquez<sup>1</sup>, Nicolás Aparicio<sup>1</sup>, Juanita Puentes<sup>1</sup>, Paula Cárdenas<sup>1</sup>, Nicolás Andrade<sup>1</sup>, Gabriel González<sup>1</sup>, Sergio Rincón<sup>1</sup>, Carolina Muñoz-Camargo<sup>1</sup> and, Pablo Arbeláez<sup>1</sup>
        </td>
    </tr>
</table>
<sup>1</sup><em>Universidad de Los Andes, Colombia</em>

___________

**ESCAPE** is a standardized experimental framework for multilabel antimicrobial peptide classification. It combines a large-scale curated dataset, a benchmark for evaluating models, and a transformer-based baseline that leverages both sequence and structural information.
___________
## ESCAPE Database

The **ESCAPE Dataset** integrates over 80,000 peptide sequences from 27 validated public repositories to address critical limitations in existing AMP resources, including data fragmentation, inconsistent annotations, and limited functional coverage. It distinguishes antimicrobial peptides from negative sequences and organizes their functional annotations into a biologically meaningful multilabel hierarchy, covering antibacterial, antifungal, antiviral, and antiparasitic activities.The dataset comprises 21,409 experimentally validated AMPs and 60,950 non-AMPs filtered from unrelated sources.



<p align="center">
<img src="Figures/overview_ESCAPE.png" width="800">
</p>




The ESCAPE Dataset is publicly available for download. You can access the complete ESCAPE Database on [Harvard Dataverse](https://doi.org/10.7910/DVN/C69MCD).

___________

## ESCAPE Benchmark


We evaluate six representative models for antimicrobial peptide classification: **AMPlify, AMP BERT, TransImbAMP, amPEPpy, AMPs Net,** and **PEP Net**, using the multilabel framework defined by ESCAPE. Each model was modified to support multilabel classification and trained with two fold cross validation. We report final performance by averaging predictions from both folds through an ensemble strategy. Evaluation uses two standard metrics for multilabel tasks: F1 score and mean Average Precision, which are suitable for datasets with class imbalance.

The table below summarizes the key methods for antimicrobial peptide classification of the ESCAPE Benchmark, their primary architectures, GitHub repositories, and the F1-score and mean Average Precision (mAP) these methods achieve by evaluating them on the ESCAPE Dataset.

| Method      | Primary Architecture          | GitHub Repository                                             | F1-score (%) | mAP (%) |
|-------------|-------------------------------|---------------------------------------------------------------|--------------|---------|
| Amps-Net    | GCN                           | [GitHub](https://github.com/BCV-Uniandes/AMPs-Net)            | 57.7         | 54.2    |
| TranslmbAMP | Transformer                   | [GitHub](https://github.com/BiOmicsLab/TransImbAMP)           | 61.9         | 64.9    |
| AMP-BERT    | BERT                          | [GitHub](https://github.com/GIST-CSBL/AMP-BERT)               | 66.1         | 66.2    |
| amPEPpy     | Random Forest (RF)            | [GitHub](https://github.com/tlawrence3/amPEPpy)               | 65.0         | 68.0    |
| PEP-Net     | Transformer                   | [GitHub](https://github.com/hjy23/PepNet)                     | 65.2         | 68.2    |
| AMPlify     | Bi-LSTM with attention layers | [GitHub](https://github.com/bcgsc/AMPlify)                    | 68.9         | 71.1    |
| **ESCAPE**  | Dual-branch transformer       | [GitHub](https://github.com/BCV-Uniandes/ESCAPE)              | **69.4**     | **72.7**|

### Getting Started

To reproduce our results, you need to set up the required environment with all dependencies. We provide a Conda environment file to streamline this process. Create the environment by running:

```
conda env create -f ESCAPE.yaml
conda activate ESCAPE
```

### Reproducing ESCAPE Benchmark Results

To reproduce the **ESCAPE Benchmark** results on the ESCAPE Dataset:

**1.**	Update the paths to both model checkpoints in the `ensemble.sh` executable script.

**2.**	Set the model architecture in the `model_ensemble.py` file.

**3.**	Run the following command:

```bash
bash ensemble.sh
```
This script loads both trained models, averages their outputs, and computes the final metrics over the test set.


___________

## ESCAPE Baseline

The **ESCAPE Baseline** is a dual-branch transformer architecture designed to classify antimicrobial peptides (AMPs) using both sequence and structural information. It processes amino acid sequences through a transformer encoder and structural representations through a second branch that encodes peptide distance matrices. These two modalities are fused using a bidirectional cross-attention mechanism, enabling the model to capture both biological context and spatial structure. This approach leads to state-of-the-art performance on the ESCAPE Benchmark across multiple AMP functional classes.

<p align="center">
<img src="Figures/Archutecture_ESCAPE.png" width="800">
</p>

### Structural Inputs

For the structural branch, each peptide is represented as a 224×224 distance matrix, where each element corresponds to the Euclidean distance between C-alpha atoms in the 3D conformation. These matrices are precomputed for all peptides in the test split and stored as **.npy** files.

You can download the distance matrices for the test set from this [link](https://drive.google.com/drive/folders/1e30YX0eztjauwTM5EJ00me-2JS_0iYSF?usp=sharing). Place the .npy files in the appropriate directory as specified in the code before running evaluation.




## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.


