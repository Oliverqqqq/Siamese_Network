# Yeast_protein_localisation_sites

# Introduction

This project was done as a part of my coursework for Data Manipulation unit. The propose of this project is using supervised machine learning technique to train a model to predict localisation site for yeast. Also,it requires to visualize the classification and results.

# Data

The file in the Dataset is protein localisations sites for yeast. The data contains 10 columns:
1. Sequence Name: Accession number for the SWISS-PROT database 
2. mcg: McGeoch’s method for signal sequence recognition. 
3. gvh: von Heijne’s method for signal sequence recognition.
4. alm: Score of the ALOM membrane spanning region prediction program.
5. mit: Score of discriminant analysis of the amino acid content of the N-terminal region (20 residues long) of mitochondrial and non-mitochondrial proteins.
6. erl: Presence of “HDEL” substring (thought to act as a signal for retention in the endoplasmic reticulum lumen). Binary attribute.
7. pox: Peroxisomal targeting signal in the C-terminus.
8. vac: Score of discriminant analysis of the amino acid content of vacuolar and extracellular proteins.
9. nuc: Score of discriminant analysis of nuclear localization signals of nuclear and non-nuclear proteins.
10. Localisation Site (see below)

The names and distribution of classes, i.e. localisation sites (column 10), are detailed below:
 * CYT (cytosolic or cytoskeletal) 463 
 * NUC (nuclear) 429 
 * MIT (mitochondrial) 244 
 * ME3 (membrane protein, no N-terminal signal) 163 
 * ME2 (membrane protein, uncleaved signal) 51 
 * ME1 (membrane protein, cleaved signal) 44 
 * EXC (extracellular) 37 
 * VAC (vacuolar) 30 
 * POX (peroxisomal) 20 
 * ERL (endoplasmic reticulum lumen) 5
