# Project Plan Intro to DS DATA 201!
## Overview 
The goal of this project is to train different machine learning algorithms to interpret of Infrared Absorbance Spectra of different chemical systems and assess the accuracy of different models. IR Absorbance spectra have been used in chemistry for many decades as a means of identifying specific functional groups in molecules. In this approach chemical structures can be thought of as Lego sets and the IR spectra reveals useful information about which pieces are used. Because each specific "piece" (for instance a C double-bonded to an O) has a different infrared absorbance (corresponding to bond vibration energies) the infrared spectra can be used to glean valueable information about key features present in a molecular system. 

For Example the IR absorbance spectra for n-butanol shows a number of useful features corresponding to the different functional groups in Butanol (CH3-CH2-CH2-CH2-OH): 
![Butanol](images/butanol.png)

Specifically, the sharp peak at ~3500 cm^-1 corresponds to the O-H stretch of the alcohol and the broad peaks at ~3000 cm^-1 correspond to the CH stretches. Note that a functional group is a multi-label classifier (ie. one compounds IR spectra often can indicate the presence of multiple functional groups). 

## 1. Data Collection 

In order to train a ML model to recognize these IR features and map them to functional groups we need a basis set of chemical systems and functional groups of interest. To limit the scope of and computational cost of this project we will focus on small organic molecules (less than 25 atoms of just C,N,H,O,Cl, Br). We will also focus on the following functional groups (C-H, C=C, C≡C, C≡N, C-OH, C=O, C=OOH, N-H). 

Another issue is the lack of mass-downloadable data for IR Spectra, meaning that digitized spectra will have to be downloaded manually from the NIST webbook. As such, 50 Spectra were specifically chosen to form the training set (and accurately capture the functional gropups of interest). These spectra include 40 Systems with only one main functional group (and C-H) and 10 that mix different functional groups together. An additional 10 spectra will be used as a test set to assess model accuracy. 

### Individual Functional Groups (40):
| C-H | -OH | COOH | H-C=O/C-C=O | C=C | C≡C | C≡N | N-H |
| ------- |  ------- | ---------- | ----------------- | ------- | ------- |------ | ------ |
| Methane | Methanol | Formic Acid | AcetylAcetone | Ethene | Prop-1-yne | Acetonitrile | Propylamine |
| Ethane | Ethanol | Ethanoic Acid | Acetaldehyde | But-1-ene | 2-Butyne | 2-methylpropanenitrile |  | cyclohexylamine|
| Butane | Butanol | Butanoic Acid | Acetone | Hex-3-ene | Pent-1-yne | pentanenitrile | Methylamine |
| Hexane | Hexanol | Hexanoic Acid | Butan-2-one | Hexene | Hex-2-yne | cyclopentylcyanide | Ethylamine |
| Heptane | Octanol | Octanoic Acid | 3-Pentanone | Oct-2-ene |  3-methylbut-1-yne | propanenitrile | Hexylamine |

### Mixed Functional Groups (10):
| Compound Name | Functional Groups |
| --------------| ------------------|
| Glycine | N-H, C=OOH |
| 2-cyclohexen-1-ol | C=C, C-OH |
| 2-(ethylamino)ethan-1-ol | N-H, C-OH |
| methallyl alcohol | C=C, C-OH |
| Acrylonitrile | C=C, C≡N |
| but-3-en-2-one | C=C, C=O |
| but-3-yn-2-one | C≡C, C=O |
| 2-butynoic acid | C≡C, C=OOH |
| formamide | N-H, C=O |

All data are sourced from NIST Webbook: https://webbook.nist.gov/chemistry/

