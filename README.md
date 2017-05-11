# Climate - Vegetation Dynamics: a Granger causality framework
This repository contains the source codes used in the following publications: 
* Papagiannopoulou et al., 2017. "A non-linear Granger-causality framework to investigate climate–vegetation dynamics". Geoscientific Model Development. https://doi.org/10.5194/gmd-2016-266.
* Papagiannopoulou et al., 2017. "Vegetation anomalies caused by antecedent precipitation in most of the world". Environmental Research Letters. https://doi.org/10.1088/1748-9326/aa7145.

### Prerequisites

The script is mainly built using the following packages:

```
python 2.7.12 | Anaconda 2.3.0 (64-bit) (or higher)
scikit-learn
```

## Running the tests

In order to test the framework, a test file is provided ([test.csv](test.csv)). This file contains all features used in the publication for one pixel on earth.

The script allows you to test both the "linear" and "non-linear" framework. In order to execute these, use the following commands:

```
python GC_script.py test.csv linear
python GC_script.py test.csv non-linear
```

## Output

The outcome of both tests provide you information on the explained variance of the baseline and full model as well as a quantification of the Granger causality. For more information, see publication.

The output should look as follows:

### 1. Linear case

```
Explained variance of baseline model: 0.076398
Explained variance of full model: 0.199819
Quantification of Granger causality: 0.123421
Total time: 58 seconds
```

### 2. Non-Linear case
```
Explained variance of baseline model: 0.055405
Explained variance of full model: 0.306413
Quantification of Granger causality: 0.251008
Total time: 6 seconds
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details


