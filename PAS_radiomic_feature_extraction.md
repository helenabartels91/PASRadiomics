# Workflow for radiomic study for Placenta Accreta Spectrum (PAS) using MR images

This script describes a suggested workflow for extracting radiomic features from MR images of patients with Placenta Accreta Spectrum (PAS). 

The main workflow of this script is as follows:

1. Defintion of PAS cases and controls
2. Feature extraction using PyRadiomics
3. Loading data
4. Feature normalization 
5. Feature selection 
6. Model training
7. Statistical analysis

# 1. Definition of PAS cases and controls

Cases of PAS are included when the following criteria are met:
- antenatal imaging: participants who underwent MRI for suspicion of PAS based on the presence of standardised ultrasound features (1)
- intraoperative findings: at the time of laparotomy, clinical features of PAS as defined by the FIGO classification are present (2) 
- histopathology: examination of hysterectomy specimen or myometrial resection with placental tissue with evidence of adherent or invasive placentation as described (3). 

Controls:
Participants with placenta previa and at least one prior caesarean section, who had no ultrasound or MRI features of PAS, and spontaneous placental separation at the time of laparotomy. Placenta previa was defined as the placenta completely covering the internal os on transvaginal ultrasound beyond 20 weeks’ gestation (4). 

In this example, 41 cases and 6 controls were included. Radiomic feature extraction was performed using Pyradiomics (5), resulting in a total of 1820 radiomic features being extracted from 7 feature families, both from the original image and using convolutional image filters such as Laplacian of Gaussian and wavelets.



References: 
1. Collins SL, Ashcroft A, Braun T, Calda P, Langhoff-Roos J, Morel O, et al. Proposal for standardized ultrasound descriptors of abnormally invasive placenta (AIP). Ultrasound in obstetrics & gynecology : the official journal of the International Society of Ultrasound in Obstetrics and Gynecology. 2016;47(3):271-5.
2. Jauniaux E, Ayres-de-Campos D, Langhoff-Roos J, Fox KA, Collins S. FIGO classification for the clinical diagnosis of placenta accreta spectrum disorders. International journal of gynaecology and obstetrics: the official organ of the International Federation of Gynaecology and Obstetrics. 2019;146(1):20-4.
3. Hecht JL, Baergen R, Ernst LM, Katzman PJ, Jacques SM, Jauniaux E, et al. Classification and reporting guidelines for the pathology diagnosis of placenta accreta spectrum (PAS) disorders: recommendations from an expert panel. Mod Pathol. 2020;33(12):2382-96.
4. Jauniaux E, Alfirevic Z, Bhide AG, Belfort MA, Burton GJ, Collins SL, et al. Placenta Praevia and Placenta Accreta: Diagnosis and Management: Green-top Guideline No. 27a. BJOG : an international journal of obstetrics and gynaecology. 2019;126(1):e1-e48.
5. Van Griethuysen JJM, Fedorov A, Parmar C, Hosny A, Aucoin N, Narayan V, et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Res. 2017;77(21):e104-e7.


```python
# Loading required libraries
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
```

# 2. Feature extraction using PyRadiomics 

Segmentations were created using Osirix. Two placental ROIs were generated, based on proximity to the area of adherence or invasive (inferior placental ROI) and remote from this area (superior placental ROI)

Segmentations were exported from Osirix as JSON files using the "Export ROIs" plugin. 

In 3D slicer, nrrd. files for the MRI image and the segmentations were created. A csv. file with column headings "study ID", "image" and "mask" was created. 

Radiomic features were then extracted using pyradiomics using the following script. 


```python
from __future__ import print_function

import logging
import os

import pandas
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor


def main():
  outPath = r'/path_to_documents'

  inputCSV = os.path.join(outPath, 'pathway_codes.csv')
  outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
  progress_filename = os.path.join(outPath, 'pyrad_log.txt')
  params = os.path.join(outPath, 'PAS.yaml')

  # Configure logging
  rLogger = logging.getLogger('radiomics')

  # Set logging level
  # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

  # Create handler for writing to log file
  handler = logging.FileHandler(filename=progress_filename, mode='w')
  handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
  rLogger.addHandler(handler)

  # Initialize logging for batch log messages
  logger = rLogger.getChild('batch')

  # Set verbosity level for output to stderr (default level = WARNING)
  radiomics.setVerbosity(logging.INFO)

  logger.info('pyradiomics version: %s', radiomics.__version__)
  logger.info('Loading CSV')

  # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

  try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case. This is easier for iteration over
    # the input cases
    flists = pandas.read_csv(inputCSV).T
  except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

  logger.info('Loading Done')
  logger.info('Patients: %d', len(flists.columns))

  #if os.path.isfile(params):
  extractor = featureextractor.RadiomicsFeatureExtractor(params)
  #else:  # Parameter file not found, use hardcoded settings instead
    #settings = {}
    #settings['binWidth'] = 25
    #settings['resampledPixelSpacing'] = None  # [3,3,3]
    #settings['interpolator'] = sitk.sitkBSpline
    #settings['enableCExtensions'] = True

    #extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # extractor.enableInputImages(wavelet= {'level': 2})

  logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
  logger.info('Enabled features: %s', extractor.enabledFeatures)
  logger.info('Current settings: %s', extractor.settings)

  # Instantiate a pandas data frame to hold the results of all patients
  results = pandas.DataFrame()

  for entry in flists:  # Loop over all columns (i.e. the test cases)
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                entry + 1,
                len(flists),
                flists[entry]['Image'],
                flists[entry]['Mask'])

    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)

    if str(label).isdigit():
      label = int(label)
    else:
      label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = flists[entry]  # This is a pandas Series
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
        # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
        # as the values in the rows.
        result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
        featureVector = featureVector.append(result)
      except Exception:
        logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

      # To add the calculated features for this case to our data frame, the series must have a name (which will be the
      # name of the column.
      featureVector.name = entry
      # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
      # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
      # it is 'joined' with the empty data frame.
      results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

  logger.info('Extraction complete, writing CSV')
  # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
  results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
  logger.info('CSV writing complete')


if __name__ == '__main__':
  main()

```

    pyradiomics version: v3.0.1
    Loading CSV
    Loading Done
    Patients: 46
    Loading parameter file /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/Radiomics feature extraction files/PAS.yaml
    Enabled input images types: {'Original': {}, 'LoG': {'sigma': [2.0, 3.0, 4.0, 5.0]}, 'Wavelet': {}, 'Exponential': {}, 'Gradient': {}, 'LBP3D': {'binWidth': 32}, 'Logarithm': {}, 'Square': {}, 'SquareRoot': {}}
    Enabled features: {'shape': None, 'firstorder': None, 'glcm': ['Autocorrelation', 'JointAverage', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage', 'DifferenceEntropy', 'DifferenceVariance', 'JointEnergy', 'JointEntropy', 'Imc1', 'Imc2', 'Idm', 'Idmn', 'Id', 'Idn', 'InverseVariance', 'MaximumProbability', 'SumEntropy', 'SumSquares'], 'glrlm': None, 'glszm': None, 'gldm': None}
    Current settings: {'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': True, 'normalizeScale': 100, 'removeOutliers': None, 'resampledPixelSpacing': [1, 1, 1], 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 32, 'voxelArrayShift': 100}
    (1/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL1_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL1_inferior_placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  35] to spacing [1. 1. 1.] and size [71, 85, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (2/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL3_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL3_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7422     0.7422     6.99998571] and size [512 512  36] to spacing [1. 1. 1.] and size [65, 127, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (3/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL4_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL4_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203     0.8203     6.99999355] and size [512 512  32] to spacing [1. 1. 1.] and size [66, 138, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (4/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL5_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL5_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203    0.8203    7.0000074] and size [512 512  29] to spacing [1. 1. 1.] and size [106, 83, 68]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (5/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL6_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL6_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 6.    ] and size [512 512  36] to spacing [1. 1. 1.] and size [135, 84, 65]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (6/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL7_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL7_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7813     0.7813     6.99999355] and size [512 512  32] to spacing [1. 1. 1.] and size [157, 117, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (7/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL8_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL8_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7813 0.7813 6.    ] and size [512 512  47] to spacing [1. 1. 1.] and size [132, 89, 59]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (8/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL9_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL9_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  35] to spacing [1. 1. 1.] and size [142, 94, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (9/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL10_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL10_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  33] to spacing [1. 1. 1.] and size [83, 103, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (10/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL11_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL11_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203     0.8203     6.99998333] and size [512 512  31] to spacing [1. 1. 1.] and size [134, 86, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (11/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL12_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL12_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7813 0.7813 6.    ] and size [512 512  49] to spacing [1. 1. 1.] and size [70, 91, 59]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (12/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL13_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL13_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8594 0.8594 6.    ] and size [512 512  38] to spacing [1. 1. 1.] and size [127, 81, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (13/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL14_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL14_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8594     0.8594     6.00000769] and size [512 512  40] to spacing [1. 1. 1.] and size [122, 71, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (14/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL15_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL15_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  35] to spacing [1. 1. 1.] and size [103, 80, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (15/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL16_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL16_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  37] to spacing [1. 1. 1.] and size [142, 97, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (16/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL17_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL17_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203     0.8203     6.99998148] and size [512 512  28] to spacing [1. 1. 1.] and size [139, 87, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (17/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL18_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL18_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  38] to spacing [1. 1. 1.] and size [140, 98, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (18/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL19B_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL19B_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8594 0.8594 7.    ] and size [512 512  38] to spacing [1. 1. 1.] and size [145, 96, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (19/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL20_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL20_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8594 0.8594 6.    ] and size [512 512  42] to spacing [1. 1. 1.] and size [63, 86, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (20/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL21_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL21_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.742188 0.742188 6.      ] and size [512 512  44] to spacing [1. 1. 1.] and size [143, 80, 65]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (21/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL22_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL22_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7422 0.7422 7.    ] and size [512 512  42] to spacing [1. 1. 1.] and size [102, 102, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (22/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL23_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL23_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203     0.8203     5.99999737] and size [512 512  39] to spacing [1. 1. 1.] and size [105, 90, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (23/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL24_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL24_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.625 0.625 6.   ] and size [512 512  35] to spacing [1. 1. 1.] and size [132, 87, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (24/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL26_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL26_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  34] to spacing [1. 1. 1.] and size [125, 87, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (25/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL27_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL27_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7422 0.7422 7.    ] and size [512 512  40] to spacing [1. 1. 1.] and size [113, 75, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (26/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL28_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL28_inferior-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.4844 1.4844 6.    ] and size [256 256  25] to spacing [1. 1. 1.] and size [131, 84, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (27/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL29_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL29_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625     1.5625     7.00000013] and size [256 256  31] to spacing [1. 1. 1.] and size [137, 80, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (28/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL30_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL30_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.4844     1.4844     7.19999979] and size [256 256  16] to spacing [1. 1. 1.] and size [48, 88, 68]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (29/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL31_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL31_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.1719     1.1719     5.99999948] and size [256 256  14] to spacing [1. 1. 1.] and size [64, 70, 63]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (30/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL32_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL32_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  37] to spacing [1. 1. 1.] and size [147, 84, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (31/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL33_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL33_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  33] to spacing [1. 1. 1.] and size [121, 79, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (32/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL34_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL34_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  37] to spacing [1. 1. 1.] and size [140, 89, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (33/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL36_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL36_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625    1.5625    7.0000002] and size [256 256  39] to spacing [1. 1. 1.] and size [164, 79, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (34/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL37_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL37_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.4063 1.4063 7.    ] and size [256 256  37] to spacing [1. 1. 1.] and size [136, 88, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (35/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL38_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL38_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  32] to spacing [1. 1. 1.] and size [120, 87, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (36/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL39_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL39_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625     1.5625     6.99999928] and size [256 256  36] to spacing [1. 1. 1.] and size [115, 81, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (37/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL40_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL40_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  36] to spacing [1. 1. 1.] and size [127, 76, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (38/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL41_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL41_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  42] to spacing [1. 1. 1.] and size [77, 83, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (39/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL42_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL42_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  36] to spacing [1. 1. 1.] and size [134, 87, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (40/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL43_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL43_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  31] to spacing [1. 1. 1.] and size [118, 112, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (41/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL44_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL44_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.8203 0.8203 7.    ] and size [512 512  45] to spacing [1. 1. 1.] and size [98, 94, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (42/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL45_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL45_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7031 0.7031 7.    ] and size [512 512  35] to spacing [1. 1. 1.] and size [98, 52, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (43/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL46A_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL46A_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  37] to spacing [1. 1. 1.] and size [140, 87, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (44/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL48_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL48_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 6.    ] and size [256 256  35] to spacing [1. 1. 1.] and size [125, 81, 66]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (45/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL49_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL49_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [0.7813 0.7813 6.    ] and size [512 512  45] to spacing [1. 1. 1.] and size [78, 87, 65]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    (46/3) Processing Patient (Image: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL50_MRI.nrrd, Mask: /Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/NRRD files/PASL50_inferior placenta-Inferior Placenta-label.nrrd)
    Calculating features with label: 1
    Loading image and mask
    Applying resampling from spacing [1.5625 1.5625 7.    ] and size [256 256  30] to spacing [1. 1. 1.] and size [96, 86, 75]
    Computing shape
    Adding image type "Original" with custom settings: {}
    Adding image type "LoG" with custom settings: {'sigma': [2.0, 3.0, 4.0, 5.0]}
    Adding image type "Wavelet" with custom settings: {}
    Adding image type "Exponential" with custom settings: {}
    Adding image type "Gradient" with custom settings: {}
    Adding image type "LBP3D" with custom settings: {'binWidth': 32}
    Adding image type "Logarithm" with custom settings: {}
    Adding image type "Square" with custom settings: {}
    Adding image type "SquareRoot" with custom settings: {}
    Calculating features for original image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 2
    Calculating features for log-sigma-2-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 3
    Calculating features for log-sigma-3-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 4
    Calculating features for log-sigma-4-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing LoG with sigma 5
    Calculating features for log-sigma-5-0-mm-3D image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LLH
    Calculating features for wavelet-LLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHL
    Calculating features for wavelet-LHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet LHH
    Calculating features for wavelet-LHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLL
    Calculating features for wavelet-HLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HLH
    Calculating features for wavelet-HLH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHL
    Calculating features for wavelet-HHL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Computing Wavelet HHH
    Calculating features for wavelet-HHH image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for wavelet-LLL image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for exponential image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for gradient image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m1 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-m2 image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for lbp-3D-k image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for logarithm image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for square image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Calculating features for squareroot image
    Computing firstorder
    Computing glcm
    Computing glrlm
    Computing glszm
    Computing gldm
    Extraction complete, writing CSV
    CSV writing complete

