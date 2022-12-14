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
