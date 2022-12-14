# PASRadiomics
Workflow for radiomic study for Placenta Accreta Spectrum (PAS) using MR images

This script describes a suggested workflow for extracting radiomic features from MR images of patients with Placenta Accreta Spectrum (PAS).

The main workflow of this script is as follows:

1. Defintion of PAS cases and controls
2. Feature extraction using PyRadiomics 
3. Using RStudio performing feature selection, model training and statistical analysis (R files)

1. Definition of PAS cases and controls

Cases of PAS are included when the following criteria are met:

- antenatal imaging: participants who underwent MRI for suspicion of PAS based on the presence of standardised ultrasound features (1)
- intraoperative findings: at the time of laparotomy, clinical features of PAS as defined by the FIGO classification are present (2)
- histopathology: examination of hysterectomy specimen or myometrial resection with placental tissue with evidence of adherent or invasive placentation as described (3).

Controls: Participants with placenta previa and at least one prior caesarean section, who had no ultrasound or MRI features of PAS, and spontaneous placental separation at the time of laparotomy. 
Placenta previa was defined as the placenta completely covering the internal os on transvaginal ultrasound beyond 20 weeks’ gestation (4).

In this example, 40 cases and 7 controls were included. Radiomic feature extraction was performed using Pyradiomics (5), resulting in a total of 1820 radiomic features being 
extracted from 7 feature families, both from the original image and using convolutional image filters such as Laplacian of Gaussian and wavelets.




References:

1. Collins SL, Ashcroft A, Braun T, Calda P, Langhoff-Roos J, Morel O, et al. Proposal for standardized ultrasound descriptors of abnormally invasive placenta (AIP). Ultrasound in obstetrics & gynecology : the official journal of the International Society of Ultrasound in Obstetrics and Gynecology. 2016;47(3):271-5.
2. Jauniaux E, Ayres-de-Campos D, Langhoff-Roos J, Fox KA, Collins S. FIGO classification for the clinical diagnosis of placenta accreta spectrum disorders. International journal of gynaecology and obstetrics: the official organ of the International Federation of Gynaecology and Obstetrics. 2019;146(1):20-4.
3. Hecht JL, Baergen R, Ernst LM, Katzman PJ, Jacques SM, Jauniaux E, et al. Classification and reporting guidelines for the pathology diagnosis of placenta accreta spectrum (PAS) disorders: recommendations from an expert panel. Mod Pathol. 2020;33(12):2382-96.
4. Jauniaux E, Alfirevic Z, Bhide AG, Belfort MA, Burton GJ, Collins SL, et al. Placenta Praevia and Placenta Accreta: Diagnosis and Management: Green-top Guideline No. 27a. BJOG : an international journal of obstetrics and gynaecology. 2019;126(1):e1-e48.
5. Van Griethuysen JJM, Fedorov A, Parmar C, Hosny A, Aucoin N, Narayan V, et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Res. 2017;77(21):e104-e7.
