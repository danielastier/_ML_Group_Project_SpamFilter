README for HS Machine Learning:

"Classifying Spam vs. Ham: Comparing Text Classification Models Employing Data from Different Approaches (Group Project)"

Lecturer: Cagri Cöltekin
Summer 2016

Deadline: September 15, 2016

------------------------------------------------------------------------------------------
GENERAL INFORMATION
----------------------------------------------

- This folder contains all relevant data concerning the present group project submission, organized in meaningful divisions.

- Since every vector approach has been prepared and evaluated in separation, the following subfolders contain the relevant data belonging to it:
	-> 01_content_vectors: Approach constructing content vectors.
	-> 02_form_vectors: Approach constructing surface form vectors.
	-> 03_raw_data_vectors: Approach constructing index and word-count vectors.
	-> 04_content+form: Approach combining content and surface form vectors.

- The basic data employed within this group project can be found in the following subfolders:
	-> data_CSDMC2010_SPAM (first corpus employed)
	-> data_SpamAssassin (second corpus employed for comparison)


----------------------------------------------
IMPLEMENTATION OVERVIEW
----------------------------------------------

- Data assessment: In order to extract the textual contents of the aforementioned corpora, the already provided Python files should be run (i.e. 'CSC_content_to_CSV.py' in the case of CSDMC2010 Spam corpus and 'SAD_content_to_CSV.py' in the case of SpamAssassin data collection). It might be necessary to change any path settings. In each case, the extracted data will be collected in a CSV file, containing both class labels and raw data ('CSC_mails.csv' for CSDMC2010 and 'SAD_mails.csv' for SpamAssassin).


- 01_content_vectors: This folder contains all Python files relevant to construct content vectors. The building of the CSV files and the running of the Logistic Regression and CNN models are done by running the bash script 'content_runfile' in a terminal. This script should be made executable and ran within this folder. It creates the '*content_wo.csv' files through processing 'CSC.txt' and 'SAD.txt' (created by executing the file 'ML_Preprocessing.py' located within the folder '02_form_vectors', see below). There are two Python files, 'read_data.py' and 'write_csv.py', which do the processing and writing to CSV files. These are executed by the bash script with the relevant arguments: (1) data TXT file, (2) labels CSV file and (3) output CSV file. 


- 02_form_vectors: This folder contains all Python files relevant to construct surface form vectors. Running the code included in 'ML_Preprocessing.py' will transform and preprocess the data provided in 'CSC_mails.csv' or 'SAD_mails.csv' into the required format, including sentence segmentation, tokenization and POS-tagging. Please make sure, the respective paths are set correctly. In each case, the formatted data will be collected in a TXT file, containing all preprocessed data (i.e. 'CSC.txt' for CSDMC2010 and 'SAD.txt' for SpamAssassin). This data is accessed by all remaining Python files which extract certain features and store them as vector representations in individual CSV files (see subfolders '01_CSC_vectors' for CSDMC2010 and '02_SAD_vectors' for SpamAssassin). Please find a detailed description of implemented methods in the attached term paper. Running the file 'ML_Models.py' will employ the constructed feature vectors by concatenating all aforementioned CSVs for a particular corpus into 'CSC_main_vectors.csv' for CSDMC2010 and 'SAD_main_vectors.csv' for SpamAssassin. Furthermore, Logistic Regression and CNN models will get trained on this data. Results will be printed out directly. Please ensure that the respective paths are set correctly.


- 03_raw_data_vectors: All files needed to construct the index vectors are collected in this folder. Before evaluating those vectors through the models in 'model.py', please run 'words_to_integers.py' on both corpora. Make sure the respective corpus is chosen at the beginning of the file. To select the CSDMC2010 Spam corpus, set CSC to True, SAD to False; to choose the Spam Assassin corpus, set SAD to True and CSC to False. Depending on the choice, the 'CSC_mails.csv' or 'SAD_mails.csv' is used to produce either a 'CSC_plain.txt' or a 'SAD_plain.txt' in which the mail content is concatenated to a single string necessary for a word frequency distribution. These files will be accessed to create a 'CSC_to_int.txt' or 'SAD_to_int.csv', respectively. The latter contain the index representations of the mail contents, using the 5,000 most frequent words, and serve as resource for 'models.py'. There, a Logistic Regression classifier is trained on the index vectors and vectors from the CountVectorizer output. The Convolutional Neural Network only receives index vectors as input. 'CSC_mails.csv' and SAD_mails.csv' provide the correct labels for classification. Make sure the input and output paths for the different files are set correctly.


- 04_content+form: This folder contains all files relevant to construct a combination of content and surface form vectors. Having constructed the CSV files concerning content and form is a prerequisite! Running the file 'ML_Models.py' will employ the constructed feature vectors created in the subfolders '01_content_vectors' and '02_form_vectors' by concatenating relevant CSVs for a particular corpus into 'CSC_main_vectors.csv' for CSDMC2010 and 'SAD_main_vectors.csv' for SpamAssassin. Furthermore, Logistic Regression and CNN models will get trained on this data. Results will be printed out directly. Please ensure that the respective paths are set correctly.

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
authors: Patricia Fischer, Daniela Stier, Kimberley Yeo
date: 09/09/2016

