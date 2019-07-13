"Predicting emotional expression in music with neuralnetworks based on musical cues" created by Andreas Devald Bisgaard and David Oppenberg 

Along with this file in the zip file should be:
run_script.py - file to be run if results and plots in paper are wished to be produced
library.py - containing classes and functions used to executed the run_script.py file
data/
	-MANIFEST.TXT
	-data_release_notes_of_Eerola_et_al_2013.pdf
	-design_matrix.tab
	-mean_emotion_ratings.tab


Program and code has been developed, tested and executed on OSx 10.13.4, with python 3.6.3 anaconda built. The following packages are used and installed, and must be installed on running environment prior execution:
-pandas
-numpy
-scipy
	.stats
-sklearn
	.preprocessing
	.model_selection
-keras
	.models
	.layers
-seaborn
-matplotlib
	.pyplot
	.patches
	.lines