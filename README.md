**** Settings ****

	Settings are set in main.py.

	num_examples_per_class			Number of examples per class
	normalize_explanations			Normalize explanations or do not
	saveSelectedImages			Save one explanation example image per method and class
	saveAllImages				Save all explanations as image
	resize_Images				Upscale the above images before saving them.
	all_xai_methods				XAI methods to run. Choose from [GradCAM,GradCAMPP,Saliency,DeconvNet, GradientInput, 
								GuidedBackprop, IntegratedGradients, SmoothGrad, SquareGrad,VarGrad,Occlusion,Rise,KernelShap,Lime] 
	all_xai_methods				Names of XAI methods to run. 
	all_xai_metrics				XAI metrics to run from xplique library. Choose from [Deletion,Insertion, MuFidelity] 
	all_xai_metrics_names 			Names of the XAI metrics above.


**** Run ****
	
	--- main program ---
	
	To execute the main program, run
	
	python main.py
	
	The program returns a latex table with average metric scores (table_results_.csv) and a latex table with the average times needed to compute metrics (table_times_.csv). 
	Images are stored in ./paper_images
	
	
	--- Results table aggregation ---
	
	table_results_.csv and table_times_.csv files were obtained by running the main program multiple times. These files are stored in the ./results folder. 
	To aggregate the results into single tables, run
		
	gen_paper_tables.py 
	
	which is stored in the folder ./results



**** Dependencies ****
	
	-- Libraries used in experiments without a gpu available --
	
	Name                      Version  		License

	python                    3.8.1     		https://docs.python.org/3/license.html
	keras                     2.8.0     		https://github.com/keras-team/keras/blob/master/LICENSE
	tensorflow                2.8.0			https://github.com/tensorflow/tensorflow/blob/master/LICENSE
	numpy                     1.22.2		https://github.com/numpy/numpy/blob/main/LICENSE.txt
	pandas                    1.4.1			https://github.com/pandas-dev/pandas/blob/main/LICENSE
	xplique                   0.2.6			https://pythonrepo.com/repo/deel-ai-xplique-python-deep-learning-model-explanation
	scikit-image              0.19.1		https://scikit-image.org/docs/stable/license.html
	scikit-learn              1.0.2			https://scikit-learn.org/stable/
	opencv-python             4.5.5.62		https://opencv.org/license/
	
	
	-- Libraries used in experiments with a gpu available --
	
	Name                      Version  		License
	
	python                    3.8.12   	 	https://docs.python.org/3/license.html
	keras                     2.7.0   	  	https://github.com/keras-team/keras/blob/master/LICENSE
	tensorflow                2.3.0			https://github.com/tensorflow/tensorflow/blob/master/LICENSE
	numpy                     1.18.5		https://github.com/numpy/numpy/blob/main/LICENSE.txt
	pandas                    1.4.1			https://github.com/pandas-dev/pandas/blob/main/LICENSE
	xplique                   0.3.0			https://pythonrepo.com/repo/deel-ai-xplique-python-deep-learning-model-explanation
	scikit-image              0.19.2		https://scikit-image.org/docs/stable/license.html
	scikit-learn              1.0.2			https://scikit-learn.org/stable/
	opencv-python             4.5.5.62		https://opencv.org/license/
	
