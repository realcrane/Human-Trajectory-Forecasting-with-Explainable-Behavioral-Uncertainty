# Human-Trajectory-Forecasting-with-Explainable-Behavioral-Uncertainty
![](https://github.com/realcrane/Human-Trajectory-Forecasting-with-Explainable-Behavioral-Uncertainty/blob/main/images/overview.png)

Human trajectory prediction is a crucial task in understanding human behaviours. The random nature of human movements, from lower-level steering to high-level affective states, dictates that uncertainty estimation is vital in prediction, which consists of data and model uncertainty. However, existing deep learning methods either only capture the data uncertainty or mix the data and model uncertainties. To model these two uncertainties explicitly and investigate their relative importance in prediction, we propose a novel Bayesian differentiable physics model that decouples them. Our model captures the data uncertainty by a new Bayesian explicit physical model, and the model uncertainty by an implicit model. Given that there is no clear separation between the two uncertainties in the data, we design a proper training scheme to enable both uncertainties to capture the ideal randomness. Finally, we use Bayesian inference to incorporate prior knowledge of human behaviours. Through comprehensive evaluation, our model demonstrates strong explainability with fine-grained uncertainty modelling. In addition, it also outperforms existing methods in prediction accuracy by up to 60.17% on public datasets SDD and ETH/UCY.

## Get Started
### Dependencies
Below is the key environment under which the code was developed, not necessarily the minimal requirements:  
  
 1 Python 3.8.8  
 2 pytorch 1.9.1  
 3 Cuda 11.1  
  
And other libraries such as numpy.  
### Prepare Data  
Raw data: SDD (https://cvgl.stanford.edu/projects/uav_data/) and ETH/UCY (https://data.vision.ee.ethz.ch/cvl/aess/dataset/)  
Algorithms in data/SDD_ini can be used to process raw data into training data and testing data. The training/testing split is same as Y-net.  

### Training  
We employ a progressive training scheme. Run trainFa.py, trainFnFs_FixFa.py and train_CVAE.nsp to train the actuation block, the Neighbour Block with parameters of the scene context and CVAE, respectively. The outputs are saved in saved_models.
For example  
`python trainFa.py                                                                                `  

### Authors  
Jiangbei Yue, Dinesh Manocha and He Wang  
Jiangbei Yue scjy@leeds.ac.uk  
He Wang, he_wang@@ucl.ac.uk, [Personal Site](http://drhewang.com/)   

### Contact  
If you have any questions, please feel free to contact me: Jiangbei Yue (scjy@leeds.ac.uk)  

### Acknowledgement  
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 899739 [CrowdDNA](https://crowddna.eu/). 

### License  
Copyright (c) 2022, The University of Leeds, UK. All rights reserved.  
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:    
 1 distributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.    
 2 distributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
