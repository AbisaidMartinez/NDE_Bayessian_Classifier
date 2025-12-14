# Bayessian Classifier

This repository contains algorithms corresponding to the implementation
of a Bayesian classifier, whose purpose is to discriminate between 
three classes designated as follows:


class	---	angle (degree)
===========================
1	---	8
2	---	16
3	---	25

where each angle corresponds to the inclination of different drill holes
made on a piece of acrylic, ultrasonic signals were captured randomly, 
using two types of frequencies, 5 and 15 MHz, capturing a total of 
100 signals per sample (50 per transducer).

On STFT_imp Branch, this feature extraction was implemented using STFT with 3 
principal components, to obtain the Bayessian Classifier

