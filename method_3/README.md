# Dishcraft Robotics Deep Learning Coding Challenge

Welcome to the Dishcraft Robotics Deep Learning Coding Challenge!

## Background

Acme Machine Parts, Inc., is a manufacturer of small metal parts. They’ve been trying to develop a computer vision-based system to help with quality inspection of batches of parts coming off their production line. Your task is to apply your machine learning skills to help them solve their problem.

They’ve provided you with a labeled dataset containing images of batches of springs and gears, some of which are damaged. If there is any damaged part in an image, the whole batch is labeled as "damaged" and is marked for further inspection by a factory employee.


## Part I

Using the dataset provided, build and train an ML model that can classify the images of the dataset as either "damaged" or "undamaged" with the highest test accuracy you can achieve.

You should create a Python script with two modes of operation: training and prediction.

1. When executed with the --train flag (e.g. python model.py --train), it should train itself using the dataset provided. (Feel free to organize the data as you see fit.)

2. When executed with the --predict _image.png_ flag (e.g. python model.py --predict _image.png_), it should load trained model weights (e.g. in hdf5 format) and display the highest probability label for the specified (previously unseen, unlabeled) image.


## Part II

Based on the results you got in Part I, write a concise PDF report to the Head of Engineering at Acme Machine Parts that contains:

1. Your reasoning behind how you chose to split the data.

2. Any data pre-processing steps you chose to apply and why you chose them.

3. A description of the experiments you did for model selection and hyperparameter tuning, your reasoning behind the experiments, and the results of the experiments.

4. The generalization performance of your best-performing model.

5. A proposal for how you would further improve generalization performance.

6. (Optional) Suggestions for how to ensure generalization performance doesn't degrade over time.

7. (Optional) A recommendation for how to improve the data collection process. Feel free to go into as much detail about cameras, lighting conditions, hardware, etc. as you would like.

Please send this PDF report to the Dishcraft Robotics recruiting team who has communicated with you regarding the job **no later than 48 hours after beginning this challenge**.

**Please leave your code in the coding-challenge directory in your home directory on the provided EC2 instance, and we'll take a look at it there.**

(Feel free to include any feedback you have on this challenge in your email!)


## Further Details

* The purpose of this challenge is to give you the opportunity to demonstrate your own machine learning skills. Getting help from other people or sharing the data or your code are not allowed.

* If you adapt anyone else's open source code, make a note of this in a code comment, including a link to the code you used.

* Consulting the internet, documentation, and academic papers is encouraged. Citations are also encouraged.

* Using your own or others' pre-trained models is allowed. Please make a note of where you got the pre-trained model in a code comment.

* Feel free to install any libraries you need, using pip or by other means.

* You will lose access to this machine 24 hours after the challenge starts, so make sure you save a copy of any results, figures, code, etc. you will want to include in your report. 

