# Option Pricing Models

## Introduction  
This repository represents simple web app for calculating option prices (European Options). It uses three different methods for option pricing:  
1. Black-Scholes model    
   A mathematical model used to calculate the theoretical price of European-style options, based on factors like current stock price, strike price, time to expiration, risk-free rate, and volatility.

2. Monte Carlo simulation    
   A probabilistic method that uses random sampling to estimate option prices by simulating multiple possible price paths of the underlying asset.

3. Binomial model    
   A discrete-time model that represents the evolution of the underlying asset's price as a binomial tree, allowing for the calculation of option prices at different time steps.

Each model has various parameters that can be input to calculate the Options price for a given ticker:  

- Strike price
- Risk-free rate (%)
- Sigma (Volatility) (%)
- Exercise date

Option pricing models are implemented in [Python 3.9](https://www.python.org/downloads/release/python-390/). Latest spot price, for specified ticker, is fetched from Yahoo Finance API using [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/). Visualization of the models through simple web app is implemented using [streamlit](https://www.streamlit.io/) library.  

When a ticker is specified the user needs to press enter which sends a request to Yahoo Finance API to get the latest ticker data which is loaded to
1. Calculate the range of strike price
2. Calculate the options price after parameters are selected

When data is fetched from Yahoo Finance API using pandas-datareader, it's cached with [request-cache](https://github.com/reclosedev/requests-cache) library is sqlite db, so any subsequent testing and changes in model parameters with same underlying instrument won't result in duplicated request for fethcing already fetched data.

## Streamlit web app  

1. Black-Scholes model    
![black-scholes-demo](./demo/streamlit-webapp-BS.gif)

2. Monte Carlo Option Pricing  
![monte-carlo-demo](./demo/streamlit-webapp-MC.gif)

3. Binomial model    
![binomial-tree-demo](./demo/streamlit-webapp-BC.gif)


## Project structure  
In this repository you will find:  

- demo directory - contains .gif files as example of streamlit app.
- option_pricing package - python package where models are implemented.
- streamlit_app.py script - web app for testing models using streamlit library.
- Requirements.txt file - python pip package requirements.
- Dockerfile file - for running containerized streamlit web app.


## How to run code?
You can use simple streamlit web app to test option pricing models by running a docker container. Don't worry you do not need deep docker knowledge to run it.


### **Running docker container locally**
Dockerfile has exposed 8080 (default web browser port), so when you deploy it to some cloud provider, it would be automatically possible to access your recently deployed webb app through browser.

> **Note:** Make sure that you have Docker installed on your machine before proceeding.

***1. Navigate to the repo directory locally***
Navigate to the directory where this repository is located on your local machine. For instance, open your terminal and navigate to the directory where this repository resides on your machine. Now, type ‘cd’ followed by the name of the directory where this repository is located.

***2. Build the docker image***
First you will need to build the docker image (this may take a while, because it's downloading all the python libraries from Requirements.txt file) and specify tag e.g. option-pricing:latest:  
`docker build -t options-pricing:latest .`  

***3. Build the docker image***
When image is built, you can execute following command, that lists all docker images, to check if image was successfully build:  
`docker image ls`

***4. Build the docker image***
Now, you can run docker container with following command:  
`docker run -p 8080:8080 options-pricing:latest`  

When you see output in command line that streamlit app is running on port 8080, you can access it with browser:  
`http://0.0.0.0:8080/`  