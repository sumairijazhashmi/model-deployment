# Model Deployment
I'm trying to learn how to integrate and deploy machine learning models in different systems (websites, mobile apps, embedded systems) and how to run models in the cloud or in the browser. 
I'll be using this repo to track all of my projects I use to learn this.

## Project 1: MNIST Digit Classification 
Tech used: Flask for Backend API, React for Frontend, Amazon EC2 for deployment

Process: Create a pickle file for the model and import the pickle file in the Flask app. Now, you can use the model's prediction (or any other) methods in the app. I also deployed the backend API onto Amazon EC2 instance.


## Project 2: Urdu to Roman Urdu Transliteration
Tech used: Streamlit for model deployment and Tensorflow for model development.

Note: The model is not fully trained due to limited computational resources. The actual fully trained model is available [here](https://github.com/sumairijazhashmi/urdu-roman-transliterator).

Process: Modularize code for model development, save model weights and store in cloud bucket, create a new Streamlit app, import the model as a Python module inside the Streamlit app, and get model's weights from the external resource. Now you can use the model's evaluate function to make predictions. I also deployed the weights to a cloud S3 bucket.


## Project 3: Stack Overflow Security/Privacy-Related Post Classification
Tech used: FastAPI, Docker

Process: Same as above for using the model as a module. Also containerized the app using Docker which can be deployed anywhere. 
