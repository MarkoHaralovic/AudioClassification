# API/tfserving/Dockerfile
FROM tensorflow/tensorflow:latest

# Install TensorFlow Serving
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
   curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

# Copy the models into the container
COPY API/model /model/

# Expose ports for TensorFlow Serving
EXPOSE 8500
EXPOSE 8501

# Run TensorFlow Serving with the provided models
CMD tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=my_model --model_base_path=/model
