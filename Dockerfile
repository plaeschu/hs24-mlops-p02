FROM pytorch/pytorch:latest
LABEL Name=hs24mlopsp02 Version=0.0.1
WORKDIR /app
COPY . .
RUN pip install torch transformers lightning datasets wandb evaluate ipywidgets scikit-learn
ENTRYPOINT ["python", "main.py", "--projectname", "hs24-mlops-p02" ]