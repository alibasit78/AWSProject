FROM python:3.11

# RUN apt update -y &&  \
#     apt-get update &&  \
#     pip install --upgrade pip
#https://stackoverflow.com/questions/48561981/activate-python-virtualenv-in-dockerfile#:~:text=Build%20a%20venv%20in%20your,you'll%20likely%20find%20success.
WORKDIR /AWSproject_container

COPY ./requirements.txt /AWSproject_container/requirements.txt

RUN pip install -r /AWSproject_container/requirements.txt

COPY . /AWSproject_container/

# EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]