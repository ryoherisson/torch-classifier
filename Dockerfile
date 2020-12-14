FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install nano wget curl

RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

# user setting
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID duser && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser && \
    adduser duser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER duser
WORKDIR /home/duser/

# streamlit
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
EXPOSE 8501