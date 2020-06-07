FROM python
COPY . /app
RUN pip install cython
RUN pip install numpy
RUN pip install torch torchvision
RUN pip install Pillow
RUN pip install imageio
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
WORKDIR /app
CMD python trainer.py --num_epochs 1  && python runMe.py && python project_test.py


