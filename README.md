# Segframe
Data Aware Data Acquisition (DADA) is an Active Learning solution that combines uncertainty and representativeness to select anotation images. It's described in detail in [Effective active learning in digital pathology: A case study in tumor infiltrating lymphocytes](https://www.sciencedirect.com/science/article/pii/S0169260722002103).

The code is a framework that lets users select their desired CNN, uncertainty functions, selection strategy, among many other possibilities.

## Dependencies
Code is compliant with Python3+. Python 2.X is **NOT** supported

1. Keras (2.1.15 or newer - tested with 2.2.4);
2. Tensorflow (>= 1.10.0 & < 2.0.0), since specific tensorflow options are used;
3. Opencv 3.X (and python bindings - cv2);
4. Pandas;
5. tqdm;
6. Pydicom;
7. pympler for memory profiling;
8. Numpy (>= 1.16.3 preferably);
9. Openslide 1.1.1 (other versions may work);

## Execution
Soon...
