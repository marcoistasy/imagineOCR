# **ImagineOCR**
<p align="center">
    <img src="Resources/Banner.png" width="890" alt="Logo"/>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ImagineOCR represents a fundamental restructuring of image-to-text software. Abandoning previous principles emphasising static models of orthography, imagine OCR approaches OCR as the province of object detection. Accordingly, it views each character on a page as a discrete object and allows training of a custom faster-rcnn implementation given as little as a single instance of the objects (read: characters) to be detected. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

#### Virtual Environment Set Up
We recommend setting up ImagineOCR in a new python virtual environment to avoid conflicting dependencies. To facilitate this, we've included a copy of our `environment.yml`. Simply enter the following command to create a virtual environment.
```
conda env create -f environment.yml
```
**Note that while we use `conda`, any virtual environment solution will work.**

### Installation

#### Initialising Submodules

ImagineOCR ships with [LabelImg](https://github.com/tzutalin/labelImg) and [Tensorflow Object Detection API](https://github.com/tensorflow/models) as `submodules` and requires their installation before proceeding. Please see the [instructions for installing LabelImg](https://github.com/tzutalin/labelImg#installation) and the [instructions for installing Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

**Note that the virtual environment created above already includes the dependencies required by these project. There is no need to run any `conda install` or `pip install` commands.**

## Utilisation
A step by step series of examples that tell you how to get a development env running

## Built With

* [Tensorflow](https://www.tensorflow.org)
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [LabelImg](https://github.com/tzutalin/labelImg)

## Contributing

Please read [CONTRIBUTING.md](https://github.com/marcoistasy/imagine-ocr/blob/master/CONTRIBUTING.md) for the process for submitting pull requests to us. Also make sure to read our [CODE_OF_CONDUCT.md](https://github.com/marcoistasy/imagine-ocr/blob/master/CODE_OF_CONDUCT.md).

## Versioning

We use [SemVer](http://semver.org/) for versioning.

## Authors

* **[Marco Istasy](https://github.com/marcoistasy)**

See also the list of [contributors](https://github.com/marcoistasy/imagine-ocr/graphs/contributors) who participated in this project.

## License

Released under the [GLP-3.0](LICENSE.md) license.

## Acknowledgments

* Sincerest thanks to Kieren Nicôlas and Stephen Lovell for their continued support and unwavering belief in the project.