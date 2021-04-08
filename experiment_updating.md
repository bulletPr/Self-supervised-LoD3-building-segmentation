# Experiment 1 implement plan

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The baseline of experiment 1 is FoldingNet, and the plan of this experiment is:

  * [x] - step1: acquire datasets
  * [x] - step2: read datasets by pytorch.data.datset
  * [x] - step3: split scene into blocks
  * [x] - step4: big dataset sampling (see Semantic3D, S3DIS, KIITI preprocessing methods)
  * [x] - step5: load splited and sampled data into batches and pytorch dataloader
  * [x] - step6: FoldingNet Model
  * [x] - step7: chamful loss
  * [ ] - step8: Training strategies (iteravely improve)
  * [x] - step9: Inference process
  * [x] - classification(svm) implementation
  * [ ] - performance analysis
  * [ ] - improve by branch AE
  * [ ] - performance analysis
  * [ ] - Ablation study
  * [ ] - Visualization

# Objects!

  - Using Labelled ArchDataset as training data
  - Using Unlabelled ArchDataset as training data
  - Compare performance and analysis
  - Involving BRANCH AE as OUR ARCHITECUTE


ref git repositories:
  - [AnTao] [gh1]
  - [XuyanBai] [gh2]
  - [PointCNN] [gh3]

### Tech

Experiment1 uses a number of open source projects to work properly:

* [pyTorch1.3] 
* [PPTK viewer] 
* [Numpy] 
* [matplotlib] 
* [pandas] 

And of course itself is open source with a [public repository][gh4]
 on GitHub.

### Installation

Dillinger requires [Node.js](https://nodejs.org/) v4+ to run.

Install the dependencies and devDependencies and start the server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

For production environments...

```sh
$ npm install --production
$ NODE_ENV=production node app
```

### Datasets

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| Name | Source |
| ------ | ------ |
| ModelNet40 | [Download link][ds1] |
| ShapeNetCore | [Download link][ds2] |
| ArCH dataset | [Download link][ds3] |


### Data-preprocessing

Two spliting method: based on pointnet(1*1m, 4096 per block) or pointcnn(1.5*1.5m, 8192 per block)

If use pointnet spliting method, to generate train .h5 files:
```sh
$ python pre_hdf5/gen_arch_h5_pointnet.py
```
If use pointnet spliting method, to generate test .h5 files:
```sh
$ python pre_hdf5/gen_arch_h5_pointnet.py --stride 1.0 --split test
```

If use pointcnn spliting method, to generate .h5 files::
```sh
$ python pre_hdf5/gen_arch_h5_pointcnn.py
```
Then generate filelists:
```sh
$ python pre_hdf5/prepare_arch_filelist_all.py --folder '../data/'
```
#### Building for source
For production release:
```sh
$ gulp build --prod
```
Generating pre-built zip archives for distribution:
```sh
$ gulp build dist --prod
```
### Docker
Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the Dockerfile if necessary. When ready, simply use the Dockerfile to build the image.

```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version} .
```
This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on your host. In this example, we simply map port 8000 of the host to port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:${package.json.version}
```

Verify the deployment by navigating to your server address in your preferred browser.

```sh
127.0.0.1:8000
```


### Todos

 - Write MORE Tests
 - Add Night Mode

License
----

MIT


   [gh1]: https://github.com/AnTao97/UnsupervisedPointCloudReconstruction
   [gh2]: https://github.com/XuyangBai/FoldingNet
   [gh3]: https://github.com/yangyanli/PointCNN
   [gh4]: https://github.com/bulletPr/Unsupervised-learning-on-LoD3-building-point-cloud
   [ds1]: https://modelnet.cs.princeton.edu/
   [ds2]: https://github.com/AnTao97/PointCloudDatasets
   [ds3]: http://archdataset.polito.it/download/
