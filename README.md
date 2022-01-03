# Learning-based compression of point cloud geometry and color

This repository includes scripts for training and testing a neural network architecture for compression of geometry and color of point clouds. The compression model is based on a convolutional auto-encoder architecture that operates on block partitions of a point cloud. Since the network is exclusively composed of three dimensional convolutional layers, the resolution of the input blocks does not need to be fixed. However, due to memory constraints, the block size used during the training process is limited. 

The architecture of the model is represented on the figure below. 

![autoencoder_architecture](images/autoencoder_architecture.png)

This architecture corresponds to the unified model's structure, where C is the number of output channels that can be set to 1, 3 or 4. These values correspond to the cases where the compression model is used to encode geometry, color, or geometry and color simultaneously. Adjusting this parameter allows for either the selection of a holistic representation of both dimensions, where both geometry and color are simultaneously fed to the network, or a sequential approach, where one model compresses the geometry and another encodes the color attributes.

For more details, the reader can refer to [1].

## Running the code

### Requirements 

The scripts contained in this repository require a number of python libraries to function. The list of requirements as well as their versions used for testing are listed below:

* Python 3.7
* Tensorflow 1.13.1
* PyntCloud 0.1.3
* Tqdm 4.55.1

### Running steps

The usage of the code should be done following these steps:

#### 1. Download pre-trained models, or a partial version of the *HighResGC* dataset to train models

The material can be downloaded from the following **FTP** by using dedicated FTP clients, such as FileZilla or FireFTP (we recommend to use [FileZilla](https://filezilla-project.org/)):

```
Protocol: FTP
FTP address: tremplin.epfl.ch
Username: datasets@mmspgdata.epfl.ch
Password: ohsh9jah4T
FTP port: 21
```

After you connect, choose the **pcc_geo_color** folder, and download the **pre-trained_models** and/or the **partial_HighResGC** sub-folders from the remote site to retrieve corresponding material. 

- The **pre-trained_models** includes compression models that were trained using the complete *HighResGC* dataset and were used in [1].
- The **partial_HighResGC** denotes a subset of the complete *HighResGC* training and test dataset, consisting of block partitions and entire point clouds whose re-distribution is not restricted by owners' licenses. 

To include additional point clouds for training or testing, please refer to the section **Adding training or test data**. To get information regarding the complete *HighResGC* dataset, please refer to the section **Complete *HighResGC* dataset**. 

#### 2. Train a model, or choose a pre-trained model

The user can train a compression model given a set of parameters, using the following script:

```shell
python train.py 'path/to/training/blocks/*.ply' 'path/to/model/checkpoint' --resolution=32 --task=geometry+color --lmbda_g=2500 --lmbda_c=2500
```

The parameter *resolution* specifies the size of point cloud blocks that are fed to the network. The parameter *task* can be set to 'geometry', 'color', or 'geometry+color' in order to enable compression of the corresponding point cloud attribute(s). The parameters *lmbda_g* and *lmbda_c* indicate tuning terms for the trade-off between quality and bitrate of geometry and color, respectively.

Alternatively, a provided pre-trained model can be selected.

#### 3. Compress and decompress a test point cloud

A trained model can be used to compress and decompress block partitions of a test point cloud, using the following scripts:

```shell
python compress.py 'path/to/test/blocks' '*.ply' 'path/to/compressed/blocks' 'path/to/model/checkpoint' --resolution=128 --task=geometry+color
```

```shell
python decompress.py 'path/to/test/blocks' '*.ply' 'path/to/compressed/blocks' '*.ply.bin' 'path/to/decompressed/blocks' 'path/to/model/checkpoint' --resolution=128 --task=geometry+color
```

#### 4. Merge the decompressed point cloud blocks 

The decompressed point cloud block partitions can be merged back to the entire point cloud, using the following script:

```shell
python merge.py 'path/to/test/pointclouds' 'path/to/decompressed/blocks' 'path/to/merged/pointclouds' --resolution=128 --task=geometry+color
```

### Adding training or test data

To use other datasets for training or testing, the corresponding point clouds must be partitioned into non-overlapping blocks, using the following script:

```shell
python partition.py 'path/to/dataset/pointclouds' 'path/to/dataset/blocks' --block_size=32 --keep_size=500
```

The parameter *keep_size* indicates the minimum amount of points per block accepted by the partition method. This parameter was set to 500 to generate the provided training set and should be set to 0 when partitioning point clouds for testing. 

An additional script is provided to randomly choose a specified number of blocks, which can be optionally used to form a training set:

```shell
python sample_dataset.py 'path/to/dataset/blocks' 'path/to/sampled/dataset/blocks' --set_size=10000 
```

The parameter *set_size* indicates the number of blocks to be selected.

### Complete *HighResGC* dataset

For reproducibility reasons, below we list the point cloud contents that were used for training and testing in [1], forming the so-called *HighResGC* dataset:

- The complete *HighResGC* test dataset is composed of the point clouds given in the **partial_HighResGC** under the **test** sub-folder, with the addition of point clouds *longdress_1300*  [3] (10-bit), *bumbameuboi* [4] (9-bit), and *romanoillamp* [4] (10-bit). 
- The complete *HighResGC* training dataset is composed of the point clouds given in the **partial_HighResGC** under the **train** sub-folder, with the addition of point clouds *loot_1200*  [3] (10-bit), *redandblack_1550* [3] (10-bit), *soldier_0690* [3] (10-bit), *matis_495K_02* [5] (10-bit), *rafa_495K_001* [5] (10-bit), *boxer_vox12* [6] (10-bit), *thaidancer_vox12* [6] (10-bit), *basketballplayer_200* [7] (10-bit), *dancer_001* [7] (10-bit), *exercise_050* [7] (10-bit), *model_064* [7] (10-bit), *egyptianmask* [8] (9-bit), *frog* [8] (10-bit), *head* [8] (10-bit), *statueklimt* [8] (9-bit), *shiva* [8] (10-bit), *the20smaria_0600* [8] (10-bit), *ulliwegner_1400* [8] (10-bit), and *unicorn_2m* [8] (10-bit).

All point clouds should be retrieved from corresponding repositories [3]-[8] and voxelized at the bit depth indicated in parenthesis using the script provided under the folder **utilities**.

## Conditions of use

A large portion of the provided scripts is adapted from the [code released](https://github.com/mauriceqch/pcc_geo_cnn) with [2].

If you wish to use any of the provided scripts in your research, we kindly ask you to cite [1].

## References

[1] Evangelos Alexiou, Kuan Tung, Touradj Ebrahimi, "Towards neural network approaches for point cloud compression," Proc. SPIE 11510, Applications of Digital Image Processing XLIII, 1151008
https://doi.org/10.1117/12.2569115

[2] M. Quach, G. Valenzise and F. Dufaux, "Learning Convolutional Transforms for Lossy Point Cloud Geometry Compression," 2019 IEEE International Conference on Image Processing (ICIP), 2019, pp. 4320-4324 
https://doi.org/10.1109/ICIP.2019.8803413

[3] http://plenodb.jpeg.org/pc/8ilabs

[4] http://uspaulopc.di.ubi.pt/

[5] https://v-sense.scss.tcd.ie/research/6dof/quality-assessment-for-fvv-compression/

[6] https://mpeg-pcc.org/index.php/pcc-content-database/8i-voxelized-surface-light-field-8ivslf-dataset/

[7] https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/

[8] http://mpegfs.int-evry.fr/



In case of questions, please contact the following email address:

davi.nachtigalllazzarotto@epfl.ch