# FIdelity: Efficient Resilience Analysis Framework for Deep Learning Accelerators

This is the open source project for FIdelity: Efficient Resilience Analysis Framework for Deep learning accelerators.

## Fault Injection to Tensorflow
We provide an error injection flow to deep learning models that built on Tensorflow.
Tensorflow 1.12 is required. We suggest using Anaconda to build tensorflow 1.12 environement.

### Fault injection to CNNs
The implementation is under the cnn folder. We provide three example networks: Inception, Resnet50 and Mobilenet with three different data precision: FP16, INT16, and INT8.\
Tensorflow slim is used to define these example networks. Please go to [https://github.com/tensorflow/models/tree/master/research/slim](https://github.com/tensorflow/models/tree/master/research/slim) to download Tensorflow slim.\
To run fault injection,
##### Step 1: 
Replace the files under slim/nets/ with the files under cnn/nets.
##### Step 2: 
Train the networks using Tensorflow slim to obtain the checkpoint.
##### Step 3: 
Prepare the target layer information in a csv file.\
Target layer information includes: (1) the tensor name for the target layer's input, weight and output tensor, and their corresponding dimensions; (2) hyper-parameters such as stride, padding, and (3) the lower bound and upper bound of quantization for this layer's output neurons, if the network uses integer precision. We provide examples of the layer info csv file under the cnn/input folder.
##### Step 4: 
Run one single fault injection with:\
`python inject.py --network xxx --precision xxx --image_id xxx --layer_path xxx --inj_type xxx --ckpt_path xxx`\
Network can be: ic (for Inception), mb (for Mobilenet) and rs (for Resnet).\
Precision can be： FP16, INT16 or INT8.\
Image_id is the inference image ID from Cifar10 dataset.\
Layer_path is the path to the layer information file\
Inj_type is the type of injection based on FIdelity sofware fault model. Currently we support INPUT, WEIGHT, INPUT16, WEIGHT16, RD, and RD_BFLIP. INPUT and WEIGHT stands for injecting single bit-flips to one input or one weight value. INPUT16 and WEIGHT16 inject single bit-flips to one input/weight value, and only affecting 16 output neurons. RD assigns a random value (within the value range of the precision used) to one output neuron. RD_BFLIP inject a random bit flip for one random output neuron.\
Ckpt_path is the checkpoint path for corresponding networks.

// Connect this to paper.

### Fault injection to Transformer network
The transformer network we use here is based on [https://github.com/Kyubyong/transformer.git](https://github.com/Kyubyong/transformer.git). We modified the network components to implement a FP16 version of the network, and then perform fault injection.
##### Step 1: 
Replace the files under the original network with our files under transformer/.
##### Step 2:
Run one single fault injection with:\
`python inject.py --ckpt xxx --precision xxx --inj_type xxx --layer_path xxx --input_path xxx --output_path xxx`\
Ckpt is the checkpoint of this network.\
inj_type is the injection type. We support the same injection types as we have shown for CNNs.\
layer_path is the layer information file. We provide an example under transformer/input folder.\
input_path is the input path (text to be translated) for the network.\
output_path is the golden output path (the golden translation output) for the network.

### Fault injection to Yolo
The yolo network we use here is based on [https://github.com/wizyoung/YOLOv3_TensorFlow.git](https://github.com/wizyoung/YOLOv3_TensorFlow.git). We modified the network components to implement a FP16 version of the network, and then perform fault injection.
##### Step 1: 
Replace the files under the original network with our files under transformer/.
##### Step 2:
Run one single fault injection with:\
`python inject.py --ckpt xxx --precision xxx --inj_type xxx --layer_path xxx --input_path xxx --output_path xxx`
Ckpt is the checkpoint of this network.\
inj_type is the injection type. We support the same injection types as we have shown for CNNs.\
layer_path is the layer information file. We provide an example under transformer/input folder.\
input_path is the input path (text to be translated) for the network.\
output_path is the golden output path (the golden translation output) for the network.

### Avalibility
For all the CNN networks, Transformer network and Yolo network we show, the injection flow is similar. The fault injection files and functions provide a general flow for our users to transport these implementation to any other networks they target.

## FIT rate calculation
After obtaining the error rate for every software fault model, we provide examples of FIT rate calculation under the fit_calc folder.\
Step1: Prepare the following information in a csv file: (1) the number of flip-flops for every software fault model, (2) the average activeness for all fault models, (3) the software masked rate obtained from software fault injection. We provide examples shown in fit_calc/input folder. \
Step2: Run python fit_calc.py --fit_raw --input_file to calculate the overall FIT rate.
Fit_raw is the raw FIT rate of the technology.
Input_file is the file we use to store the input.

## Fault Injection to RTL (to validate FIdelity framework)
We also open sourced the method we use to inject errors to NVDLA's RTL code, which we use to validate our FIdelity framework. (For details, see paper). If users apply FIdelity to their own network/accelerator design, and want to verify the accuracy of the framework, follow the procedure below to obtain RTL fault injection results.

### Fault injection flow

##### Step 1
Download NVDLA hardware repo from [https://github.com/nvdla/hw.git](https://github.com/nvdla/hw.git). Make sure you are in version nvdlav1. Compile NVDLA following the instructions.

##### Step 2 
Copy folder traces/inception_conv_3x3_fp16 under "hw/verif/traces/traceplayer/".

##### Step 3
Go to folder "injection". Modify the "TOOL LOCATIONS" in Makefile_origin to fit your file locations. Especially, the TOPDIR variable should point to your hw/verif folder. Run fault injection with:\
`python inject.py --inj_info xxx --program xxx --hw_path xxx`\
Inj_info is the path to the file that includes inject information, such as the name of the flip-flop, the golden value of the flip-flop, the injection clock cycle and and the bit to flip. We provide an example in the input folder.\
Program is the program we use for injection. Here the example is inception_conv_3x3_fp16.\
Hw_path is the path to NVDLA, which is the path to the /hw folder.\

##### Step 4
Check the injection results with:\
`python collect_result.py --data_path xxx --date xxx --test xxx --golden_result xxx`\
Data_path is the path to the /data folder created when perform injection.\
Date is the date of injection, which is the subfolder of /data.\
Test is the test we run, in our example is inception_conv_3x3_fp16.\
Golden_result is the golden result of the run, we provide an example in the output folder.\

We provide an example output for collect_result.py also in the output folder, which shows the position and value changes of the 16 faulty neurons.

### Configuration and Availability
To demonstrate how the faults are injected, we provide an example of a FP16 convolution layer from Inception network.\
User can configure and run your own layers by applying the following two methods:

#### Method 1
Manually modifying:
(1) The register values in input.txn
(2) The values of input activation in input_feature_map.dat
(3) The values of weights in input_weight.dat
Doing so would require the user understand the functionality of NVDLA registers. We recommend users to read the NVDLA documents [http://nvdla.org/contents.html](http://nvdla.org/contents.html) as well as the source code. As we have succesfully configured many layers with layer output matching Tensorflow exeuction results, we believe this is totally achievable with decend understanding of the architecture. For layer/networks that are currently not supported by the NVDLA compiler, this is the only way to run the layer.

#### Method 2
Users can also make use of the NVDLA compiler if the target layers/networks are supported by the compiler. \
The simplest way we use is inserting "printk" messages to register read/write functions in NVDLA KMD (provided in [https://github.com/nvdla/sw/tree/master/kmd](https://github.com/nvdla/sw/tree/master/kmd)), where CPU talks to NVDLA.\
Then, by compiling and running the target network on NVDLA in virtual environment (provided in [https://github.com/nvdla/vp](https://github.com/nvdla/vp)), the kernel print messages will reveal the register address and value whenever a register in NVDLA communicates to CPU.
One possible position to insert these "printk" messages is in the dla_reg_write and dla_reg_read functions in "kmd/port/linux/nvdla_core_callback.c" file.

### Optimization of injection flow
1. In our example, the target FFs and their golden values are stored in a text file under "injection/input". For FIdelity users, in order to make a correct injection, he/she needs to collect the golden FF values for the target FF before injection. This can be done by simply run the layer (like other exmaples given by NVDLA) without injection. For all the injections on one layer, this only needs to be done once.

2. In our example, the golden network neuron output are stored in a text file under "injection/output". For our FIdelity users, in order to observe the injection outcome, he/she also needs to collect the golden layer output before injection. This can also be done by run the layer without fault injection. For all error injections on one layer, this only needs to be done once.


