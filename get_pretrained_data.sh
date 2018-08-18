#!/bin/bash
output=pretrained
mkdir $output
wget --directory-prefix=$output {https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward_newtf.py,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/caffe_classes.py,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/poodle.png,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/laska.png,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/dog.png,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/dog2.png,https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/quail227.JPEG}
