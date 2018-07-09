export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
nohup python -u getFaceNorm.py  \
    --models =xx \
    --image-list=./data/cfp_fp.lst \
    --image-dir=./data/cfp_fp \
    --save-dir=./ > nohup.log 2>&1 &
