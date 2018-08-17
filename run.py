import argparse
import logging
import sys
import time
import io
import pickle

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    #with io.open('SingleFrame.yml', 'w') as outfile:
    #    yaml.dump(humans, outfile, default_flow_style=False)
    with io.open('human_data.pkl', 'wb') as output:
        pickle.dump(humans, output, pickle.HIGHEST_PROTOCOL)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)#show heat map result of (Num0:nose)1~(Num17:LEar)18,the 19 is BACKGROUND!!)
    #!!!!!!!!!!!!!!!!!!!!!Altered!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('!!!(It is preprocessed)e.heatMat!!!')
    print(e.heatMat.shape)
    print(tmp)
    tmp255 = (tmp * 255).astype(int)
    x = tmp255.shape[0]
    y = tmp255.shape[1]
    print (x)
    print (y)
    img255=np.zeros([x,y,3])
    img255[:,:,0]  = tmp255[:,:]
    img255[:,:,1]  = tmp255[:,:]
    img255[:,:,2]  = tmp255[:,:]
    cv2.imshow('Image', img255)
    cv2.imwrite('TestImage.jpg',img255)
    #print('!!!Original heatMat data!!!')
    #print(e.tensor_heatMat.shape)
    #print(e.tensor_heatMat[:,:,:,:1])
    #mmap = e.tensor_heatMat[0]
    #print('!!!mmap!!!')
    #print(mmap)
    #print(mmap[:,:,:1])
    #tmp0 = tf.reshape(tf.cast(layer[0], tf.uint8), [16, 16, 1])
    #tmp0 = np.amax(mmap[:,:,:-1],axis =1)
    #plt.imshow(tmp0, cmap=plt.com.gray,alpha=0.5)
    #plt.colorbar()
    #
    #cv2.imwrite('heatMat.jpg',tmp)
    #!!!!!!!!!!!!!!!!!!!!!Altered!!!!!!!!!!!!!!!!!!!
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
