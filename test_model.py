#!/usr/bin/env python

from keras.models import model_from_json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    model_dir = './models'
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights.json')

    model = model_from_json(open(model_json_filename, 'r').read())
    try:
        model.load_weights(model_weight_filename)
    except:
        print "image_dim_ordering should be 'th' in your ~/.keras/keras.json"
        raise
    model.compile(loss='mean_squared_error', optimizer='sgd')

    with open('sports1m/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    cap = cv2.VideoCapture('dM06AMFLsrc.mp4')
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)
    plt.imshow(vid[2000]/256)
    plt.show()

    X = vid[2000:2016, 8:120, 30:142, :].transpose((3, 0, 1, 2))
    output = model.predict_on_batch(np.array([X]))
    plt.plot(output[0])
    plt.show()

    print "[Debug] output[0]={}".format(output[0])
    print('Position of maximum probability: {}'.format(output[0].argmax()))
    print('Maximum probability: {:.5f}'.format(max(output[0])))
    print('Corresponding label: {}'.format(labels[output[0].argmax()]))

    # sort top five predictions from softmax output
    top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    print('\nTop 5 probabilities and labels:')
    for i in top_inds:
        print('{1}: {0:.5f}'.format(output[0][i], labels[i]))


if __name__ == '__main__':
    main()
