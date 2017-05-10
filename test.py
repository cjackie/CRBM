from CRBM import CRBM
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt

plot_enabled = False

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    img = data.train.images[0].reshape(28,28) # a image
    img = img -0.5

    # test it..
    filter_shape = (4, 4)
    visible_shape = (28, 28)
    k = 6
    params_id = 'test1'

    # testing runnability of crbm
    crbm = CRBM(filter_shape, visible_shape, k, params_id)
    hidden = crbm.generate_hidden_units(np.expand_dims(img,0))
    rec = crbm.generate_visible_units(hidden, 1)
    print('CRBM running ok')

    #### test Trainer
    crbm = CRBM(filter_shape, visible_shape, k, params_id)
    trainer = crbm.Trainer(crbm)
    img_fitted = np.expand_dims(img,0)
    for _ in range(40):
        trainer.train(img_fitted,sigma=0.01, lr=0.000001)

    if plot_enabled:
        hidden = crbm.generate_hidden_units(img_fitted)
        rec = crbm.generate_visible_units(hidden, 1)
        plt.figure()
        plt.suptitle('original image')
        plt.imshow(img)
        plt.figure()
        plt.suptitle('reconstructed image')
        plt.imshow(rec[0,:,:])
        plt.show()

    #### test Saver
    Saver = CRBM.Saver
    save_path = 'test/save_variables/params_1/w'
    Saver.save(crbm, save_path, 0)

    crbm = CRBM(filter_shape, visible_shape, k, params_id)
    Saver.restore(crbm, save_path+'-0')
    # resume training with
    for _ in range(40):
        trainer.train(img_fitted,sigma=0.01, lr=0.000001)

    #### test Trainer with summary.
    summary_dir = 'test/summaries/params_1/'
    trainer = crbm.Trainer(crbm, summary_enabled=True, summary_dir=summary_dir)
    for _ in range(10):
        trainer.train(img_fitted,sigma=0.01, lr=0.000001)
        print('see {0} with tensorboard'.format(summary_dir))

    print('Ctrl C to exit')
    while True:
        pass
