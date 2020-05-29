import numpy as np
from keras import optimizers
import skimage.io as io
from Custom_Model import unet as seg # Importing the Unet model defined earlier

#Training the model with binary cross entropy (or dice loss) for 5 epochs:
adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
seg.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_crossentropy'])
seg.fit(train_x, train_y, epochs=5, batch_size=16, verbose=1, validation_split=0.10)

#Visualizing the model results on selected test images:

img=train_x[50]
io.imshow(img)

test=train_y[50]
test=test.squeeze()
io.imshow(test, cmap="gray")

img = np.expand_dims(img, 0)
pred = seg.predict(img, batch_size=1)
pred=pred.squeeze()
pred=pred*255
io.imshow(pred, cmap="gray")

