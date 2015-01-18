StructuredForests
=================

### Version 1.1

Updates:
* Use compression to reduce model size.
* Rewrite the histogram function by Cython to accelerate detection.
* Finetune some parameters to slightly improve accuracy.

It seems the libjpeg package installed by Anaconda has some bugs in decoding images. The decoding result is different
from the one outputted by Matlab's imread. Thus if you used Anaconda, you may consider uninstalling it and re-installing
libjpeg by apt-get (for Ubuntu).


### Version 1.0

A Python Implementation for Piotr's ICCV Paper "Structured Forests for Fast Edge Detection". The performance is almost
the same as Piotr's original (Matlab) implementation.

For the original implementation, please check the author's Github Page:
https://github.com/s-gupta/rcnn-depth/tree/master/structured-edges
.


## How to use
* Platform:
  Ubuntu 14.04 + Anaconda is highly recommended.
  Nevertheless, I don't use any platform-dependent API. Hence the codes should work on Windows / Mac OS X as well.
  The only problem is related to Cython: it may not be easy to setup 64-bit Cython correctly on Windows, according to
  my previous experience ╮(╯▽╰)╭.


* Toy Demo:
  A very small dataset "toy" was provided. You can run the code via "python StructureForests.py". This
  command will do the following things:
  * Extract 20,000 features from the training data, and save them in "model/data";
  * Train 8 decision trees on the above features, and save them in "model/trees";
  * Merge trees to build the final model, and save it in "model/forests".
  * Use the trained model to detect edges for the testing data, and save them in "edges".


* Actual Usage:
    * Download the BSDS500 dataset from http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/, 
      and uncompress it. As a result, a directory named "BSR" is obtained, containing BSDS500, bench, and documentation.
    * Modify the bottom two lines in "StructuredForests.py" to: 
      `model.train(bsds500_train("BSR"))` and `bsds500_test("BSR", "edges")`. That is, use the "BSR" dataset
      as input, instead of the "toy" dataset.
    * Also modify the model parameters, i.e., "n_pos" and "n_neg" in "StructuredForests.py" to 500,000.
      That is, use 1,000,000 features in total for training (the same as Piotr's ICCV paper).
    * Train and test the model via "python StructuredForests.py". On my machine, about 12 GB memory is required
      for training.


## What is missing
* Multi-scale detection. However, implementing it should only require several lines of codes.
* Speed. I didn't strive for speed. The current implementation is slower than the Author's Matlab
  version, since only one thread is used, and there is no stochastic optimization like SSE. 
  Nonetheless, the speed is acceptable: for BSDS500, detection requires about 1.0s per testing image;
  training requires about 11 hours.
* Depth images. I never tried the NYU depth dataset.


## License
I took some C++ codes from Piotr's original implementation. Those codes are licensed as the author required (see the
author's Github page).

For other codes, they are licensed under the IDC (I don't care) License. O(∩_∩)O.
