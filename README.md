# image-analysis

This is a short script written to test different image analysis algorithms used in the lab.

### Image cropping

Download the file using:

```
$ git clone https://github.com/microfluidix/image-analysis
```

#### By well

One calls the cropping function `_cropAll(PATH,maskSize,wellSize,aspectRatio)` from the library by calling `import cropper` in Jupyter Notebook.

The parameters are as follows:

 - `PATH`: path to Images
 - `maskSize`: size (um) of the mask
 - `wellSize`: size (um) of the feature
 - `aspectRatio`: um-to-px ratio

#### By spheroid

As above, one calls the cropping function `_cropBySph(PATH,livePosition,wellSize,aspectRatio)` from the library by calling `import cropper` in Jupyter Notebook.

The parameters are as follows:

 - `PATH`: path to Images
 - `livePositon`: channel of feature of interest (i.e. cell nuclei)
 - `wellSize`: size (um) of the feature
 - `aspectRatio`: um-to-px ratio
