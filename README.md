# VMVS (Variational Multi-View Stereo)

This project is based on Keriven Faugeras's paper. Fast Marching Method(FMM) and Narrow Band Method were used.

For data discrepancy between multi-images, normalized cross correlation should have been utilized but this module has not been tested since 3d object is requred to be rich in texture. But silhouette information came into use.

There still are some modules partially incompleted (not very robust) and commented out, but they will be finished soon.

Here is the result using silhouette infomation

![image](http://github.com/hyperzy/VMVS/raw/master/doc_images/result.png)