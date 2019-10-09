Simple ranging is a TF py script to display distance data for a tracked object. 

Contents of the repo:
* ranging-tf.py code, super annotated to make it easy to digest and follow along with.

* labels.txt, contains all of the objects this model can identify, this is basically the master list of what you can show this code to make it start ranging ( if you use the linked model ).

Running the program:
* Make sure the entire repo is in the same folder
* wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
* tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
* $python3 ranging-tf.py

Using the program:
* Simply find an object listed in the labels ( phones work well )
* Measure out 1ft from the camera 
* And then follow the on screen prompts

Troubleshooting:
* If the model is having difficult detecting your choosen object, try:
-removing your body as much as possible from the frame
-changing backgrounds / removing clutter
-changing lighting conditions
