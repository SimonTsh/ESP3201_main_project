# ESP3201_main_project
Code repo for algorithms used in ESP3201 Machine Learning in Robotics module

## Installation steps
1. 

## Possible image segmentation networks (try to find one with a 'road' label already included)
i. Mask R-CNN
ii. SegNet

## Datasets needed
- kitti semantic segmentation labels:<br />
i. http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015<br />
ii. http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015<br />
iii. http://adas.cvc.uab.es/s2uad/?page_id=11<br />
iv. https://rsu.data61.csiro.au/people/jalvarez/research_bbdd.php<br />
v. http://www.zemris.fer.hr/~ssegvic/multiclod/kitti_semseg_unizg.shtml (more accurate version of Ros)<br />

- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/

- CBCL StreetScenes Challenge: http://cbcl.mit.edu/software-datasets/streetscenes/

## From consultation session

- Possible control-demo environments:
i. 2D (overhead): https://gym.openai.com/envs/CarRacing-v0/
ii. 2D (perspective) (OpenAI Gym): Enduro Atari
ii. 3D (difficult to set up): https://www.duckietown.org/research/ai-driving-olympics

- End-to-end self-driving perception & control example (1989):
i. https://colab.research.google.com/github/stephencwelch/self_driving_cars/blob/master/notebooks/Self-Driving%20Cars%20%5BPart%202%20-%20ALVINN%5D.ipynb#scrollTo=GGqen5tmSDUW


## Additional notes
- What is image segmentation?
https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

- Training segnet:
http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

- Semantic segmentation tutorial:
https://pixellib.readthedocs.io/en/latest/image_ade20k.html

- How to use mask R-CNN:
https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

- How to train mask R-CNN with custom datasets:
https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079
https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/
https://github.com/matterport/Mask_RCNN (links to tutorial on next link)
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
