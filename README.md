# ESP3201_main_project
Code repo for algorithms used in ESP3201 Machine Learning in Robotics module

## Installation steps
1. For SegNet, follow instructions from https://github.com/navganti/SegNet to install
2. For Fast-SCNN, follow https://github.com/Tramac/Fast-SCNN-pytorch to setup and test with Cityscape dataset

## Possible image segmentation networks (try to find one with a 'road' label already included)
i. Mask R-CNN <br/>
ii. SegNet<br/>
iii. Fast-SCNN<br/>


## Datasets needed
- Cityscapes dataset: https://www.cityscapes-dataset.com/dataset-overview/ <br/>
This looks the most comprehensive, has road labels, and contains 5k (!!) images (fully labeled) + 20k images (partially labeled) <br/>

- kitti semantic segmentation labels:<br />
i. http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015<br />
ii. http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015<br />
iii. http://adas.cvc.uab.es/s2uad/?page_id=11<br />
iv. https://rsu.data61.csiro.au/people/jalvarez/research_bbdd.php<br />
v. http://www.zemris.fer.hr/~ssegvic/multiclod/kitti_semseg_unizg.shtml (more accurate version of Ros)<br />

- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/

- CBCL StreetScenes Challenge: http://cbcl.mit.edu/software-datasets/streetscenes/

## From consultation session

- Possible control-demo environments:<br/>
i. 2D (overhead): https://gym.openai.com/envs/CarRacing-v0/<br/>
ii. 2D (perspective) (OpenAI Gym): Enduro Atari<br/>
iii. 3D (difficult to set up): https://www.duckietown.org/research/ai-driving-olympics<br/>
iv. 3D: Webots, https://www.cyberbotics.com/ <br/>

- End-to-end self-driving perception & control example (1989):<br/>
i. https://colab.research.google.com/github/stephencwelch/self_driving_cars/blob/master/notebooks/Self-Driving%20Cars%20%5BPart%202%20-%20ALVINN%5D.ipynb#scrollTo=GGqen5tmSDUW<br/>


## Additional notes
- What is image segmentation?<br/>
https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/<br/>

- Training segnet:<br/>
http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html<br/>

- Training R-CNN and Tensorflow:<br/>
https://medium.com/analytics-vidhya/indian-driving-dataset-instance-segmentation-with-mask-r-cnn-and-tensorflow-b03617156d44

- Training Fast-SCNN:<br/>
https://medium.com/deep-learning-journals/fast-scnn-explained-and-implemented-using-tensorflow-2-0-6bd17c17a49e

- Semantic segmentation tutorial:<br/>
https://pixellib.readthedocs.io/en/latest/image_ade20k.html<br/>

- How to use mask R-CNN:<br/>
https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/<br/>

- How to train mask R-CNN with custom datasets:<br/>
https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079<br/>
https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/<br/>
https://github.com/matterport/Mask_RCNN (links to tutorial on next link)<br/>
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46<br/>

- Adapting image segmentation to simulated environment:<br/>
https://junyanz.github.io/CycleGAN/<br/>
