# emotion and action recognition
**update from old[repo]https://github.com/harvitronix/five-video-classification-methods,https://github.com/oarriaga/face_classification**
*data are selected from kinetics datasets(including action information and facial expression information)
*standard training/testing data are prepared in fold hy_bimodal_sentimentasl_analysis/data/,including
   (1) how to extract sequeence frame from video
   (2) the ground truth for training set and testing set
* data include eight classes, the detail of datasets can turn to baiducloud **https://pan.baidu.com/s/1bYkPp2qozSQA3qJD-t9n4w**



##instructions
###to train models for emotion feature extraction
* download fer2013 datasets with seven emotion classes
* cd multimodal_sentiment_recognition/emotion_recognition/src/utils, modified the path of emotion datasets in datasets.py
* cd multimodal_sentiment_recognition/emotion_recognition/src/, run train_emotion_classifier.py and the trained models are stored into specified path
* cd multimodal_sentiment_recognition/hy_bimodel_sentimental_analysis, runing extract_emotion_features.py to extract facial expression features


###to train models for action feature extraction
* download our collected datasets in baiducloud **https://pan.baidu.com/s/1bYkPp2qozSQA3qJD-t9n4w**
* cd multimodal_sentiment_recognition/hy_bimodel_sentimental_analysis, running extract_action.py to extract action feature


###to extract synchronous feature from traiend emotion model and traiend action model
* cd multimodal_sentiment_recognition/hy_bimodel_sentimental_analysis, running extract_synchronous_feature.py and storing the features into specified path for traing RNN network


####to train multimodel_model for sentiment analysis 
* cd multimodal_sentiment_recognition/hy_bimodel_sentimental_analysis, running training.py


