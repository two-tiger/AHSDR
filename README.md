## 📋Dependence
torch==1.3.1 

torchvision==0.4.2 

numpy==1.16.4 

absl-py==0.9.0 

cachetools==4.0.0 

certifi==2019.11.28

chardet==3.0.4 

Cython==0.29.15

google-auth==1.11.2 

google-auth-oauthlib==0.4.1 

googledrivedownloader==0.4 

grpcio==1.27.2 

idna==2.8 

Markdown==3.2.1 

oauthlib==3.1.0 

Pillow==6.1.0 

protobuf==3.11.3 

pyasn1==0.4.8 

pyasn1-modules==0.2.8 

quadprog==0.1.7 

requests==2.22.0 

requests-oauthlib==1.3.0 

rsa==4.0 

six==1.14.0 

tensorboard==2.0.1 

urllib3==1.25.8 

Werkzeug==1.0.0 

## 📋Running

- Use ./utils/main.py to run experiments. 

- New models can be added to the models/ folder.

- New datasets can be added to the mydatasets/ folder.

## 📋Results

All the results can be achieved after running ablation.sh



## 📋Conclusion

This paper revisited the memory buffer selection problem in replay-based continual learning and proposed the innovative AHDSR method. We addressed the challenge of effective and efficient buffer selection by formulating a min-max local search problem to efficiently select representative samples from HSDR within each class for memory buffer inclusion. Specifically, we recasted the localization of RSLER as the min-max optimization problem and explored it from an adversarial attack perspective. This strategy introduces OTM information during the evaluation of individual samples, ensuring optimal performance while avoiding expensive computations on multiple samples. Extensive experimental evaluations on various benchmarks demonstrated AHDSR's superiority over current state-of-the-art methods, highlighting notable advantages in both performance and computational efficiency.