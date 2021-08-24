#!/bin/sh

#download dataset
#https://drive.google.com/file/d/1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8/view?usp=sharing
#https://drive.google.com/file/d/1wJ8oux_ZaFO3Ec9YmV1_YwGAwKqlfVAl/view?usp=sharing
#https://drive.google.com/file/d/1jpDODx7eeTbszmrM8TVTclSR_iwIXu-U/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jpDODx7eeTbszmrM8TVTclSR_iwIXu-U' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jpDODx7eeTbszmrM8TVTclSR_iwIXu-U" -O checkpoints.zip && rm -rf /tmp/cookies.txt

unzip checkpoints.zip 
rm -rf checkpoints.zip
