wget -O "prepro/img_feat.dat" "https://www.dropbox.com/s/d4nxr46f8il16mh/img_feat.dat?dl=0" 
wget -O "model_gan/gan_56000.ckpt.data-00000-of-00001" "https://www.dropbox.com/s/m4klh2vqd57hd4o/gan_56000.ckpt.data-00000-of-00001?dl=0"
python3 image_generation.py $1
