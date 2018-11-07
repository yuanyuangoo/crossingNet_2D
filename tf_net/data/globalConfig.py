import sys
import os
sys.path.append('./data/')
# os.path.expanduser('C:/Users/yuany/.theanorc.txt')


# if len(sys.argv) < 2:
#     raise ValueError('should input the dataset')

if len(sys.argv) < 2:
    dataset = 'h36m'
else:
    dataset = sys.argv[1]
dataset = dataset.upper()


# path to the corresponding dataset
msra_base_path = 'database/msra15'
nyu_base_path = 'database/nyu'
icvl_base_path = 'database/icvl/'
lsp_base_path = '/media/a/D/datasets/lspet_dataset'
h36m_base_path = '/media/hsh65/Portable/h36m/'
# h36m_base_path = 'h36m/'

# h36m_base_path = 'D:/datasets/h36m'
cache_base_path = './cache/data/'
skate_base_path = '/media/hsh65/Portable/h36m/'
# path to save the network parameter, intermediate outputs and test result
model_dir = './cache/model/'

# path to the pretrained_model of depthGAN and poseVAE
gan_pretrain_path = os.path.join(
    model_dir, 'depth_gan', '%s_dummy/params/-1' % dataset)
vae_pretrain_path = os.path.join(
    model_dir, 'pose_vae', '%s_dummy/params/-1' % dataset)
print('globalConfig.dataset', dataset)
