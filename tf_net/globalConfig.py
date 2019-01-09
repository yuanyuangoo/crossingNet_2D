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
h36m_base_path = '/media/a/Portable/h36m/'
# h36m_base_path = 'h36m/'

# h36m_base_path = 'D:/datasets/h36m'
# cache_base_path = './cache/data/'
h36m_base_path = '/media/a/Portable/h36m/'
# path to save the network parameter, intermediate outputs and test result
model_dir = os.path.join(h36m_base_path, 'cache/model/')
cache_base_path = os.path.join(h36m_base_path, 'cache/data/')
ape_base_path='/media/a/Portable/ape/APE/'
# path to the pretrained_model of depthGAN and poseVAE
gan_pretrain_path = os.path.join(
    model_dir, 'Image_gan', '%s_dummy/params/' % dataset)
p2i_pretrain_path = os.path.join(
    model_dir, 'p2i_gan', '%s_dummy/params/' % dataset)
vae_pretrain_path = os.path.join(
    model_dir, 'pose_vae', '%s_dummy/params/' % dataset)
gan_Render_pretrain_path = os.path.join(
    model_dir, 'gan_Render', '%s_dummy/params/' % dataset)
Forward_Render_pretrain_path = os.path.join(
    model_dir, 'Forward_Render', '%s_dummy/params/' % dataset)
p2p_pretrain_path = os.path.join(
    model_dir, 'p2p_gan', '%s_dummy/params/' % dataset)
pganR_pretrain_path = os.path.join(
    model_dir, 'pganR', '%s_dummy/params/' % dataset)
p2pr_pretrain_path = os.path.join(
    model_dir, 'p2pr_gan', '%s_dummy/params/' % dataset)
vnect_pretrain_path=os.path.join(
    model_dir, 'vnect', '%s_dummy/params/' % dataset)
# print('globalConfig.dataset', dataset)
