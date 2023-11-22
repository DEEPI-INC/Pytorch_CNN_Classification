from f_classification import *

dst_tr = '//192.168.0.70/deepi-nas/[1]_AI_model/dataset_2023/[2]_kjs_study/classification_regression_dataset/train'
dst_te = '//192.168.0.70/deepi-nas/[1]_AI_model/dataset_2023/[2]_kjs_study/classification_regression_dataset/test'

kls = os.listdir(dst_tr)

size, ratio, batch, lr, epoch = 224, 0.7, 10, 1e-5, 10

k_lists_tr = data_prepare(dst_tr, kls, size, exts = ('.jpg', '.png'))
k_lists_te = data_prepare(dst_te, kls, size, exts = 'all')

tr_loader, val_loader = dataloader(k_lists_tr, ratio, batch, mode='train')
te_loader = dataloader(k_lists_te, None, 1, mode='test')

model = set_model(kls, size,  model='vgg')

model, result_tr, result_val = train(tr_loader, val_loader, epoch, lr, model)

# torch.save(model, 'my_model.pth')
# model = torch.load('my_model.pth')

result_te, tl_ta, cf_matrix = test(te_loader, model)
