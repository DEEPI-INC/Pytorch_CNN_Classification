# 1. Prepare
## 1-1. Dataset Download
[Sample Dataset](https://drive.google.com/drive/folders/1LxU4ZwYH4VW1z0jtsFZUjOgSqFdU0U_9?usp=drive_link)

## 1-2. Run
### (1) Dataset
* Set Path
```python
dst_tr = 'path/for/train/dataset'
dst_te = 'path/for/test/dataset'
```

* Set class idx and params
```python
kls = os.listdir(dst_tr) # set class idx
size, ratio, batch, lr, epoch = 224, 0.7, 10, 1e-5, 10
```

* Dataset creation and validation (set extentions)
```python
k_lists_tr = data_prepare(dst_tr, kls, size, exts = 'all')
k_lists_te = data_prepare(dst_te, kls, size, exts = ('.jpg', '.png'))
```

* Set Dataloader (batch size of test Dataloader = 1)
```python
tr_loader, val_loader = dataloader(k_lists_tr, ratio, batch, mode='train')
te_loader = dataloader(k_lists_te, None, 1, mode='test')
```

* Train
```python
model = set_model(kls, size,  model='vgg')
model, result_tr, result_val = train(tr_loader, val_loader, epoch, lr, model)
```
![image](https://github.com/DEEPI-INC/pytorch_classification/assets/148835003/2608644f-f1c3-42fb-95f3-aec08b66311b)

* Evaluation
```python
# torch.save(model, 'my_model.pth')
# model = torch.load('my_model.pth')
result_te, tl_ta, cf_matrix = test(te_loader, model)
```


