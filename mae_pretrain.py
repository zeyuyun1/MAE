import os
import argparse
import math
import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model import *
from model_origin import *
from utils import setup_seed


def eval_model(model):
    fea_ls = []
    i_ls = []
    for x,i in train_dl:
    #     print(i)
        x = x.to(device)
        features = model.module.encoder.feature_extract(x)
        features = features.cpu().detach()
        fea_ls.append(features)
        i_ls.append(i)
    #     break
    X = torch.cat(fea_ls).numpy()
    y = torch.cat(i_ls).numpy()
    fea_ls_test = []
    i_ls_test = []
    for x,i in test_dl:
        x = x.to(device)
        features = model.module.encoder.feature_extract(x)
        features = features.cpu().detach()
        fea_ls_test.append(features)
        i_ls_test.append(i)
    X_test = torch.cat(fea_ls_test).numpy()
    y_test = torch.cat(i_ls_test).numpy()
    neigh = KNeighborsClassifier(n_neighbors=20,metric='cosine')
    neigh.fit(X, y)
    y_test_hat = neigh.predict(X_test)
    acc_score = accuracy_score(y_test,y_test_hat)
#     print(acc_score)
    return acc_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--emb_dim', type=int, default=192)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='vit-t-mae')
    
    args = parser.parse_args()

    setup_seed(args.seed)
    
    model_path = "{}_h_dim_{}_mlp_ratio_{}_patch_size_{}_batch_size_{}.pt".format(args.model_name,args.emb_dim,args.mlp_ratio,args.patch_size,args.batch_size)


    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    
    test_dl = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=4)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    wandb.init(project="mae_train_cifar10",name = model_path[:-3])
    wandb.config.update(args)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MAE_ViT_origin(mask_ratio=args.mask_ratio)
#     model = MAE_ViT(patch_size=args.patch_size,mask_ratio=args.mask_ratio,mlp_ratio=args.mlp_ratio,emb_dim=args.emb_dim)
    model = torch.nn.DataParallel(model).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        if e%50==0:
            torch.save(model.module, model_path)
            
        if e%100==0:
            eval_acc = eval_model(model)
            wandb.log({'eval_acc': eval_acc})
            
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
#         print(f'In epoch {e}, average traning loss is {avg_loss}.')
        wandb.log({'epoch':e, 'mae_loss': avg_loss, 'lr':lr_scheduler.get_last_lr()[0]})

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model.module, model_path)