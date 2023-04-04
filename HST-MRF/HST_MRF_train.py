import argparse
from eval_function import *
from tqdm import tqdm
from HST_MRF_utils import *
from torch.autograd import Variable
import transformers as tfs
import gc
from torch.cuda.amp import GradScaler
import logging
from skin_dataloader import *

torch.cuda.empty_cache()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# 损失函数 加权的IOU和加权BCE
def structure_loss(pred, mask):
    '''
    :param pred: 模型预测出来的数值型二值图（或者是经过了 head 之后的二值图）
    :param mask: 真实的二值标签
    :return:
    '''
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou+tversky_loss(pred, mask)).mean()

def tversky_loss(pred, mask):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1,1)
    mask = mask.contiguous().view(-1,1)
    mask_pos = torch.sum(mask*pred)
    false_neg = torch.sum(mask*(1-pred))
    false_pos = torch.sum((1-mask)*pred)
    alpha = 0.7
    return 1-mask_pos/(mask_pos+alpha*false_neg+(1-alpha)*false_pos)



def adjust_lr(optimizer, epoch, max_lr,decay_epoch=220):
    decay_rate = (max_lr/1e-5)**(1/109.5)
    decay = decay_rate ** (epoch / decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def eval_net(net, loader, device):
    net.eval()
    total_iou, total_pre, total_rec, total_dice = 0, 0, 0, 0

    N = 0
    with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for img, img_bi in loader:
            img, img_bi = img.to(device,dtype = torch.float32), img_bi.to(device, dtype=torch.float32)
            with torch.no_grad():
                img_pred, _, _ = net(img)
            pred = torch.sigmoid(img_pred)
            del img_pred

            eval = evaluate(pred, img_bi)
            iou= eval.iou()
            pre = eval.precision()
            rec = eval.recall()
            dice = eval.dice()

            del eval
            total_iou += iou
            total_pre += pre
            total_rec += rec
            total_dice += dice

            N += img_bi.size()[0]

            pbar.update()

    return total_pre / N, total_rec / N, total_iou / N, total_dice / N

def train_net(net, device, args):

    train_loader = get_loader2(args.train_data, args.train_mask, batchsize=args.batch_size, trainsize=args.img_size,
                              augmentation=True)
    test_loader = get_loader2(args.test_data, args.test_mask, batchsize=args.test_batch_size, trainsize=args.img_size,
                            augmentation=True)

    filename = args.states
    logger = get_logger(filename)
    logger.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {args.img_size}
        Device:          {device.type}
        Optimizer:       {args.optimizer}
        Weight_decay:    {args.weight_decay}
        Channels:        {args.channels}
        Grad_accumul:    {args.grad_num}
        augmentation:    True
        Dropout:         {args.dropout}
        
    ''')
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = tfs.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps, num_cycles=0.5)
    scaler = GradScaler()
    # best_model=0.913

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0
        train_dice, train_iou = 0, 0
        count = 0
        print(f'Epoch {epoch + 1}/{args.epochs}')
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', leave=False) as bar:
            for img, img_bi in train_loader:
                img = Variable(img).to(device)
                img_bi = Variable(img_bi).to(device)

                with torch.cuda.amp.autocast():
                    masks_pred, l1, l2 = net(img)
                    loss = 0.6 * structure_loss(masks_pred, img_bi)\
                           + 0.2 * structure_loss(l1,img_bi)\
                           + 0.2 * structure_loss(l2, img_bi)
                    epoch_loss += loss.item()
                    loss = loss / args.grad_num
                scaler.scale(loss).backward()
                if (count % args.grad_num)==0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                bar.update()
                eval = evaluate(torch.sigmoid(masks_pred), img_bi)
                dice = eval.dice()
                iou = eval.iou()
                train_dice += dice
                train_iou += iou

                count += 1
                # break
        train_loss = epoch_loss/len(train_loader)
        print('-----------------Train-------------------------')
        print(f'loss: {train_loss}, train dice: {train_dice/count}, train iou: {train_iou/count}')

        gc.collect()
        torch.cuda.empty_cache()
        pre, rec, iou, dice = eval_net(net, test_loader, device)
        print('-----------------Test1------------------------')
        logger.info('epoch: {}, dice: {:.3f}, iou: {:.3f}, rec: {:.3f}, pre: {:.3f}'.format(epoch + 1, dice.item(), iou.item(), rec.item(), pre.item()))
        # if best_model<dice:
        #     torch.save(model.state_dict(), f'model/HST-MRF-CVC-{dice:.3f}.pt')
        #     best_model=dice

        del pre, rec, iou, dice, masks_pred, l1, l2, train_dice, train_loss, train_iou

        gc.collect()
        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=10, help='every n epochs decay learning rate')

    parser.add_argument('--train_data', type=str, default='CVC-ClinicDB/train/image/',choices=['Kvasir/Kvasir_SEG_Train/image/',
                                                                                                    'CVC-ClinicDB/train/image/'],
                        help='Load model from a .pth file')
    parser.add_argument('--train_mask', type=str, default='CVC-ClinicDB/train/mask/', choices=['Kvasir/Kvasir_SEG_Train/mask/',
                                                                                                    'CVC-ClinicDB/train/mask/'],
                        help='Load model from a .pth file')
    parser.add_argument('--test_data', type=str, default='CVC-ClinicDB/test/image/', choices=['Kvasir/Kvasir_SEG_Valid/image/',
                                                                                                    'CVC-ClinicDB/test/image/'],
                        help='Load model from a .pth file')
    parser.add_argument('--test_mask', type=str, default='CVC-ClinicDB/test/mask/',choices=['Kvasir/Kvasir_SEG_Valid/mask/',
                                                                                                    'CVC-ClinicDB/test/mask/'],
                        help='Load model from a .pth file')

    parser.add_argument('--states', type=str, default='预测.log',
                        choices=['log文件/norm-Flatten/Kavsir.log',
                                 'log文件/No-C-A/Kvasir.log',
                                 'log文件/norm-C-A/Kvasir.log'])
    parser.add_argument('--patience', type=int, default=10, help='patience for rlp')
    parser.add_argument('--factor', type=float, default=0.1, help='factor for rlp')

    parser.add_argument('--img_size', type=int, default=256, choices='512,384，,256')
    parser.add_argument('--grad_num', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--head_dim', type=int, default=16)
    parser.add_argument('--num_head', type=int, default=4)
    parser.add_argument('--num_groups', type=int, default=32)
    parser.add_argument('--channels', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42, help='定义随机种子')  # 42
    parser.add_argument('--optimizer', type=str, default='AdamW',choices=['Adam', 'SGD', 'AdamW'])

    return parser.parse_args()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_sate_dict = torch.load('model/HST-MRF-CVC-0.920.pt')
    model = model(args)
    # model.load_state_dict(model_sate_dict)
    model = model.to(device)
    train_net(model, device, args)
