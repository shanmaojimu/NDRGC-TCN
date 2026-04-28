import torch
import math
import sys
import time
import datetime
from .tools import AverageMeter, accuracy
from thop import profile
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score


# 定义非对称焦点损失（ASL）
def asymmetric_focal_loss(predicts, targets, alpha=0.25, gamma=2.0):
    """
    计算非对称焦点损失 (ASL)
    :param predicts: 模型的预测输出 (batch_size, num_classes)
    :param targets: 真实标签 (batch_size,)
    :param alpha: 平衡正负类的系数
    :param gamma: 调节难易样本的因子
    :return: 计算的ASL损失
    """
    # 对每个样本计算其预测概率
    probs = F.softmax(predicts, dim=1)

    # 获取每个样本真实类别的预测概率
    p_t = probs.gather(1, targets.unsqueeze(1))  # (batch_size, 1)

    # 计算非对称焦点损失
    loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t + 1e-8)  # 防止log(0)

    return loss.mean()  # 返回均值作为损失值


# 修改train_one_epoch函数，加入混合损失
def train_one_epoch(epoch, iterator, data, model, device, optimizer, criterion, tensorboard, start_time, args, rta,
                    theta=0.5):
    print('--------------------------Start training at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)

    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    criterion = criterion.to(device)

    model.train()
    data, data_labels = data
    steps = data.shape[0] // args.batch_size + 1 if data.shape[0] % args.batch_size else data.shape[
                                                                                             0] // args.batch_size
    step = 0

    # 存储每个batch的acc和loss
    batch_losses = []

    for features, labels in iterator:
        features, labels = rta(features, labels)
        features = features.to(device)
        labels = labels.to(device)

        predicts, _, node_weights = model(features)

        # 计算交叉熵损失（CE）
        ce_loss = criterion(predicts, labels)

        # 计算非对称焦点损失（ASL）
        asl_loss = asymmetric_focal_loss(predicts, labels)

        # 总损失是交叉熵损失和非对称焦点损失的加权和
        loss = ce_loss + theta * asl_loss

        acc = accuracy(predicts.detach(), labels.detach())[0]
        # 保存batch的loss
        batch_losses.append(loss.item())

        dict_log['loss'].update(loss.item(), len(features))
        dict_log['acc'].update(acc.item(), len(features))
        optimizer.zero_grad()
        loss.backward()
        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"Warning: NaN or Inf in gradients at epoch {epoch + 1}, step {step}")
                    optimizer.zero_grad()
                    break
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if total_norm > args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

        param_has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN or Inf in parameter {name}")
                param_has_nan = True

        if not param_has_nan:
            optimizer.step()

        all_steps = epoch * steps + step + 1
        if 0 == (all_steps % args.print_freq):
            lr = list(optimizer.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]
            print_information = 'id:{}   time consumption:{}    epoch:{}/{}  lr:{}    '.format(
                args.id, et, epoch + 1, args.epochs, lr)
            for key, value in dict_log.items():
                loss_info = "{}(val/avg):{:.3f}/{:.3f}  ".format(key, value.val, value.avg)
                print_information = print_information + loss_info
                tensorboard.add_scalar(key, value.val, all_steps)
            print(print_information)
        step = step + 1

    # 计算acc和loss的min, mean, max
    min_loss, mean_loss, max_loss = min(batch_losses), sum(batch_losses) / len(batch_losses), max(batch_losses)
    avg_train_loss = dict_log['loss'].avg

    print('--------------------------End training at epoch:{}--------------------------'.format(epoch + 1))
    return node_weights, (min_loss, mean_loss, max_loss)


def evaluate_one_epoch(epoch, iterator, data, model, device, criterion, tensorboard, args, start_time,rta):
    print('--------------------------Start evaluating at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    model.eval()
    data, data_labels = data
    step = 0
    start_time = time.time()

    all_preds = []
    all_labels = []
    for features, labels in iterator:
        features,labels = rta(features,labels)
                # x = x.permute(0,2,1,3)
        # x = x.contiguous().view(x.shape[0],x.shape[1],-1)
        # features = features.permute(0,2,1,3)
        # features = features.contiguous().view(features.shape[0],features.shape[1],-1)

        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predicts,_,_,_ = model(features)
            loss = criterion(predicts, labels)
        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))

        all_preds.append(torch.argmax(predicts, dim=1).cpu())
        all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro')

    end_time = time.time()
    now_time = time.time() - start_time
    et = str(datetime.timedelta(seconds=now_time))[:-7]
    print_information = 'time consumption:{}    epoch:{}/{}   '.format(et, epoch + 1, args.epochs, len(data))

    for key, value in dict_log.items():
        loss_info = "{}(avg):{:.3f} ".format(key, value.avg)
        print_information = print_information + loss_info
        tensorboard.add_scalar(key, value.val, epoch)

    duration_time = '    ' + str(end_time - start_time)
    print(print_information+duration_time)
    print_information += f"Kappa: {kappa:.3f}  F1: {f1:.3f}  Recall: {rec:.3f}  Precision: {prec:.3f}"
    print('--------------------------Ending evaluating at epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg, kappa, f1, rec, prec


def train(model, device, train_loader, optimizer, epoch, scheduler=None, max_grad_norm=0.5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"Warning: NaN or Inf in input data at batch {batch_idx}")
            continue

        optimizer.zero_grad()
        output = model(data.float())

        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: NaN or Inf in model output at batch {batch_idx}")
            nan_count += 1
            if nan_count > 5:
                print("Too many NaN outputs, reducing learning rate")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                nan_count = 0
            continue

        loss = F.nll_loss(output, target)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss at batch {batch_idx}")
            continue

        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"Warning: NaN or Inf in gradients at batch {batch_idx}")
                    optimizer.zero_grad()
                    break
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if total_norm > max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        param_has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN or Inf in parameter {name}")
                param_has_nan = True

        if not param_has_nan:
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t'
                  f'Grad Norm: {total_norm:.6f}')

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0

    print(f'Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy
