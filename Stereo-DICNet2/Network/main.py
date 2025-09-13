import torch
import time
from UniDIC.network import UniDIC
from datasets.SpeckleDataset import SpeckleDataset,Normalization
from utils.util import save_checkpoint,AverageMeter
from utils.experiment import tensor2float,adjust_learning_rate
from loss.loss import flow_loss_func, disp_loss_func,flow_loss_func_test,disp_loss_func_test
import torchvision.transforms as transforms
import gc
import os
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

# model: learnable parameters
feature_channels = 128
num_scales = 1
upsample_factor = 8
num_head = 1
ffn_dim_expansion = 4
num_transformer_layers = 6 #6
reg_refine = True #True #False
task = "flow"

# load parameters
start_epoch = 0
epochs = 1000
batch_size = 10  #20 16
test_batch_size = 10  #20
num_workers = 16 #16
summary_freq = 20
MULTI_GPU = 'True'


save_path = 'result/xxxxxxxxxx'  
resume = False
loadckpt = False   #result/checkpoint_{epoch_id}.ckpt
transform = transforms.Compose([Normalization()])

#dataset, dataloader
train_data = SpeckleDataset(
     csv_file='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
     root_dir='xxxxxxxxxxxxxxxxxxxxxx', transform=transform)
test_data = SpeckleDataset(csv_file='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
                            root_dir='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', transform=transform)

print('{} samples found, {} train samples and {} test samples '.format(len(test_data) + len(train_data),
                                                                          len(train_data),len(test_data)))

#load dataset
train_loader = torch.utils.data.DataLoader(
        train_data , batch_size= batch_size,
        num_workers=num_workers, pin_memory=True, shuffle=True),

val_loader = torch.utils.data.DataLoader(
        test_data, batch_size= test_batch_size,
        num_workers=num_workers, pin_memory=True, shuffle=True),

# model, optimizer
model = UniDIC(feature_channels=feature_channels,
                     num_scales=num_scales,
                     upsample_factor=upsample_factor,
                     num_head=num_head,
                     ffn_dim_expansion=ffn_dim_expansion,
                     num_transformer_layers=num_transformer_layers,
                     reg_refine=reg_refine,
                     task=task),


model =model[0]
model = model.to(device)
model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.004, betas=(0.9, 0.999)),
optimizer = optimizer[0]
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.0002, 90000,
                                               pct_start=0.3, cycle_momentum=False, anneal_strategy='linear')


# continue training the model
if resume:
        # find all checkpoints file and sort according to epoch id
        # all_saved_ckpts = [fn for fn in os.listdir(save_path) if fn.endswith(".ckpt")]
        # all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = 'result/xx/best_train_model.ckpt'        
        print("loading the lastest model in save_path: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'], strict=False)
        start_epoch = state_dict['epoch'] + 1

# load the weights from a specific checkpoint
elif loadckpt:
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model_dict = model.state_dict()
        pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        print("start at epoch {}".format(start_epoch))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 100:
        lr = 0.0005  
    elif epoch <= 300:
        lr = 0.0004   
    elif epoch <= 500:
        lr = 0.0003  
    elif epoch <= 600:
        lr = 0.0001  
    elif epoch <= 700:
        lr = 0.00008  
    elif epoch <= 800:
        lr = 0.00005     
   elif epoch <= 900:
        lr = 0.00003   
    else:
        lr = 0.00001 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def train(train_loader, model, optimizer, epoch, scheduler):
def train(train_loader, model, optimizer, min_train_loss, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_size = len(train_loader[0])
    # switch to train mode
    model.train()
    model.training = True
    end = time.time()

    # measure data loading time
    data_time.update(time.time() - end)

    total_train_loss = 0
    for batch_idx, sample in enumerate(train_loader[0]):
            disp_gt = sample['Dispx'].to(device) 
            u = sample['U'].to(device).unsqueeze(1)
            v = sample['V'].to(device).unsqueeze(1)
            flow_gt = torch.cat([u,v],1).to(device) 
            L0 = sample['L0'].float().to(device)
            R0 = sample['R0'].float().to(device)
            L1 = sample['L1'].float().to(device)
            results_dict = model(L0, R0, L1)

            flow_preds = results_dict[0]   
            disp_preds = results_dict[1]

            loss_flow = flow_loss_func(flow_preds, flow_gt)    
            #loss_disp = disp_loss_func(disp_preds, disp_gt)

            loss = 0.6 * loss_flow + 0.4 * loss_disp  
           
            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_last_lr()
            #print('epoch:{0}\t batch:{1}'.format(epoch, batch_idx))
            print('epoch:{0}\t batch:{1}\t lr{2},'.format(epoch, batch_idx,lr))
            print('Epoch: [{0}][{1}/{2}]\t Loss {3}'.format(epoch, batch_idx, epoch_size, loss))

    train_loss = total_train_loss / len(train_loader[0])
    return train_loss


def validate(val_loader, model, min_test_loss,epoch):
    total_test_loss = 0
    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()
    # switch to evaluate mode
    model.eval()
    model.training=False
    end = time.time()
    for batch_idx, sample in enumerate(val_loader[0]):
        disp_gt = sample['Dispx'].to(device)  # n*1*256*256
        u = sample['U'].to(device).unsqueeze(1)
        v = sample['V'].to(device).unsqueeze(1)
        flow_gt = torch.cat([u, v], 1).to(device)
        L0 = sample['L0'].float().to(device)
        R0 = sample['R0'].float().to(device)
        L1 = sample['L1'].float().to(device)
        results_dict = model(L0, R0, L1)
        flow_preds = results_dict[0]
        disp_preds = results_dict[1]

        loss_flow = flow_loss_func_test(flow_preds, flow_gt)
        loss_disp = disp_loss_func_test(disp_preds, disp_gt)

        #loss = loss_flow + 0.9 * loss_disp
        loss = loss_flow
        #loss = loss_disp

        total_test_loss += loss
        print('Test: [{0}/{1}]\t Time {2}\t loss_flow {3}\t loss_disp {4}'
                   .format(batch_idx, len(val_loader), batch_time, loss_flow, loss_disp))
        # print('Test: [{0}/{1}]\t Time {2}\t loss_flow {3}'
        #       .format(batch_idx, len(val_loader), batch_time, loss_flow))


    test_loss = total_test_loss/len(val_loader[0])

    return test_loss  #.data.cpu()


def main():

    trainloss = []
    testloss = []
    min_train_loss = 20
    min_test_loss = 20
    for epoch_idx in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch_idx)
        # train_loss = train(train_loader, model, optimizer, epoch_idx, scheduler)
        train_loss = train(train_loader, model, optimizer, min_train_loss,epoch_idx)    
        print('Epoch {}/{}, train loss = {:.3f}'.format(epoch_idx, epochs, train_loss))

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            print("save model")
            checkpoint_data_0 = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        gc.collect()

        train_log_filename = "train_log.txt"
        train_log_filepath = os.path.join(save_path, train_log_filename)
        train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch_idx,
                                                  loss_str=" ".join(["{}".format(train_loss)]))


        trainloss.append(train_loss)
        with torch.no_grad():
          test_loss = validate(val_loader, model, min_test_loss,epoch_idx)
          testloss.append(test_loss)
          if test_loss < min_test_loss:
              min_test_loss = test_loss
              print("save model")
              checkpoint_data_1 = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

          gc.collect()
          test_log_filename = "test_log.txt"
          test_log_filepath = os.path.join(save_path, test_log_filename)
          test_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
          to_write = test_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                    epoch=epoch_idx,
                                                    loss_str=" ".join(["{}".format(test_loss)]))

        if (epoch_idx + 1) % 50 == 0:

            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            # id_epoch = (epoch_idx + 1) % 100
            #torch.save(checkpoint_data, "{}/1_checkpoint_{:0>3}.ckpt".format(save_path, epoch_idx))

        gc.collect()



    x1 = range(0, 1000)
    x2 = range(0, 1000)
    y1 = trainloss
    y2 = testloss
    plt.subplot(2, 1, 1)
    plt.plot(torch.tensor(x1,device='cpu'), torch.tensor(y1,device='cpu'), '.')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.subplot(2, 1, 2)
    plt.plot(torch.tensor(x2, device='cpu'), torch.tensor(y2, device='cpu'), '.')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("test_loss.jpg")


if __name__ == '__main__':
   main()




